"""
Abliteration: 移除开源大模型的拒绝方向（纯 transformers 实现）

核心流程：
  1. 用有害/无害 prompt 收集所有层的 resid_pre 激活值
  2. 计算每层的拒绝方向候选
  3. 通过推理时 hook 介入测试每个候选（关键步骤！）
  4. 选择最佳方向，永久正交化权重

用法:
    python abliterate_nous.py --model /16tb/llm/Qwen2.5-1.5B-Instruct --output ./qwen1.5b-abliterated
"""

import argparse
import torch
import gc
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_grad_enabled(False)


# ============================================================
# 数据
# ============================================================

def get_prompts():
    harmful = load_dataset("mlabonne/harmful_behaviors")["train"]["text"]
    harmless = load_dataset("mlabonne/harmless_alpaca")["train"]["text"]
    return harmful, harmless


def format_prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")


# ============================================================
# 激活值收集（resid_pre = 层的输入）
# ============================================================

def collect_resid_pre(model, tokenizer, prompts, n_samples=128):
    """收集每层的 resid_pre（即层输入）的 last-token 激活值"""
    n_layers = model.config.num_hidden_layers
    n_samples = min(n_samples, len(prompts))

    layer_acts = {i: [] for i in range(n_layers)}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs=None):
            # register_forward_pre_hook 的 args[0] 是 hidden_states
            hidden = args[0] if isinstance(args, tuple) else args
            if isinstance(hidden, torch.Tensor):
                layer_acts[layer_idx].append(hidden[:, -1, :].detach().cpu())
        return hook_fn

    for i in range(n_layers):
        handle = model.model.layers[i].register_forward_pre_hook(make_hook(i))
        handles.append(handle)

    for i in tqdm(range(n_samples)):
        tokens = format_prompt(tokenizer, prompts[i]).to(model.device)
        with torch.no_grad():
            model(tokens)

    for h in handles:
        h.remove()

    return {i: torch.cat(v, dim=0) for i, v in layer_acts.items() if len(v) > 0}


# ============================================================
# 拒绝方向计算
# ============================================================

def compute_refusal_directions(harmful_by_layer, harmless_by_layer):
    """计算每层的拒绝方向，返回按 abs(mean) 排序的列表"""
    directions = []
    for layer_idx in sorted(harmful_by_layer.keys()):
        if layer_idx not in harmless_by_layer:
            continue
        h_mean = harmful_by_layer[layer_idx].float().mean(dim=0)
        s_mean = harmless_by_layer[layer_idx].float().mean(dim=0)
        refusal_dir = h_mean - s_mean
        refusal_dir = refusal_dir / refusal_dir.norm()
        strength = abs(refusal_dir.mean()).item()
        directions.append((layer_idx, refusal_dir, strength))

    # 过滤掉 NaN 方向，按 strength 排序
    directions = [(l, d, s) for l, d, s in directions if not (torch.isnan(d).any() or s != s)]
    directions.sort(key=lambda x: x[2], reverse=True)
    return directions


# ============================================================
# 推理时介入测试（关键步骤）
# ============================================================

def test_direction_with_hook(model, tokenizer, refusal_dir, test_prompts, max_new_tokens=32):
    """
    在推理时用 hook 从所有层的残差流中减去 refusal_dir 的投影，
    然后生成回复，检查是否仍然拒绝。
    返回 (score, responses) —— score 越高说明拒绝越少。
    """
    n_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device
    direction = refusal_dir.to(device, dtype=torch.bfloat16)

    handles = []

    def make_ablation_hook(layer_idx):
        def hook_fn(module, args):
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                return args
            # 从 hidden states 中减去 refusal_dir 方向的投影
            d = direction.to(hidden.device)
            proj = torch.einsum("...d, d -> ...", hidden, d).unsqueeze(-1) * d
            new_hidden = hidden - proj
            return (new_hidden,) + args[1:]
        return hook_fn

    for i in range(n_layers):
        handle = model.model.layers[i].register_forward_pre_hook(make_ablation_hook(i))
        handles.append(handle)

    responses = []
    for prompt_text in test_prompts:
        tokens = format_prompt(tokenizer, prompt_text).to(model.device)
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        response = tokenizer.decode(output[0][tokens.shape[1]:], skip_special_tokens=True)
        responses.append(response)

    for h in handles:
        h.remove()

    # 评分：不拒绝 + 输出连贯（非乱码）
    refusal_keywords = [
        "I cannot", "I can't", "I'm sorry", "I apologize",
        "As an AI", "not able to", "I'm not going to",
        "I must decline", "unable to assist", "against my",
        "I won't", "I refuse", "不能", "抱歉", "无法"
    ]

    def is_coherent(text):
        """检查输出是否连贯（非乱码）"""
        if len(text.strip()) < 5:
            return False
        # 检查是否全是重复字符
        unique_chars = set(text.strip()[:50])
        if len(unique_chars) <= 3:
            return False
        # 检查是否有足够多的字母/汉字
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count < len(text) * 0.3:
            return False
        return True

    score = sum(
        1 for r in responses
        if not any(kw.lower() in r.lower() for kw in refusal_keywords) and is_coherent(r)
    )
    return score, responses


# ============================================================
# 权重正交化
# ============================================================

def orthogonalize_weight(W, direction):
    """从权重矩阵中移除 direction 方向分量"""
    direction = direction.to(W.device, W.dtype)
    if W.dim() == 1:
        return W - (W @ direction) * direction
    if W.shape[0] == direction.shape[0] and W.shape[1] != direction.shape[0]:
        # o_proj / down_proj: [d_model, d_in], direction 在 dim 0
        proj_coeffs = direction @ W
        return W - torch.outer(direction, proj_coeffs)
    else:
        # embed_tokens: [vocab_size, d_model], direction 在 dim 1
        proj_coeffs = W @ direction
        return W - proj_coeffs.unsqueeze(1) * direction.unsqueeze(0)


def abliterate_model(model, refusal_dir):
    """永久正交化权重：embedding + 所有层的 o_proj 和 down_proj"""
    n_layers = model.config.num_hidden_layers

    print("  正交化 embedding...")
    model.model.embed_tokens.weight.data = orthogonalize_weight(
        model.model.embed_tokens.weight.data, refusal_dir
    )

    for layer_idx in tqdm(range(n_layers), desc="  正交化各层权重"):
        layer = model.model.layers[layer_idx]
        layer.self_attn.o_proj.weight.data = orthogonalize_weight(
            layer.self_attn.o_proj.weight.data, refusal_dir
        )
        layer.mlp.down_proj.weight.data = orthogonalize_weight(
            layer.mlp.down_proj.weight.data, refusal_dir
        )

    return model


# ============================================================
# 生成测试
# ============================================================

def test_model(model, tokenizer, prompts, max_new_tokens=128):
    results = []
    for prompt_text in prompts:
        tokens = format_prompt(tokenizer, prompt_text).to(model.device)
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        response = tokenizer.decode(output[0][tokens.shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt_text, "response": response})
    return results


# ============================================================
# 主流程
# ============================================================

def main(args):
    print(f"=== Abliteration: {args.model} ===\n")

    print("[1/7] 加载数据集...")
    harmful_prompts, harmless_prompts = get_prompts()

    print(f"[2/7] 加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    # 基线测试
    test_prompts = harmful_prompts[:4]
    print("[3/7] 基线测试...")
    baseline = test_model(model, tokenizer, test_prompts)
    print("\n--- 基线回复 ---")
    for r in baseline:
        print(f"  Q: {r['prompt'][:80]}...")
        print(f"  A: {r['response'][:200]}")
    print()

    # 收集激活值
    print(f"[4/7] 收集 resid_pre 激活值 (n={args.n_samples})...")
    print("  有害 prompt...")
    harmful_by_layer = collect_resid_pre(model, tokenizer, harmful_prompts, n_samples=args.n_samples)
    print("  无害 prompt...")
    harmless_by_layer = collect_resid_pre(model, tokenizer, harmless_prompts, n_samples=args.n_samples)

    # 计算候选方向
    print("[5/7] 计算拒绝方向候选...")
    directions = compute_refusal_directions(harmful_by_layer, harmless_by_layer)

    del harmful_by_layer, harmless_by_layer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  共 {len(directions)} 个候选方向, Top-5 强度:")
    for layer_idx, _, strength in directions[:5]:
        print(f"    Layer {layer_idx}: abs(mean) = {strength:.6f}")

    # 推理时介入测试 —— 找到真正有效的方向
    n_eval = min(args.eval_n, len(directions))
    print(f"\n[6/7] 推理时介入测试 (top-{n_eval} 候选)...")
    eval_prompts = harmful_prompts[:4]  # 用 4 个 prompt 评估

    best_score = -1
    best_idx = 0
    best_responses = []

    for rank, (layer_idx, refusal_dir, strength) in enumerate(tqdm(directions[:n_eval])):
        score, responses = test_direction_with_hook(
            model, tokenizer, refusal_dir, eval_prompts, max_new_tokens=64
        )
        if score > best_score:
            best_score = score
            best_idx = rank
            best_responses = responses
        if rank < 5 or score > 0:
            print(f"  候选 #{rank} (Layer {layer_idx}, strength={strength:.6f}): "
                  f"score={score}/{len(eval_prompts)}")

    best_layer_idx, best_dir, best_strength = directions[best_idx]
    print(f"\n  最佳: 候选 #{best_idx} (Layer {best_layer_idx}), score={best_score}/{len(eval_prompts)}")
    print("  介入测试样例:")
    for i in range(min(4, len(best_responses))):
        print(f"    Q: {eval_prompts[i][:60]}...")
        print(f"    A: {best_responses[i][:150]}")
        print()

    if best_score == 0:
        print("  警告: 所有候选方向都未能移除拒绝！尝试组合 top-3 方向...")
        # 尝试组合多个方向
        combined_dir = torch.zeros_like(directions[0][1])
        for _, d, _ in directions[:3]:
            combined_dir += d
        combined_dir = combined_dir / combined_dir.norm()
        score, responses = test_direction_with_hook(
            model, tokenizer, combined_dir, eval_prompts, max_new_tokens=64
        )
        print(f"  组合方向 score: {score}/{len(eval_prompts)}")
        if score > best_score:
            best_dir = combined_dir
            best_score = score
            best_responses = responses
            print("  组合方向效果更好！使用组合方向。")
        for i in range(min(4, len(responses))):
            print(f"    Q: {eval_prompts[i][:60]}...")
            print(f"    A: {responses[i][:150]}")
            print()

    # 正交化权重
    print("[7/7] 正交化权重（永久修改）...")
    model = abliterate_model(model, best_dir)

    # 验证
    print("\n验证...")
    abliterated = test_model(model, tokenizer, test_prompts)
    print("\n--- 消融后回复 ---")
    for i, r in enumerate(abliterated):
        print(f"  Q: {r['prompt'][:80]}...")
        print(f"  A(before): {baseline[i]['response'][:150]}")
        print(f"  A(after):  {r['response'][:150]}")
        print()

    # 保存
    if args.output:
        print(f"保存模型到 {args.output}...")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        torch.save(best_dir, Path(args.output) / "refusal_direction.pt")
        results = {"baseline": baseline, "abliterated": abliterated}
        with open(Path(args.output) / "comparison.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"完成! 模型已保存到 {args.output}")
    else:
        print("完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abliteration")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--eval-n", type=int, default=20, help="评估 top-N 个候选方向")
    args = parser.parse_args()
    main(args)
