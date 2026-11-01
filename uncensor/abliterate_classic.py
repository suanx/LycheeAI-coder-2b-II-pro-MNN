"""
Abliteration: 移除开源大模型的拒绝方向
基于 Arditi et al. "Refusal in LLMs is mediated by a single direction"
和 mlabonne 的 HuggingFace 教程

使用 TransformerLens 经典方法实现。
适用于 Llama 架构模型（Gemma、Llama、Mistral 等）。

用法:
    python abliterate_classic.py --model google/gemma-4-12b-it --output ./gemma-4-12b-abliterated
"""

import argparse
import torch
import functools
import einops
import gc

from datasets import load_dataset
from tqdm import tqdm
from torch import Tensor
from typing import List
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float, Int
from collections import defaultdict

torch.set_grad_enabled(False)


# ============================================================
# 1. 数据准备：有害 vs 无害 prompt
# ============================================================

def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]


def get_harmful_instructions():
    dataset = load_dataset("mlabonne/harmful_behaviors")
    return reformat_texts(dataset["train"]["text"]), reformat_texts(dataset["test"]["text"])


def get_harmless_instructions():
    dataset = load_dataset("mlabonne/harmless_alpaca")
    return reformat_texts(dataset["train"]["text"]), reformat_texts(dataset["test"]["text"])


# ============================================================
# 2. Tokenization
# ============================================================

def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids


# ============================================================
# 3. 生成函数
# ============================================================

def _generate_with_hooks(model, tokenizer, tokens, max_tokens_generated=64, fwd_hooks=[]):
    all_tokens = torch.zeros(
        (tokens.shape[0], tokens.shape[1] + max_tokens_generated),
        dtype=torch.long,
        device=tokens.device,
    )
    all_tokens[:, : tokens.shape[1]] = tokens
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_tokens[:, : -max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            all_tokens[:, -max_tokens_generated + i] = next_tokens
    return tokenizer.batch_decode(all_tokens[:, tokens.shape[1] :], skip_special_tokens=True)


def get_generations(model, tokenizer, instructions, fwd_hooks=[], max_tokens_generated=64, batch_size=4):
    generations = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        tokens = tokenize_instructions(tokenizer, instructions=instructions[i : i + batch_size])
        generation = _generate_with_hooks(
            model, tokenizer, tokens,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations


# ============================================================
# 4. 方向消融 Hook
# ============================================================

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = (
        einops.einsum(activation, direction.view(-1, 1), "... d_act, d_act single -> ... single")
        * direction
    )
    return activation - proj


# ============================================================
# 5. 正交化函数（永久修改权重）
# ============================================================

def get_orthogonalized_matrix(matrix, vec):
    proj = einops.einsum(matrix, vec.view(-1, 1), "... d_model, d_model single -> ... single") * vec
    return matrix - proj


# ============================================================
# 主流程
# ============================================================

def main(args):
    print(f"[1/7] 加载数据集...")
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()

    print(f"[2/7] 加载模型: {args.model}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        dtype=torch.bfloat16,
        default_padding_side="left",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    n_inst_train = min(args.n_train, len(harmful_inst_train), len(harmless_inst_train))
    harmful_tokens = tokenize_instructions(tokenizer, instructions=harmful_inst_train[:n_inst_train])
    harmless_tokens = tokenize_instructions(tokenizer, instructions=harmless_inst_train[:n_inst_train])

    print(f"[3/7] 收集激活值 (n={n_inst_train}, batch_size={args.batch_size})...")
    harmful = defaultdict(list)
    harmless = defaultdict(list)

    num_batches = (n_inst_train + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(num_batches)):
        start_idx = i * args.batch_size
        end_idx = min(n_inst_train, start_idx + args.batch_size)

        harmful_logits, harmful_cache = model.run_with_cache(
            harmful_tokens[start_idx:end_idx],
            names_filter=lambda hook_name: "resid" in hook_name,
            device="cpu",
            reset_hooks_end=True,
        )
        harmless_logits, harmless_cache = model.run_with_cache(
            harmless_tokens[start_idx:end_idx],
            names_filter=lambda hook_name: "resid" in hook_name,
            device="cpu",
            reset_hooks_end=True,
        )

        for key in harmful_cache:
            harmful[key].append(harmful_cache[key])
            harmless[key].append(harmless_cache[key])

        del harmful_logits, harmless_logits, harmful_cache, harmless_cache
        gc.collect()
        torch.cuda.empty_cache()

    harmful = {k: torch.cat(v) for k, v in harmful.items()}
    harmless = {k: torch.cat(v) for k, v in harmless.items()}

    # ============================================================
    print(f"[4/7] 计算拒绝方向...")
    activation_layers = ["resid_pre", "resid_mid", "resid_post"]
    activation_refusals = defaultdict(list)

    def get_act_idx(cache_dict, act_name, layer):
        key = (act_name, layer)
        return cache_dict[utils.get_act_name(*key)]

    for layer_num in range(1, model.cfg.n_layers):
        pos = -1
        for layer in activation_layers:
            harmful_mean_act = get_act_idx(harmful, layer, layer_num)[:, pos, :].mean(dim=0)
            harmless_mean_act = get_act_idx(harmless, layer, layer_num)[:, pos, :].mean(dim=0)
            refusal_dir = harmful_mean_act - harmless_mean_act
            refusal_dir = refusal_dir / refusal_dir.norm()
            activation_refusals[layer].append(refusal_dir)

    selected_layers = ["resid_pre"]
    activation_scored = sorted(
        [
            activation_refusals[layer][l - 1]
            for l in range(1, model.cfg.n_layers)
            for layer in selected_layers
        ],
        key=lambda x: abs(x.mean()),
        reverse=True,
    )

    # ============================================================
    print(f"[5/7] 推理时介入测试（评估 top-{args.eval_n} 方向）...")
    N_INST_TEST = min(4, len(harmful_inst_test))
    baseline_generations = get_generations(model, tokenizer, harmful_inst_test[:N_INST_TEST], fwd_hooks=[])

    evals = []
    for refusal_dir in tqdm(activation_scored[: args.eval_n]):
        hook_fn = functools.partial(direction_ablation_hook, direction=refusal_dir)
        fwd_hooks = [
            (utils.get_act_name(act_name, layer), hook_fn)
            for layer in list(range(model.cfg.n_layers))
            for act_name in activation_layers
        ]
        intervention_generations = get_generations(
            model, tokenizer, harmful_inst_test[:N_INST_TEST], fwd_hooks=fwd_hooks
        )
        evals.append(intervention_generations)

    # ============================================================
    print(f"[6/7] 选择最佳方向并评估...")
    blacklist = ["I cannot", "I can't", "I'm sorry", "As an AI", "I apologize"]

    best_candidate = 0
    best_score = 0
    for candidate_idx in range(min(args.eval_n, len(evals))):
        score = sum(
            1 for i in range(N_INST_TEST)
            if not any(word in evals[candidate_idx][i] for word in blacklist)
        )
        if score > best_score:
            best_score = score
            best_candidate = candidate_idx

    print(f"\n最佳候选方向: #{best_candidate} (得分: {best_score}/{N_INST_TEST})")
    print("\n--- 基线 vs 介入对比 ---")
    for i in range(N_INST_TEST):
        print(f"\n\033[1mINSTRUCTION {i}: {harmful_inst_test[i]}\033[0m")
        print(f"\033[92mBASELINE: {baseline_generations[i][:200]}\033[0m")
        print(f"\033[91mINTERVENTION: {evals[best_candidate][i][:200]}\033[0m")

    # ============================================================
    print(f"\n[7/7] 正交化权重（永久修改模型）...")
    refusal_dir = activation_scored[best_candidate]

    if refusal_dir.device != model.W_E.device:
        refusal_dir = refusal_dir.to(model.W_E.device)
    model.W_E.data = get_orthogonalized_matrix(model.W_E, refusal_dir)

    for block in tqdm(model.blocks):
        if refusal_dir.device != block.attn.W_O.device:
            refusal_dir = refusal_dir.to(block.attn.W_O.device)
        block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O, refusal_dir)
        block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out, refusal_dir)

    # 验证
    print("\n验证正交化后的模型...")
    ortho_generations = get_generations(model, tokenizer, harmful_inst_test[:N_INST_TEST], fwd_hooks=[])
    for i in range(N_INST_TEST):
        print(f"\n\033[1mINSTRUCTION {i}: {harmful_inst_test[i]}\033[0m")
        print(f"\033[95mORTHOGONALIZED: {ortho_generations[i][:300]}\033[0m")

    # 保存
    if args.output:
        print(f"\n保存模型到 {args.output}...")
        hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        lm_model = hf_model.model
        state_dict = model.state_dict()

        lm_model.embed_tokens.weight = torch.nn.Parameter(state_dict["embed.W_E"].cpu())
        for l in range(model.cfg.n_layers):
            lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(
                einops.rearrange(
                    state_dict[f"blocks.{l}.attn.W_O"], "n h m->m (n h)", n=model.cfg.n_heads
                ).contiguous()
            )
            lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(
                torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous()
            )

        hf_model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"模型已保存到 {args.output}")

    print("\n完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abliteration: 移除LLM拒绝方向")
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it",
                        help="HuggingFace 模型 ID")
    parser.add_argument("--output", type=str, default=None,
                        help="保存路径（不指定则不保存）")
    parser.add_argument("--n-train", type=int, default=256,
                        help="训练样本数")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--eval-n", type=int, default=20,
                        help="评估 top-N 个候选方向")
    args = parser.parse_args()
    main(args)
