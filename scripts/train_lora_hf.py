#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 HuggingFace 原生生态（transformers + peft + Trainer）做 LoRA 微调
—— 不依赖 ms-swift，不依赖 HF Hub 数据集，只吃本地 jsonl。

适用：给 LycheeAI-coder-2b-II-pro 补代码能力 / 风格 / 工具调用格式。

⚠️ 顺序铁律：训练必须在【消音之前】。
   先消音再训练，新梯度会把拒答方向重新学回来，消音白做。
   正确链路：original → [本脚本训练 → merge] → 消音 → 转 MNN。

数据格式（jsonl，每行一条，system 可选）：
{"messages":[{"role":"system","content":"..."},
             {"role":"user","content":"..."},
             {"role":"assistant","content":"..."}]}

用法：
  python scripts/train_lora_hf.py --data my_data.jsonl --model ./original --out ./lora_out
  python scripts/train_lora_hf.py --data my_data.jsonl --merge --out_dir ./merged
"""
import argparse, json, os, sys
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model


# ── 消掉 MLX 留下的 TokenizersBackend 字段 ──────────────────
def fix_tokenizer_config(model_dir):
    """MLX 转存的权重会带 "tokenizer_class": "TokenizersBackend"，
    HF transformers 不认，AutoTokenizer 会直接崩。删掉即可。"""
    p = os.path.join(model_dir, 'tokenizer_config.json')
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding='utf-8'))
    if 'tokenizer_class' in d or 'auto_map' in d:
        d.pop('tokenizer_class', None)
        d.pop('auto_map', None)
        json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print('[fix] 已移除 tokenizer_config.json 中的 TokenizersBackend 字段')


class SFTData(Dataset):
    """只让 assistant 回复部分参与 loss（prompt 全部 mask 成 -100）"""
    def __init__(self, path, tok, max_len=2048):
        self.ex = []
        bad = 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msgs = d.get('messages') or d.get('conversations')
                if not msgs:
                    bad += 1
                    continue
                # 完全手工拼接 token id，不依赖 apply_chat_template 的返回类型
                im_start = tok.convert_tokens_to_ids('<|im_start|>')
                im_end = tok.convert_tokens_to_ids('<|im_end|>')
                nl = tok.convert_tokens_to_ids('\n')
                def enc(t):
                    return tok.encode(t, add_special_tokens=False) if t else []
                ids, lab = [], []
                ids.append(tok.bos_token_id); lab.append(-100)
                for m in msgs:
                    role = m.get('role', 'user')
                    seg = [im_start] + enc(role) + [nl] + enc(m.get('content', '')) + [im_end] + [nl]
                    is_ass = (role == 'assistant')
                    for j, tid in enumerate(seg):
                        ids.append(tid)
                        lab.append(tid if (is_ass and j >= 3) else -100)
                if len(ids) > max_len:
                    ids, lab = ids[:max_len], lab[:max_len]
                ids_t = torch.tensor(ids, dtype=torch.long)
                labels = torch.tensor(lab, dtype=torch.long)
                if (labels != -100).sum().item() == 0:
                    labels = ids_t.clone()
                self.ex.append({'input_ids': ids_t, 'labels': labels})
        print(f'[data] 有效样本 {len(self.ex)} 条，跳过 {bad} 条')

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id):
    m = max(len(b['input_ids']) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        n = m - len(b['input_ids'])
        ids.append(torch.cat([b['input_ids'], torch.full((n,), pad_id, dtype=torch.long)]))
        lab.append(torch.cat([b['labels'], torch.full((n,), -100, dtype=torch.long)]))
        att.append(torch.cat([torch.ones_like(b['input_ids']),
                              torch.zeros(n, dtype=torch.long)]))
    return {'input_ids': torch.stack(ids),
            'labels': torch.stack(lab),
            'attention_mask': torch.stack(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='本地 jsonl 数据集')
    ap.add_argument('--model', default='./original', help='基座权重目录')
    ap.add_argument('--out', default='./lora_out', help='LoRA adapter 输出目录')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--max_len', type=int, default=2048)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--merge', action='store_true', help='训完直接合并回完整模型')
    ap.add_argument('--out_dir', default='./merged', help='--merge 时的输出目录')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    fix_tokenizer_config(args.model)

    print(f'[1/4] 加载 tokenizer & 模型: {args.model}')
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f'      vocab = {len(tok)}  (本模型应为 130560)')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = torch.bfloat16 if dev == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, trust_remote_code=True, low_cpu_mem_usage=True)
    model.config.use_cache = False          # 训练时关掉 KV cache，省显存

    print('[2/4] 挂 LoRA')
    lc = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()

    print(f'[3/4] 读数据: {args.data}')
    ds = SFTData(args.data, tok, args.max_len)
    if len(ds) == 0:
        print('❌ 数据集为空，检查 jsonl 格式'); sys.exit(1)

    ta = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type='cosine',
        warmup_ratio=0.05, logging_steps=5, save_strategy='no',
        bf16=(dev == 'cuda'), report_to=[], seed=args.seed,
        gradient_checkpointing=True, optim='adamw_torch')

    tr = Trainer(model=model, args=ta, train_dataset=ds,
                 data_collator=lambda b: collate(b, tok.pad_token_id))
    tr.train()

    print(f'[4/4] 保存 adapter -> {args.out}')
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    if args.merge:
        print(f'合并回完整模型 -> {args.out_dir}')
        merged = model.merge_and_unload()
        merged = merged.to(torch.float32)     # 转 MNN 前统一 fp32，避免导出踩 dtype 坑
        merged.save_pretrained(args.out_dir, safe_serialization=True)
        tok.save_pretrained(args.out_dir)
        for f in ('chat_template.jinja', 'generation_config.json'):
            src = os.path.join(args.model, f)
            if os.path.exists(src):
                import shutil; shutil.copy(src, os.path.join(args.out_dir, f))
        print(f'✅ 已合并: {args.out_dir}')


if __name__ == '__main__':
    main()
