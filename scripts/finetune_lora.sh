#!/usr/bin/env bash
# ③ 可选：用 ms-swift 做 LoRA 微调（在转 MNN 之前进行）
#
# 用法：
#   DATASET=my_data.jsonl ./scripts/finetune_lora.sh
#
# 数据集格式（JSONL，每行一条，system 可选）：
# {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
set -euo pipefail

MODEL="${MODEL:-whcl412/LycheeAI-coder-2b-II-pro-f16}"
DATASET="${DATASET:-my_data.jsonl}"
OUT_DIR="${OUT_DIR:-./output}"

if [ ! -f "$DATASET" ]; then
  echo "❌ 找不到数据集 $DATASET"
  exit 1
fi

echo ">> 安装 ms-swift ..."
pip install -q ms-swift -U

echo ">> 开始 LoRA 微调 ..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" swift sft \
  --model "$MODEL" \
  --train_type lora \
  --dataset "$DATASET" \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS:-1}" \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 8 \
  --max_length 2048 \
  --output_dir "$OUT_DIR"

echo "✅ 训练完成。checkpoint 在 ${OUT_DIR}/vx-xxx/checkpoint-xxx"
echo ">> 下一步：./scripts/merge_lora.sh"
