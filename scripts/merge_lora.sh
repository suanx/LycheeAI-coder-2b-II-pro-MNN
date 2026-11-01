#!/usr/bin/env bash
# ④ 把 LoRA 权重合并回完整模型
#
# 建议先合并再转 MNN —— llmexport 虽然也支持 --lora_path，
# 但先合并并验证过，能少排查一层问题。
#
# 用法：
#   CKPT=./output/vx-xxx/checkpoint-xxx ./scripts/merge_lora.sh
set -euo pipefail

CKPT="${CKPT:?请设置 CKPT=你的 checkpoint 目录，例如 ./output/vx-000/checkpoint-100}"
OUT="${OUT:-./merged_model}"

echo ">> 合并 LoRA ..."
swift export \
  --adapters "$CKPT" \
  --merge_lora true \
  --output_dir "$OUT" 2>/dev/null || \
swift export --adapters "$CKPT" --merge_lora true

echo ">> 产物："
ls -lh "$OUT" 2>/dev/null || ls -lh "${CKPT}-merged" 2>/dev/null || true
echo "✅ 合并完成，把该目录传给 ./scripts/convert_to_mnn.sh 的 MODEL_DIR"
