#!/usr/bin/env bash
# 一键消音（去拒答/abliteration）LycheeAI-coder-2b-II-pro
# 用法: ./abliterate_mincpm.sh [bf16模型目录] [输出目录] [采样数]
# 详见 UNCENSOR_TRAINING.md
set -euo pipefail

MODEL_DIR="${1:-./original-f16}"
OUTPUT_DIR="${2:-./abliterated}"
N_SAMPLES="${3:-128}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "错误: 模型目录 $MODEL_DIR 不存在"
    echo "请先下载 bf16 权重: modelscope download --model whcl412/LycheeAI-coder-2b-II-pro-f16 --local_dir $MODEL_DIR"
    exit 1
fi

echo "==> 模型: $MODEL_DIR"
echo "==> 输出: $OUTPUT_DIR"
echo "==> 采样数: $N_SAMPLES (有害/无害 prompt 各取 N 条计算拒绝方向)"

# 快速版: 直接对 top-N 候选方向正交化 + 验证, 2B 模型单卡 A10 几分钟
python uncensor/abliterate_fast.py \
    --model "$MODEL_DIR" \
    --output "$OUTPUT_DIR" \
    --n-samples "$N_SAMPLES" \
    --try-n 10

echo "==> 完成。产物: $OUTPUT_DIR (消音后 bf16 模型 + refusal_direction.pt + comparison.json)"
echo "==> 下一步: 用 uncensor/refusal_test_zh.json 测拒答率, 顽固类别按 UNCENSOR_TRAINING.md 第 4 步 QLoRA 收尾"
