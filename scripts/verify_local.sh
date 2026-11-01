#!/usr/bin/env bash
# 在电脑上先用 llm_demo 验证转换结果，确认能正常说话再推到手机。
#
# 用法：
#   MODEL_DIR=./lychee-mnn ./scripts/verify_local.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./lychee-mnn}"
MNN_DIR="${MNN_DIR:-./MNN}"
DEMO="${DEMO:-${MNN_DIR}/build/llm_demo}"

if [ ! -x "$DEMO" ]; then
  echo "❌ 找不到 llm_demo：$DEMO（需 -DMNN_BUILD_LLM=ON 编译）"
  exit 1
fi

PROMPT_FILE="$(mktemp /tmp/lychee_prompt_XXXX.txt)"
printf '帮我写一个 Python 二分查找，带注释。\n' > "$PROMPT_FILE"

echo ">> 用 llm_demo 跑一次推理（config.json 需指向同目录的权重）..."
"$DEMO" "${MODEL_DIR}/config.json" "$PROMPT_FILE"

echo ""
echo ">> 性能基准（可选）："
echo "   ${MNN_DIR}/build/llm_bench -m ${MODEL_DIR}/config.json -a cpu -t 4,8 -p 32,64 -n 32 -rep 3"

rm -f "$PROMPT_FILE"
