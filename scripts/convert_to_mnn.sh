#!/usr/bin/env bash
# ⑤ 核心：把 HuggingFace 格式模型转成 MNN Chat 可用的 .mnn 格式
#
# 用法：
#   MODEL_DIR=./original ./scripts/convert_to_mnn.sh
#
# 建议先用未经微调的原版跑通一次，确认能转出来，再回头做微调。
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./original}"
MNN_DIR="${MNN_DIR:-./MNN}"
MNNCONVERT="${MNNCONVERT:-${MNN_DIR}/build/MNNConvert}"
DST="${DST:-./lychee-mnn}"

# 量化档位：4bit ≈1.5GB（手机推荐）；8bit ≈2.7GB（质量更好、更吃内存）
QUANT_BIT="${QUANT_BIT:-4}"
QUANT_BLOCK="${QUANT_BLOCK:-64}"
# lm_head 保持 8bit：该模型词表 130560 偏大，这里掉精度很明显
LM_QUANT_BIT="${LM_QUANT_BIT:-8}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "❌ 模型目录不存在：$MODEL_DIR"
  exit 1
fi
if [ ! -x "$MNNCONVERT" ]; then
  echo "❌ 找不到 MNNConvert：$MNNCONVERT"
  echo "   请先运行 ./scripts/build_mnn.sh"
  exit 1
fi

echo ">> 安装导出依赖 ..."
pip install -q -r "${MNN_DIR}/transformers/llm/export/requirements.txt"

echo ">> 导出 MNN（quant_bit=${QUANT_BIT}, hqq, lm_quant_bit=${LM_QUANT_BIT}）..."
python "${MNN_DIR}/transformers/llm/export/llmexport.py" \
  --path "$MODEL_DIR" \
  --export mnn \
  --quant_bit "$QUANT_BIT" \
  --quant_block "$QUANT_BLOCK" \
  --hqq \
  --lm_quant_bit "$LM_QUANT_BIT" \
  --mnnconvert "$MNNCONVERT" \
  --dst_path "$DST" \
  --test "帮我写个快速排序"

echo ">> 检查产物 ..."
MISSING=0
for f in llm.mnn llm.mnn.weight tokenizer.txt llm_config.json config.json; do
  if [ -f "${DST}/${f}" ]; then
    printf '   ✓ %-20s %s\n' "$f" "$(du -h "${DST}/${f}" | cut -f1)"
  else
    printf '   ✗ %-20s 缺失\n' "$f"
    MISSING=1
  fi
done

# chat template 没带过去会导致对话格式错乱
if ! grep -q 'chat_template\|use_template' "${DST}/config.json" 2>/dev/null; then
  echo "⚠️  config.json 里未见 chat_template 字段，对话可能格式错乱，请手动补上"
fi

echo ">> 产物总大小：$(du -sh "$DST" | cut -f1)"
[ "$MISSING" -eq 0 ] && echo "✅ 转换完成：$DST" || {
  echo "❌ 产物不完整，尝试两段式回退（先 onnx 再 MNNConvert）"
  echo "   python ${MNN_DIR}/transformers/llm/export/llmexport.py --path $MODEL_DIR --export onnx --dst_path ./onnx_out"
  echo "   ${MNNCONVERT} --modelFile ./onnx_out/onnx/llm.onnx --MNNModel llm.mnn \\"
  echo "     --keepInputFormat --weightQuantBits=${QUANT_BIT} --weightQuantBlock=128 \\"
  echo "     -f ONNX --transformerFuse=1 --allowCustomOp --saveExternalData"
  exit 1
}
