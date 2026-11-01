#!/usr/bin/env bash
# ② 从魔搭下载 bf16 无损权重
#
# ⚠️ 必须用 -f16 无损版。默认的 LycheeAI-coder-2b-II-pro 是 MLX 4bit 量化版，
#    有损，不能用于微调，也不能作为二次转换的基础。
set -euo pipefail

MODEL_ID="${MODEL_ID:-whcl412/LycheeAI-coder-2b-II-pro-f16}"
OUT_DIR="${OUT_DIR:-./original}"

echo ">> 安装 modelscope ..."
pip install -q modelscope -U

echo ">> 下载 ${MODEL_ID} -> ${OUT_DIR} ..."
python - <<PY
from modelscope import snapshot_download
p = snapshot_download('${MODEL_ID}', local_dir='${OUT_DIR}')
print('downloaded to', p)
PY

echo ">> 校验权重体积 ..."
SAFE="$(find "$OUT_DIR" -name '*.safetensors' | head -1)"
if [ -z "$SAFE" ]; then
  echo "❌ 未找到 .safetensors，请检查 git-lfs / 网络"
  exit 1
fi
SIZE="$(stat -c%s "$SAFE" 2>/dev/null || stat -f%z "$SAFE")"
echo "   权重文件: $SAFE"
echo "   大小: $(( SIZE / 1024 / 1024 )) MB"
if [ "$SIZE" -lt 100000000 ]; then
  echo "❌ 体积明显偏小，疑似空壳文件（git-lfs 未安装？）"
  exit 1
fi
echo "✅ 权重就绪，约 $(( SIZE / 1024 / 1024 )) MB"
