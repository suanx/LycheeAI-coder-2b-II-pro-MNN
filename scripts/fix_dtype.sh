#!/usr/bin/env bash
# 修复导出时的 dtype 不匹配
#
# 症状（llmexport.py 报错）：
#   RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16
#   栈：llmexport.py response -> model.py forward -> transformers.py:296 self.q_proj(hidden_states)
#
# 根因：
#   MNN 的 Embedding 类（utils/transformers.py:14）会把词表重建为 torch.nn.Embedding
#   以做词表对齐，新建层默认 float32，于是 embedding 输出 float32；
#   随后 RMSNorm 中 `bf16权重 * float32输入` 类型提升为 float32，
#   喂给 bf16 的 q_proj 时两者 dtype 不一致而报错。
#   注意：本模型是 bf16 权重，故必须整体以 float32 加载才能全程自洽；
#         改成 fp16 同样不匹配（float32 输入 vs fp16 权重）。
#
# 修法：把 config.json 的 torch_dtype 改为 float32。
#   MNN 加载时用的是 torch_dtype='auto'，会直接读这个字段。
#   量化在导出阶段完成，float32 加载不影响最终 4bit 产物体积，精度反而略好。
#
# 代价：内存占用从 ~5GB 升到 ~10GB（2.5B × 4B），导出稍慢。32GB 内存足够。
#
# 用法：
#   ./scripts/fix_dtype.sh [模型目录，默认 ./original]
set -euo pipefail

MODEL_DIR="${1:-./original}"
CFG="${MODEL_DIR}/config.json"

if [ ! -f "$CFG" ]; then
  echo "❌ 找不到 $CFG"
  exit 1
fi

cp "$CFG" "${CFG}.bak"
echo ">> 已备份 -> ${CFG}.bak"

python - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
old = d.get('torch_dtype') or d.get('dtype')
print(f'>> torch_dtype: {old} -> float32')
d['torch_dtype'] = 'float32'
d.pop('dtype', None)
json.dump(d, open(p, 'w'), ensure_ascii=False, indent=2)
print('>> 已写入')
PY

echo ">> 当前 config.json："
cat "$CFG"

echo ""
echo "✅ 完成。现在重新导出（内存峰值约 10-15GB）："
echo "   nohup ./scripts/convert_to_mnn.sh > convert.log 2>&1 &"
