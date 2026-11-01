#!/usr/bin/env bash
# 修复 MLX 转存权重带来的 tokenizer 兼容问题
#
# 症状（llmexport.py 报这个错）：
#   RuntimeError: Failed to load tokenizer from ./original
#   ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
#
# 根因：原模型用 MLX 训练，转存 f16 时把 MLX 专有的
#   "tokenizer_class": "TokenizersBackend" 带进了 tokenizer_config.json，
#   而 HuggingFace transformers 不认识这个类名，导致 AutoTokenizer 加载失败。
#   MNN 的 LlmTokenizer 会吞掉真实异常，只抛 RuntimeError，所以要先单独复现才看得到。
#
# 修法：删掉该字段，让 AutoTokenizer 依据 tokenizer.json 自动推断（实际是 PreTrainedTokenizerFast）。
#
# 用法：
#   ./scripts/fix_tokenizer.sh [模型目录，默认 ./original]
set -euo pipefail

MODEL_DIR="${1:-./original}"
CFG="${MODEL_DIR}/tokenizer_config.json"

if [ ! -f "$CFG" ]; then
  echo "❌ 找不到 $CFG"
  exit 1
fi

echo ">> 修正前："
python -c "import json;d=json.load(open('$CFG'));print('   tokenizer_class =', d.get('tokenizer_class'))"

cp "$CFG" "${CFG}.bak"
echo ">> 已备份 -> ${CFG}.bak"

python - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d.pop('tokenizer_class', None)   # MLX 专有，transformers 不认
d.pop('auto_map', None)          # 可能指向不存在的自定义类
json.dump(d, open(p, 'w'), ensure_ascii=False, indent=2)
print('>> 已移除 tokenizer_class / auto_map')
PY

echo ">> 验证加载："
python - "$MODEL_DIR" <<'PY'
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
print('   ✅ 加载成功，vocab =', len(t))
print('   eos =', t.eos_token, t.eos_token_id)
ids = t.encode('帮我写个快速排序')[:10]
print('   试编码:', ids)
# LycheeAI / MiniCPM5-2B 的词表应为 130560，对不上会导致转换后输出乱码
if len(t) != 130560:
    print(f'   ⚠️  词表 {len(t)} 与预期的 130560 不符，请核对基座')
else:
    print('   ✅ 词表大小符合预期 (130560)')
PY

echo "✅ 完成，可继续运行 ./scripts/convert_to_mnn.sh"
