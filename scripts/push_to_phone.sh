#!/usr/bin/env bash
# ⑥ 把转换好的 MNN 模型推到手机，供 MNN Chat 加载
#
# 用法：
#   MODEL_DIR=./lychee-mnn ./scripts/push_to_phone.sh
#
# 推送后在 MNN Chat 里点「添加本地模型」，选中该目录即可。
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./lychee-mnn}"
NAME="${NAME:-$(basename "$MODEL_DIR")}"
PHONE_DIR="${PHONE_DIR:-/sdcard/MNN/chat}"

for f in llm.mnn llm.mnn.weight tokenizer.txt; do
  [ -f "${MODEL_DIR}/${f}" ] || { echo "❌ 缺少 ${f}，请先运行 convert_to_mnn.sh"; exit 1; }
done

if ! command -v adb >/dev/null 2>&1; then
  echo "❌ 未找到 adb。可手动把 ${MODEL_DIR} 拷贝到手机的 ${PHONE_DIR}/ 下"
  exit 1
fi

adb devices | sed -n '2p' | grep -q 'device' || {
  echo "❌ 未检测到已连接的设备（需开启 USB 调试）"
  adb devices
  exit 1
}

echo ">> 推送 ${MODEL_DIR} -> ${PHONE_DIR}/${NAME} ..."
adb shell "mkdir -p ${PHONE_DIR}/${NAME}"
adb push "${MODEL_DIR}/." "${PHONE_DIR}/${NAME}/"

echo "✅ 推送完成"
echo "   MNN Chat → 「添加本地模型」→ 选择 ${PHONE_DIR}/${NAME}"
echo ""
echo "   建议设置："
echo "   · 计算后端：CPU（兼容最好）/ OpenCL（旗舰机更快）"
echo "   · 系统提示词：你是 LycheeAI-coder-2b-II-pro，由 MiniCPM5-2B 通过 LoRA 微调而来的编程助手。"
echo "     回答简洁、直接，不谄媚；做不到的事直接说明，不要编造。"
echo "   · 温度 ≤ 0.4，输出更稳定"
