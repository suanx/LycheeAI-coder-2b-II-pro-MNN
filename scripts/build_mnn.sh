#!/usr/bin/env bash
# ① 编译 MNN，产出 MNNConvert（转换链路的必需工具）
set -euo pipefail

MNN_DIR="${MNN_DIR:-./MNN}"
BUILD_DIR="${MNN_DIR}/build"
JOBS="$(nproc 2>/dev/null || echo 4)"

if [ ! -d "$MNN_DIR/.git" ]; then
  echo ">> 克隆 MNN 源码 ..."
  git clone --depth 1 https://github.com/alibaba/MNN.git "$MNN_DIR"
else
  echo ">> MNN 源码已存在，跳过克隆"
fi

echo ">> 生成 schema ..."
( cd "$MNN_DIR" && ./schema/generate.sh )

echo ">> cmake 配置 ..."
cmake -S "$MNN_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON

echo ">> 编译中（${JOBS} 线程，约 15-30 分钟）..."
cmake --build "$BUILD_DIR" --parallel "$JOBS"

if [ -x "$BUILD_DIR/MNNConvert" ]; then
  echo "✅ 完成：$BUILD_DIR/MNNConvert"
  echo "   llm_demo: $BUILD_DIR/llm_demo（可用于本地验证）"
else
  echo "❌ 未找到 MNNConvert。请确认 -DMNN_BUILD_CONVERTER=ON 已生效"
  find "$BUILD_DIR" -name 'MNNConvert*' 2>/dev/null || true
  exit 1
fi
