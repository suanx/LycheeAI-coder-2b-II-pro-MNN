#!/usr/bin/env bash
# 心跳：每 60 秒向 heartbeat.log 追加一行状态
#
# 用途：魔搭 Notebook 空闲 1 小时会自动关机、单实例最长 10 小时。
#       长任务挂后台时跑着它，回来 tail 一眼就知道进度；
#       若实例被关停，日志时间戳会出现断层，一眼可辨。
#
# 用法：
#   nohup ./scripts/heartbeat.sh >/dev/null 2>&1 &
#   tail -f heartbeat.log
set -uo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${LOG:-$BASE/heartbeat.log}"
INTERVAL="${INTERVAL:-60}"

have() { command -v "$1" >/dev/null 2>&1; }

# 阶段判定：按产物是否存在推断整体进度
stage() {
  if [ -f "$BASE/lychee-mnn/llm.mnn" ] || [ -f "$BASE/lychee-mnn.tar.gz" ]; then echo "转换完成";
  elif [ -x "$BASE/MNN/build/MNNConvert" ] && [ -d "$BASE/original" ]; then echo "转换中";
  elif [ -x "$BASE/MNN/build/MNNConvert" ]; then echo "编译完成/待下载";
  elif [ -d "$BASE/MNN" ]; then echo "编译中";
  else echo "未开始"; fi
}

tick=0
while true; do
  tick=$((tick+1))
  ts="$(date '+%F %T')"

  mem="$(free -h 2>/dev/null | awk '/^Mem:/{print $3"/"$2}')"
  [ -z "$mem" ] && mem="n/a"

  disk="$(df -h "$BASE" 2>/dev/null | awk 'NR==2{print $4}')"
  [ -z "$disk" ] && disk="n/a"

  mnnc="$([ -x "$BASE/MNN/build/MNNConvert" ] && echo READY || echo building)"
  w="$(du -sh "$BASE/original" 2>/dev/null | cut -f1)"
  [ -z "$w" ] && w="-"

  # 编译进程是否还在
  if pgrep -f 'cmake --build|make -j|cc1plus' >/dev/null 2>&1; then cc="编译进程:在"; else cc="编译进程:无"; fi
  if pgrep -f 'llmexport.py' >/dev/null 2>&1; then cv="转换进程:在"; else cv="转换进程:无"; fi

  printf '[%s] tick=%d | 阶段=%s | 内存=%s | 可用磁盘=%s | MNNConvert=%s | 权重=%s | %s | %s\n' \
    "$ts" "$tick" "$(stage)" "$mem" "$disk" "$mnnc" "$w" "$cc" "$cv" >> "$LOG"

  sleep "$INTERVAL"
done
