#!/bin/bash
# Run skill-set ablation items against a planner that is already serving. Resumable: finished items are skipped,
# failed items (run_item.sh moves them to <out>/failed/) run again on the next pass.
# Needs HF_HOME pointing at the cache holding the SAM3 checkpoint, usually with HF_HUB_OFFLINE=1.
# Usage: PORT=8100 WORKERS=3 setsid nohup bash scripts/skillset_run.sh <items.txt> <out_root> > run.out 2>&1 &
#   STOP: touch <out_root>/STOP (workers finish their item, nothing new starts).
#   MIN_FREE_MIB: free GPU memory an item needs before it starts (default 6500).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ITEMS=$1 OUT=$2 WORKERS=${WORKERS:-1}
export OUT ROOT MIN_FREE_MIB=${MIN_FREE_MIB:-6500}
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

run_item() {
  [ -e "$OUT/STOP" ] && exit 0
  key=$(echo "$*" | tr ' :' '__')
  [ -d "$OUT/$key" ] && exit 0
  # One item at a time starts, with room for a simulation, and the lock is held until it has loaded:
  # a CUDA OOM discards the item and could hurt the planner.
  exec 9>"$OUT/.gate"; flock 9
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$MIN_FREE_MIB" ]; do
    [ -e "$OUT/STOP" ] && exit 0
    sleep 30
  done
  t0=$(date +%s)
  bash "$ROOT/scripts/run_item.sh" "$OUT" "$@" 9>&- &
  pid=$!
  sleep 150
  flock -u 9
  wait $pid
  rc=$?
  # run_item.sh scores a trial that failed on a checkpoint download as a failed trial: not a result here.
  if grep -qE "Client Error|LocalEntryNotFound" "$OUT/$key/log.txt" 2>/dev/null; then
    mkdir -p "$OUT/failed"; mv "$OUT/$key" "$OUT/failed/$key.$(date +%s)"; echo "[item] FAILED setup $key" >&2
  fi
  # A CUDA OOM stops the run: the items are not valid and the planner shares this GPU.
  if grep -qs "CUDA OOM\|out of memory" "$OUT"/failed/*/log.txt; then
    touch "$OUT/STOP"; echo "$(date -u +%H:%M:%S) STOP: CUDA OOM in failed/" >> "$OUT/progress.log"
  fi
  echo "$(date -u +%H:%M:%S) $* exit=$rc $(( $(date +%s) - t0 ))s" >> "$OUT/progress.log"
}
export -f run_item

while :; do
  [ -e "$OUT/STOP" ] && break
  xargs -a "$ITEMS" -P "$WORKERS" -I{} bash -c 'run_item {}'
  left=$(while read -r line; do [ -d "$OUT/$(echo "$line" | tr ' :' '__')" ] || echo x; done < "$ITEMS" | wc -l)
  [ "$left" = 0 ] && { echo "ALLDONE $(date -u +%H:%M:%S)" >> "$OUT/progress.log"; break; }
  sleep 30
done
