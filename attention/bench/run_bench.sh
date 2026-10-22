#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# Controlled run behind benches/<gpu>.md (methodology: docs/performance.md).
#
#   bench/run_bench.sh
#   GPU=1 COOLING="open bench, 100 % fan" NOTES="..." bench/run_bench.sh
#
# Raises the power limit to its maximum, allows the SM clock up to its top and locks the memory clock at its
# maximum. All three are restored on exit, including Ctrl+C. Needs sudo for nvidia-smi. Raw output and the
# 250 ms board telemetry stay in out/bench_<gpu>/; report.py writes benches/<gpu>.md and benches/table.md.
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
PACK_COOL=${PACK_COOL:-10}
export MLOPS_BENCH_COOL=${MLOPS_BENCH_COOL:-1}
export MLOPS_BENCH_WARMUP=${MLOPS_BENCH_WARMUP:-4}

q() { nvidia-smi -i "$GPU" --query-gpu="$1" --format=csv,noheader,nounits | tr -d ' '; }
NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)
FILE=${GPU_NAME:-$(python3 -c "import sys; sys.path.insert(0, 'bench'); from report import slug; print(slug('''$NAME'''))")}
OUT=out/bench_$FILE
rm -rf "$OUT" && mkdir -p "$OUT"

echo "== $NAME (GPU $GPU) -> benches/$FILE.md; processes on the GPU (should be none):"
nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
make -s bench-bin GPU="$GPU" || exit 1

PL_OLD=$(q power.limit); PL_MAX=$(q power.max_limit); SM_MAX=$(q clocks.max.sm); MEM_MAX=$(q clocks.max.mem)
restore() {
  sudo nvidia-smi -i "$GPU" -rgc > /dev/null
  sudo nvidia-smi -i "$GPU" -rmc > /dev/null 2>&1
  sudo nvidia-smi -i "$GPU" -pl "$PL_OLD" > /dev/null
  echo "restored: power limit $PL_OLD W, SM and memory clocks unlocked"
}
trap restore EXIT
sudo -v || exit 1
sudo nvidia-smi -i "$GPU" -pm 1 > /dev/null
sudo nvidia-smi -i "$GPU" -pl "$PL_MAX" > /dev/null && PL="$PL_MAX W (maximum; default $PL_OLD W)" || PL="$PL_OLD W"
LOCKS=""
sudo nvidia-smi -i "$GPU" -lgc "$SM_MAX,$SM_MAX" > /dev/null && LOCKS="SM clock allowed up to $SM_MAX MHz" || LOCKS="SM clock: GPU Boost"
if sudo nvidia-smi -i "$GPU" -lmc "$MEM_MAX,$MEM_MAX" > /dev/null 2>&1; then
  LOCKS="$LOCKS, memory clock locked at $MEM_MAX MHz"
else
  LOCKS="$LOCKS, memory clock not lockable"
  echo "!! memory clock lock not supported: check the memory clock in the logs"
fi

{
  echo "name=$NAME"
  echo "file=$FILE"
  echo "compute_cap=$(q compute_cap)"
  echo "memory=$(q memory.total) MiB"
  echo "driver=$(q driver_version)"
  echo "cuda=$(${CUDA_HOME:-/usr/local/cuda}/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  echo "library=$(sed -n 's/^#define MLOPS_ATTENTION_VERSION "\(.*\)"/\1/p' include/mlops_attention.h)"
  echo "date=$(date +%Y-%m-%d)"
  echo "power_limit=$PL"
  echo "clock_lock=$LOCKS"
  echo "warmup=$MLOPS_BENCH_WARMUP x 2048^3 FP32"
  echo "row_cool=$MLOPS_BENCH_COOL s"
  echo "repeats=${MLOPS_BENCH_REPS:-per-scenario defaults (~1 s timed work)}"
  echo "cooling=${COOLING:-}"
  echo "notes=${NOTES:-}"
} > "$OUT/env.txt"

echo "== bench_fbla (after ${PACK_COOL} s idle)"
sleep "$PACK_COOL"
nvidia-smi -i "$GPU" --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active \
  --format=csv,noheader -lms 250 > "$OUT/clocks_bench_fbla.csv" &
LOG=$!
CUDA_VISIBLE_DEVICES=$GPU ./out/bench_fbla | tee "$OUT/bench_fbla.txt"
kill $LOG; wait $LOG 2> /dev/null
python3 bench/report.py "$OUT" benches
