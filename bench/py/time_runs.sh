#!/usr/bin/env bash
# time_runs.sh <graph>...: best-of-3 wall times (same seed 100) under the shared lock; members bench/out/baselines/<graph>/time-*/times.json.
# Refuses to run unless the machine is idle (no sc-brrr score.py/runner.py; 1-min load < 2).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PY=$PHD/sc-brrr/runs/.conda/613289a77143e229a42a609590d65392_/bin/python
LOCK=$PHD/detleiden/bench/.cpu.lock
idle() { ! pgrep -f "score.py|runner.py" >/dev/null && awk '{exit !($1 < 2)}' /proc/loadavg || { echo "not idle: $(uptime)" >&2; exit 1; }; }
m() { idle; flock $LOCK "$@"; }
for g in "$@"; do
  for t in 1 2 8; do
    (cd $PHD/brrr-gve && OMP_NUM_THREADS=$t m pixi run python -I "$here/run_member.py" --impl gve --graph $g --name time-gve-seed-t$t --order seed --replicates 3 --seed_stride 0)
  done
  for i in igraph igraph2 igraph-fix leidenalg; do
    OMP_NUM_THREADS=1 m $PY -I "$here/run_member.py" --impl $i --graph $g --name time-$i --replicates 3 --seed_stride 0
  done
done
