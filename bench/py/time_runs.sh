#!/usr/bin/env bash
# time_runs.sh <graph>...: best-of-3 wall times (same seed 100) under the shared lock; members bench/out/baselines/<graph>/time-*/times.json.
# Each run waits, holding the lock, until the machine is idle (see idle()).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PY=$PHD/sc-brrr/runs/.conda/613289a77143e229a42a609590d65392_/bin/python
LOCK=$PHD/detleiden/bench/.cpu.lock
# idle = no sc-brrr scoring and <= 2 runnable tasks (this check included), twice 1 s apart; waited for while holding the lock
idle() { ! pgrep -f "python[^ ]* [^ ]*(score|runner)\.py" >/dev/null && [ "$(cut -d' ' -f4 /proc/loadavg | cut -d/ -f1)" -le 2 ]; }
export -f idle
m() { flock $LOCK bash -c 'until idle && sleep 1 && idle; do sleep 5; done; echo "start $(date +%T) $(cut -d" " -f1-4 /proc/loadavg)"; "$@"; echo "end $(date +%T) $(cut -d" " -f1-4 /proc/loadavg)"' _ "$@"; }
for g in "$@"; do
  for t in 1 2 8; do
    (cd $PHD/brrr-gve && m env OMP_NUM_THREADS=$t pixi run python -I "$here/run_member.py" --impl gve --graph $g --name time-gve-seed-t$t --order seed --replicates 3 --seed_stride 0)
  done
  for i in igraph igraph2 igraph-fix leidenalg; do
    r=3; [[ $g == syn_1M && $i != igraph2 && $i != igraph ]] && r=1  # ponytail: 1M converged runs take minutes; best of 1 there
    m env OMP_NUM_THREADS=1 $PY -I "$here/run_member.py" --impl $i --graph $g --name time-$i --replicates $r --seed_stride 0
  done
done
