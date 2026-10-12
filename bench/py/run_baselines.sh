#!/usr/bin/env bash
: "${PHD:?set PHD to the directory holding the brrr-gve, sc-brrr and detleiden checkouts}"
# run_baselines.sh <graph>...: untimed baseline members (labels + determinism) into bench/out/baselines/<graph>/, then bench_score.py.
# IMPLS="igraph2 igraph-fix leidenalg" (default) picks the 20-seed single-threaded arms. Runs hold the lock (untimed, but they would
# perturb the other track's timings).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PY=$PHD/sc-brrr/runs/.conda/613289a77143e229a42a609590d65392_/bin/python
L="flock $HOME/phd/detleiden/bench/.cpu.lock"
gve() { (cd $PHD/brrr-gve && OMP_NUM_THREADS=$1 $L pixi run python -I "$here/run_member.py" --impl gve "${@:2}"); }
for g in "$@"; do
  [[ " ${IMPLS-igraph2 igraph-fix leidenalg} " == *" igraph2 "* ]] && $L $PY -I "$here/run_member.py" --impl igraph2 --graph $g --name igraph-niter2-seeds
  [[ " ${IMPLS-igraph2 igraph-fix leidenalg} " == *" igraph-fix "* ]] && $L $PY -I "$here/run_member.py" --impl igraph-fix --graph $g --name igraph-fix-seeds
  [[ " ${IMPLS-igraph2 igraph-fix leidenalg} " == *" leidenalg "* ]] && $L $PY -I "$here/run_member.py" --impl leidenalg --graph $g --name leidenalg-seeds
  for o in input seed; do
    gve 1 --graph $g --name gve-$o-t1-same --order $o --replicates 10 --seed_stride 0
    gve 2 --graph $g --name gve-$o-t2-same --order $o --replicates 20 --seed_stride 0
    gve 8 --graph $g --name gve-$o-t8-same --order $o --replicates 20 --seed_stride 0
  done
  gve 1 --graph $g --name gve-seed-t1-seeds --order seed --replicates 20 --seed_stride 1
  gve 8 --graph $g --name gve-seed-t8-seeds --order seed --replicates 20 --seed_stride 1
  $PY -I "$here/bench_score.py" $g $(ls -d "$here/../out/baselines/$g"/*/ | grep -v -e /time- -e /_score -e /ablation-) | tee "$here/../out/baselines/$g/score.txt"
done
