#!/usr/bin/env bash
# run_baselines.sh <graph>...: untimed baseline members (labels + determinism) into bench/out/baselines/<graph>/, then bench_score.py.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PY=$PHD/sc-brrr/runs/.conda/613289a77143e229a42a609590d65392_/bin/python
gve() { (cd $PHD/brrr-gve && OMP_NUM_THREADS=$1 pixi run python -I "$here/run_member.py" --impl gve "${@:2}"); }
for g in "$@"; do
  $PY -I "$here/run_member.py" --impl igraph2 --graph $g --name igraph-niter2-seeds
  $PY -I "$here/run_member.py" --impl igraph-fix --graph $g --name igraph-fix-seeds
  $PY -I "$here/run_member.py" --impl leidenalg --graph $g --name leidenalg-seeds
  for o in input seed; do
    gve 1 --graph $g --name gve-$o-t1-same --order $o --replicates 10 --seed_stride 0
    gve 2 --graph $g --name gve-$o-t2-same --order $o --replicates 20 --seed_stride 0
    gve 8 --graph $g --name gve-$o-t8-same --order $o --replicates 20 --seed_stride 0
  done
  gve 1 --graph $g --name gve-seed-t1-seeds --order seed --replicates 20 --seed_stride 1
  gve 8 --graph $g --name gve-seed-t8-seeds --order seed --replicates 20 --seed_stride 1
  $PY -I "$here/bench_score.py" $g "$here/../out/baselines/$g"/[a-z]*/ | tee "$here/../out/baselines/$g/score.txt"
done
