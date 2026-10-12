#!/usr/bin/env bash
# Fixed-point variants benchmark (detleiden): strict / min-gain (reuse-colouring was benchmarked and removed, b87d839), 8 threads,
# under the CPU lock. Labels -> bench/out/detleiden/<graph>/<variant>/run<seed>.npy, one log line per run.
# usage: bench/fixpoint.sh LOG
cd "$(dirname "$0")/.." || exit 1
log=$1
run() { # graph variant seeds... -- flags
  local g=$1 v=$2; shift 2; local seeds=(); while [ "$1" != -- ]; do seeds+=("$1"); shift; done; shift
  mkdir -p bench/out/detleiden/$g/$v
  for s in "${seeds[@]}"; do
    echo "$g $v $(flock bench/.cpu.lock target/release/examples/run bench/graphs/$g.npz --seed $s --threads 8 "$@" --out bench/out/detleiden/$g/$v/run$s.npy)" >> "$log"
  done
}
for g in hao2021_10k hao2021_100k hao2021_154k; do
  S=$(seq 100 119)
  run $g strict $S --
  run $g gain1e-5 $S -- --min-gain 1e-5
done
for v in "strict --" "gain1e-5 -- --min-gain 1e-5"; do
  set -- $v; name=$1; shift; run syn_1M $name 100 101 102 "$@"
done
echo DONE >> "$log"
