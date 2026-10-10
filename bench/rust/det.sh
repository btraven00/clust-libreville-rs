#!/bin/sh
# Determinism runs (untimed, no lock): det.sh <size> [reruns=20] [seed=0] [niter=-1]
# Writes bench/out/leiden-rs/hao2021_<size>/<variant>/<T>t/run<k>.{npy,json[,trace]}
#   stock  = crates.io leiden-rs 0.8.1, unmodified
#   trace  = stock + trace.patch (LEIDEN_TRACE=1: per-level coarse-graph hash incl. weight bits)
#   seqagg = stock + seqagg.patch (+ env-gated trace)
set -e
cd "$(dirname "$0")"
size=$1; reruns=${2:-20}; seed=${3:-0}; niter=${4:--1}
g=../graphs/hao2021_$size.npz
out=../out/leiden-rs/hao2021_$size
B=target/release/ldbench
one() { # variant impl threads k
    d=$out/$1/$3t; mkdir -p $d; [ -f $d/run$4.json ] && return 0
    LEIDEN_TRACE=1 nice -n 19 $B run --graph $g --impl $2 --threads $3 --seed $seed --niter $niter \
        --out $d/run$4 2> $d/run$4.trace > /dev/null
}
for v in stock seqagg trace; do
    for k in $(seq 1 $reruns); do one $v $v 8 $k; done
    for t in 1 2 4 16; do for k in 1 2 3; do one $v $v $t $k; done; done
done
