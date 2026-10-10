#!/bin/sh
# Seed sweep (untimed): seeds.sh <size> -> out/leiden-rs/hao2021_<size>/<variant>-seeds/<T>t/run<k>.npy, seed = 99+k (k=1..20,
# i.e. the reference's seeds 100..119). Per-seed comparison of stock 1t/8t/16t finds thread-count flips; all feed ARI_ref.
set -e
cd "$(dirname "$0")"
g=../graphs/hao2021_$1.npz; out=../out/leiden-rs/hao2021_$1; B=target/release/ldbench
for k in $(seq 1 20); do
    for vt in stock:1 stock:8 stock:16 seqagg:8 sc:1; do
        v=${vt%:*}; t=${vt#*:}; d=$out/$v-seeds/${t}t; mkdir -p $d; [ -f $d/run$k.json ] && continue
        nice -n 19 $B run --graph $g --impl $v --threads $t --seed $((99+k)) --niter -1 --out $d/run$k > /dev/null
    done
done
