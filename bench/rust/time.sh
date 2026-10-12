#!/bin/sh
: "${PHD:?set PHD to the directory holding the brrr-gve, sc-brrr and detleiden checkouts}"
# Timed runs, each holding the shared CPU lock: time.sh <graph> <impl> <threads> <niter> <reps> [extra ldbench flags]
# -> out/leiden-rs/<graph>/<impl>[<flags>]-time-n<niter>/<T>t/run<k>.{npy,json}; best-of = min time_s over runs.
set -e
cd "$(dirname "$0")"
g=$1; imp=$2; t=$3; ni=$4; reps=$5; shift 5
tag=$(echo "$*" | tr -d ' -'); d=../out/leiden-rs/$g/$imp${tag:+-$tag}-time-n$ni/${t}t; mkdir -p $d
for k in $(seq 1 $reps); do
    [ -f $d/run$k.json ] && continue
    flock $PHD/detleiden/bench/.cpu.lock target/release/ldbench run --graph ../graphs/$g.npz --impl $imp \
        --threads $t --seed 0 --niter $ni --out $d/run$k "$@" > /dev/null
done
python3 -c "import json,glob; r=[json.load(open(f)) for f in glob.glob('$d/run*.json')]; print('$d', 'best', min(x['time_s'] for x in r), 'n', len(r), 'k', r[0]['n_clusters'], 'Q', round(r[0]['quality_impl'],6), 'passes', r[0]['passes'], 'disc', r[0]['disconnected'])"
