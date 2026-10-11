#!/usr/bin/env bash
# time_fair.sh [-r ROUNDS] [-t "1 8"] [-b DET_BIN] GRAPH...: "algo only" times of GVE vs detleiden (one iteration,
# seed 100), alternating: each round runs one fresh GVE process, then one fresh detleiden process, per thread count.
# No call is discarded as warm-up. Each process waits under bench/.cpu.lock until the machine is idle (as time_runs.sh).
# -> bench/out/time_fair/<stamp>.tsv (graph threads impl round algo_s wall_s) and a min/median/max summary on stdout.
# DET_BIN: build it from a clean worktree when someone else has uncommitted changes in src/.
set -euo pipefail
cd "$(dirname "$0")/.."
rounds=5 threads="1 8" bin=target/release/examples/run
while getopts r:t:b: o; do case $o in r) rounds=$OPTARG;; t) threads=$OPTARG;; b) bin=$OPTARG;; *) exit 2;; esac; done
shift $((OPTIND - 1))
LOCK=$PWD/bench/.cpu.lock
idle() { ! pgrep -f "python[^ ]* [^ ]*(score|runner)\.py" >/dev/null && [ "$(cut -d' ' -f4 /proc/loadavg | cut -d/ -f1)" -le 2 ]; }
export -f idle
m() { flock "$LOCK" bash -c 'until idle && sleep 1 && idle; do sleep 5; done; "$@"' _ "$@"; }
mkdir -p bench/out/time_fair
out=bench/out/time_fair/$(date +%Y%m%dT%H%M%S).tsv
echo "# $(date -Is) power=$(powerprofilesctl get 2>/dev/null || system76-power profile 2>/dev/null | sed -n "s/^Power Profile: //p" || echo ?) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?) det=$bin@$(git -C "$(dirname "$bin")" rev-parse --short HEAD 2>/dev/null || echo ?)" | tee "$out"
echo -e "graph\tthreads\timpl\tround\talgo_s\twall_s" >> "$out"
for g in "$@"; do
  for r in $(seq 1 "$rounds"); do
    for t in $threads; do
      # GVE: run_member's stdout ends "... time min W ... algo min A ..."; one replicate per process
      l=$(cd $PHD/brrr-gve && m env OMP_NUM_THREADS=$t pixi run python -I "$OLDPWD/bench/py/run_member.py" --impl gve --graph "$g" \
            --name time-fair-gve-t$t --order seed --replicates 1 2>/dev/null | tail -1)
      echo -e "$g\t$t\tgve\t$r\t$(sed -E 's/.*algo min ([0-9.]+)s.*/\1/' <<<"$l")\t$(sed -E 's/.*time min ([0-9.]+)s.*/\1/' <<<"$l")" | tee -a "$out"
      l=$(m "$bin" "bench/graphs/$g.npz" --seed 100 --niter 1 --threads "$t" --repeat 1)
      echo -e "$g\t$t\tdetleiden\t$r\t$(sed -E 's/.* time_s ([0-9.]+) .*/\1/' <<<"$l")\t" | tee -a "$out"
    done
  done
done
python3 - "$out" <<'EOF'
import csv, statistics, sys
rows = [r for r in csv.DictReader((l for l in open(sys.argv[1]) if not l.startswith("#")), delimiter="\t")]
keys = sorted({(r["graph"], int(r["threads"]), r["impl"]) for r in rows})
print("graph\tthreads\timpl\tn\tmin\tmedian\tmax")
for k in keys:
    x = [float(r["algo_s"]) for r in rows if (r["graph"], int(r["threads"]), r["impl"]) == k]
    print(f"{k[0]}\t{k[1]}\t{k[2]}\t{len(x)}\t{min(x):.3f}\t{statistics.median(x):.3f}\t{max(x):.3f}")
EOF
