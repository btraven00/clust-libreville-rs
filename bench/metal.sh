#!/usr/bin/env bash
# metal.sh [-g 10k,100k,1M] [-n] [--] [cargo test args]: mirror this working tree to the Mac and run the GPU
# test harness there with --features gpu-metal (the same tests/gpu.rs as on CUDA). The tree is synced with
# rsync (tracked and untracked files, not the git-ignored build products and data; the remote .git and target
# are left alone), so uncommitted kernel edits are tested as they are. Plain rsync options only: macOS ships openrsync. -g: which bench/graphs to send (incremental; 1M is
# 350 MB once) for the ignored bench-graph test; -n: sync only.
#   bench/metal.sh                       # the synthetic GPU tests
#   bench/metal.sh -g 10k,100k,154k      # + the bench graphs with times (-- --ignored --nocapture is implied by -g)
#   bench/metal.sh -g 1M -- bench_graphs --ignored --nocapture
# REMOTE (ssh host) and REMOTE_DIR (the checkout on it) are required.
set -euo pipefail
cd "$(dirname "$0")/.."
remote=${REMOTE:?set REMOTE to the ssh host of the Mac} dir=${REMOTE_DIR:?set REMOTE_DIR to the checkout on it}
graphs= sync_only=
while getopts g:n o; do case $o in g) graphs=$OPTARG;; n) sync_only=1;; *) exit 2;; esac; done
shift $((OPTIND - 1))
ex=(); for x in .git target .venv vendor __pycache__ .pytest_cache '*.npz' '*.npy' '*.so' '*.o' ref bench/out bench/.cpu.lock Makevars Makevars.win .cargo; do ex+=(--exclude "$x"); done
rsync -a --delete "${ex[@]}" ./ "$remote:$dir/"
for g in ${graphs//,/ }; do
  f=bench/graphs/$([[ $g == 1M ]] && echo syn_1M || echo hao2021_$g).npz
  rsync -a "$f" "$remote:$dir/bench/graphs/"
done
[[ $sync_only ]] && exit 0
args=("$@")
[[ $graphs && ${#args[@]} -eq 0 ]] && args=(-- --ignored --nocapture)
# a tty when we have one (cargo's progress bars); the remote shell is non-interactive, hence cargo env
ssh $([[ -t 0 ]] && echo -t) "$remote" "cd $dir && source ~/.cargo/env && cargo test --release --features gpu-metal --test gpu ${args[*]}"
