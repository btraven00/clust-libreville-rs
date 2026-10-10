# ldbench: leiden-rs / single-clustering bench (detleiden phase 0)

```sh
./setup.sh                                   # vendor/ from the crates.io tarball + patches
CARGO_TARGET_DIR=target cargo build --release
target/release/ldbench run --graph ../graphs/hao2021_100k.npz --impl seqagg --threads 8 --seed 0 --out ../out/leiden-rs/hao2021_100k/seqagg/8t/run1
./det.sh 100k 20                             # determinism: 20 reruns at 8t, plus 3 each at 1/2/4/16t, for stock/seqagg/trace
python3 summarize.py ../out/leiden-rs/hao2021_100k
target/release/ldbench prop --graphs 40      # disconnected final communities on random graphs
```

- `--impl stock`: crates.io leiden-rs 0.8.1, unmodified (feature `rayon`).
- `--impl seqagg`: `seqagg.patch`, the candidate upstream PR. `aggregate_edges_parallel` emits `(key, w)` in vertex order, runs a stable `par_sort`, then sums each key sequentially. That makes it bit-identical to `aggregate_edges_sequential` at any thread count. Also carries `trace.patch`.
- `--impl trace`: stock plus `trace.patch`. With `LEIDEN_TRACE=1`, it prints to stderr, per level: the hash of the coarse graph (structure and exact weight bits), the disconnected communities before and after refinement, the colours and rounds, and the phase times. Off, the cost is one cached env check per level.
- `--impl sc`: single-clustering 0.7.0, 1 thread.
- Threads: `--threads T`, else `RAYON_NUM_THREADS`, else 1. An explicit `rayon::ThreadPool` is built, and the run is wrapped in `pool.install`.
- Output: `PREFIX.npy` (int64 labels in graph row order, renumbered by first occurrence) and `PREFIX.json`. `time_s` is the clustering call alone: graph build and I/O are excluded and reported as `build_s` / `load_s`. `quality_impl` is the implementation's own modularity, not the shared scorer.

## Mapping to igraph `community_leiden(objective_function="modularity", resolution=1, beta=0.01, n_iterations=N)`

| igraph | leiden-rs 0.8.1 | single-clustering 0.7.0 |
|---|---|---|
| modularity, resolution 1 | `QualityType::Modularity`, `resolution(1.0)` | `ObjectiveKind::Rb { resolution: 1.0 }` |
| weights (f64) | f64, built from the upper triangle of the symmetric CSR | **f32** storage (`from_csr_parts`), f64 arithmetic |
| `seed` | `seed(seed + pass)`, one `StdRng` per pass | `seed` (ChaCha8, one stream) |
| `n_iterations` = one full Leiden run | one `Leiden::run` = all levels. **`max_iterations` (default 100) counts levels**, not iterations | `n_iterations` |
| `n_iterations=-1` (until no change) | `--niter -1`: passes of `run_with_initial_partition(prev)` until the labels are identical (cap 1000) | `n_iterations=1000, patience=1` (stops on the first pass that finds nothing, above `tolerance=1e-9` relative) |
| `beta=0.01` (θ-random refinement) | **none**: refinement is a greedy argmax | `refine_randomness=0.01` |
| — | thresholds left at their defaults: parallel moving at ≥ 2000 CSR slots and ≥ 100 nodes, parallel aggregation at ≥ 10 000 slots. `--seqmove` / `--aggseq` set them to `usize::MAX` | — |

Known gap: leiden-rs reseeds each pass (`seed + pass`), whereas igraph uses one RNG stream across iterations. Both are deterministic. They give different trajectories, not different semantics.
