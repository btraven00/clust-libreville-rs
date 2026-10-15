# clust-libreville-rs: design

A parallel, seeded Leiden (Traag, Waltman & van Eck 2019)[^traag2019] for modularity with a resolution γ.

The primary use case are single-cell kNN graphs (10⁴–10⁶ vertices, degree ~10–30), for other ranges YMMV.


## The guarantee

**Same graph, same options, same seed: the same labels, at any thread count; on
the CPU, across machines by design (checked on x86_64 and aarch64).**

- Thread count: tested bit for bit (labels and per-level graph hashes) at 1, 2, 3, 4, 8 and 16 threads
  (`tests/threads.rs`, the hao2021 lock, a random-graph harness).
- Machine: no step that decides a label depends on the platform's libm or on FMA contraction (below);
  the scores (NVI, AMI, `assess`, `neosbm`) do use `ln` and may differ in the last bits. The evidence
  is narrow: the fixture labels in `tests/fixtures/` (3k cells, small enough that the parallel
  colouring never runs) were produced on x86_64 and are reproduced on aarch64 (Apple M1, `cargo test`),
  and CI compares every wheel (Linux and macOS, x86_64 and aarch64) against them
  (`python/tests/test_fixtures.py`). The hao2021 locks have been run on x86_64 only.
- GPU (experimental): with `Apply::OrderFree` the CUDA backend gives the CPU's labels bit for bit
  (`tests/gpu.rs`, one RTX 2000 Ada). It relies on f64 kernels rounding as the CPU does: cubecl fuses
  `a * b + c` into FMAs by default, so this repository builds a patched cubecl-llvm without them
  (`vendor/README.md`; no `fma` left in the PTX, no measurable cost). Metal/WGSL have no
  f64 and run the kernels in f32 (`gpu-f32`): repeatable on a device, equality with the CPU not guaranteed
  (on hao2021 100k, one iteration, f32 kernels gave the CPU's labels on CUDA and on Metal (M1); run to
  convergence on Metal they differed, Q 0.894627 vs 0.894672).

Determinism is not seed independence: different seeds give different partitions, as in every Leiden.

## How determinism is achieved

Two rules, applied everywhere:

1. **Every random choice is a pure function of (seed, stream, step, vertex)** (`src/rng.rs`, a SplitMix64
   counter-based generator). No RNG stream is consumed in an order that depends on scheduling.
2. **Every floating-point sum runs in a fixed order**, independent of how work is split: fixed-size chunks
   combined sequentially, order-preserving collects, no float atomics, no `mul_add`. Idempotent flag writes
   may race.

Stage by stage (each is a GVE-Leiden kernel made deterministic):

| stage | GVE-Leiden | here |
|---|---|---|
| local moving | asynchronous sweep over live labels and totals | vertices coloured into independent sets (greedy first-fit over a seeded order, computed in parallel by Jones–Plassmann, which reproduces the sequential colouring); per colour group, every vertex *decides* its best move in parallel against the frozen state, then moves are *applied* in group order |
| apply rule | — | `Apply::Sequential` (default): in-order re-check against the updated totals. `Apply::OrderFree`: each move accepted against the group's starting totals made pessimistic by all earlier proposals (per-community prefix sums, as in Gottesbüren & Hamann's deterministic Louvain), so the result depends on the group's seeded order but not on scheduling or summation order, and a GPU can reproduce it. Its refinement re-checks the gain, not the target cluster's well-connectedness, after earlier merges in the group |
| pruning | affected flags on neighbours of moved vertices | the same, set after each group, so each group's work set is deterministic |
| refinement | greedy, singletons only, atomic compare-and-undo | parallel across communities, sequential inside each; singletons only, constrained to the parent community (Algorithm 3); draw ∝ exp(Δ/θ) from the counter RNG; θ = 0 is GVE's greedy rule |
| exp in refinement | — | graspologic-native's `(1 + x/256)^256` by eight squarings: IEEE multiplies only, bit-identical on every platform (`rng::exp_neg`) |
| community members | atomic slot counter | counting sort in vertex-id order |
| aggregation | one coarse row per community via per-thread hashtables | the same, with members in id order and neighbours in CSR order, so each row's sums and column order are fixed |
| totals for OrderFree / GPU | atomic double adds | fixed-point integers (scale 2²⁴, `moving::FX`): integer sums are exact and associative, so atomics in any order give the same bits. The weights (and θ) are first scaled by the power of two that puts 2m in [2^27, 2^28) (exact), so the totals neither lose small weights nor overflow, at any input scale |
| final labels | — | 0.. by community size (largest first), then smallest member id |

Debugging aid: `CLUST_LIBREVILLE_TRACE=1` prints per level a hash of the coarse graph (structure and exact weight
bits), colours, rounds, moves and phase times. The first level whose hash differs between two thread counts
is where a determinism bug lives.

## Structure

- **`Graph`** (`src/graph.rs`): a validated symmetric CSR. Weights stored as f32 when every input weight is
  exactly an f32 (scanpy connectivities are), else f64; all arithmetic is f64. Validation runs in parallel;
  symmetry is checked by a mirror lookup on canonical rows, else an O(m) transpose.
  `Graph::from_csr_trusted` skips only the symmetry check, for graphs symmetric by construction.
  A self-loop counts once in k_v (igraph counts it twice), so on graphs with a diagonal (Seurat's SNN
  graph) partitions differ from igraph's and leidenalg's; scanpy's connectivities have none.
- **`Backend`** (`src/backend.rs`): `colour` / `move_nodes` / `refine` / `aggregate`. Contract: every
  kernel's output is a pure function of its inputs, never of thread count or scheduling. The level loop,
  fixed point and canonical labels (`src/lib.rs`) are shared by all backends.
  - `Cpu`: rayon.
  - `gpu` (experimental; cubecl, features `gpu-cuda`, `gpu-wgpu`, `gpu-metal`): local moving,
    refinement and aggregation on the device for level 0 from 50k vertices and coarse levels from 500k;
    smaller levels on the CPU. The host reads one move counter per round; CUDA rounds are captured and
    replayed as a graph. No speedup over the CPU yet (converged 1M: 21 s vs 24 s; 100k: slower).
- **Threads** (`with_threads`, `Options.threads`): each calling thread keeps a cache of rayon pools by size,
  so repeated calls spawn no threads and an ensemble of single-threaded runs on N OS threads uses exactly N
  cores. The cache records the process id and is rebuilt in a forked child (R `mclapply`, Python
  `multiprocessing` with fork), whose copied pools have no worker threads.
- **Stopping** (`leiden_run` → `LeidenRun { labels, iterations, stop, modularity }`):
  - `n_iterations < 0` (default): iterate until an iteration changes nothing, really (igraph before C core
    0.10.17 ran one iteration for `n_iterations = -1`). Capped at 1000 iterations (`Stop::IterationCap`).
  - `min_gain`: also stop when an iteration gains less modularity than this. The fast mode; non-inferior
    to strict convergence (below).
  - `Stop::{Converged, MinGain, IterationCap}` says which.

## Tests

`cargo test --release` (CI: `.github/workflows/rust.yml`, Linux x86_64 and macOS aarch64, with
`cargo clippy --all-targets -- -D warnings`), about 12 s:

- unit tests in the modules: the parallel colouring equals the sequential greedy pass, refined clusters
  are connected inside their community, the O(m) symmetry check agrees with sorting, CHAMP's envelope,
  NVI/AMI against sklearn and PyGenStability;
- `tests/leiden.rs`: small known graphs (karate optimum, cliques, components), connectedness of
  communities on random graphs for both `Apply` rules, the paper's guarantees at convergence by brute
  force (no vertex gains by moving, no two communities gain by merging) for both rules and several γ,
  invariance to the weights' scale, degenerate graphs (empty, no edges, self-loops), the trusted graph
  build, `leiden_run`, CHAMP against brute force;
- `tests/threads.rs`: labels and every level's graph hash identical at 1, 2, 3, 4, 8 and 16 threads, both
  `Apply` rules, random graphs, and a 120k-vertex graph (above `JP_MIN_N`, so the parallel colouring
  runs); `-- --ignored`: the hao2021 graphs against locked label hashes;
- `tests/fixtures.rs`: the hao3k labels shared with the R and Python tests (regression locks: produced
  by this code, `bench/py/make_fixtures.py`), plus an `OrderFree` label lock;
- `tests/pools.rs`: the per-thread pool cache survives `fork()`, and `with_threads` around `leiden` does
  not nest a second pool.

Graph builders and the .npz loader shared by tests, examples and the bench live in `tests/common/mod.rs`.

**GPU** (needs a device; not in CI): `tests/gpu.rs` is the one harness for every runtime,
`cargo test --release --features gpu-cuda --test gpu` (or `gpu-metal`, `gpu-wgpu`). It runs the device
against the CPU reference (`leiden` with `Apply::OrderFree`) on synthetic graphs that force each path:
level 0 on the device, every level on the device, hub rows, a row too long for the hub kernels, f64
weights. With f64 kernels (CUDA) labels and level hashes must match bit for bit, so a failure names the
first level that differs; with `gpu-f32` (Metal) two runs must agree and the partition must be close to
the CPU's (modularity within 1e-3, NVI below 0.1, both printed). `-- --ignored --nocapture` adds the
bench graphs (`bench/graphs`, or `CLUST_LIBREVILLE_GRAPHS`) with best-of-3 times of the CPU default, the CPU
reference and the GPU. `bench/metal.sh [-g 10k,100k,154k,1M]` mirrors the working tree (and the chosen
graphs) to the Mac over rsync and runs that harness there with `gpu-metal`.

Knobs read from the environment: `CLUST_LIBREVILLE_TRACE` (per-iteration, per-level and per-phase diagnostics,
read once per process), `CLUST_LIBREVILLE_DECIDE=thread` (the thread-per-vertex decide kernel instead of the
plane kernel, for A/B), `CLUST_LIBREVILLE_NO_GRAPH` (no CUDA graph capture), `CLUST_LIBREVILLE_GRAPHS` (the bench
graphs' directory), `RAYON_NUM_THREADS` (the default thread count).

## Ensembles: CHAMP and agreement (experimental)

The interface and the scores may change.

`champ` (`src/champ.rs`) implements CHAMP (Weir, Emmons, Gibson, Taylor & Mucha 2017)[^weir2017]. For a fixed partition, modularity is a line in γ, Q(γ) = a − γb
(`modularity_line`); the best partition of a pool at each γ is the upper envelope of the lines. `champ_grid`
runs Leiden over resolutions × seeds and returns each envelope partition with its exact γ interval, which
is optimal relative to the pool only.

Each kept partition is scored by agreement (`Agreement`) under a `Metric`:

- `to_prev` / `to_neighbours`: with the adjacent envelope partitions (a plateau vs a boundary);
- `to_seeds`: with the other seeds' partitions at the same grid resolution (algorithmic noise).

Metrics (`src/compare.rs`):

- **NVI** (default) = VI / H(a, b), PyGenStability's normalisation. A metric (triangle inequality), so steps
  along the envelope add up; not corrected for chance.
- **AMI** with max normalisation (Vinh, Epps & Bailey 2010[^vinh2010]; as in the CHAMP paper): corrected for chance
  agreement between partitions into many small communities.

Determinism makes the whole ensemble, and therefore the envelope and its scores, reproducible.

## Validation

`cargo build --release --example run && uv run bench/validate/validate.py` recomputes this section (about
15 minutes on 16 cores). It downloads the graphs from the Hugging Face dataset
`btraven/clust-libreville-graphs` (hao2021 PBMC kNN graphs at 10k, 50k, 100k and 154k cells; build recipe
and checksums in its card), runs 20 seeds of each clust-libreville-rs arm and of two converged references
(leidenalg, n_iterations = −1; igraph iterated one iteration at a time to a fixed point), and writes
`bench/out/validate/report.md`. The procedure is in the script's docstring.

- **Quality:** mean modularity equals the references' within a seed SD (SDs 0.0003–0.0009):

  | | strict | min_gain 1e-5 | OrderFree | leidenalg | igraph fixed point |
  |---|---|---|---|---|---|
  | 10k | 0.86350 | 0.86347 | 0.86340 | 0.86307 | 0.86334 |
  | 50k | 0.88768 | 0.88758 | 0.88765 | 0.88766 | 0.88757 |
  | 100k | 0.89472 | 0.89460 | 0.89460 | 0.89493 | 0.89484 |
  | 154k | 0.89846 | 0.89837 | 0.89832 | 0.89844 | 0.89840 |

- **Non-inferiority** (ARI to each reference against the reference's own seed-to-seed floor, jackknife
  over both seed sets, δ = 0.024 frozen): all three arms are non-inferior to both references at every
  size, D between −0.011 and +0.005. With δ recomputed on each reference (0.021–0.033), the one exception
  is OrderFree vs leidenalg at 100k (LB −0.023 vs δ 0.021): inconclusive.
- **Reproducibility** (10 same-seed runs at 8 threads): clust-libreville-rs gives one partition at every
  size. networkit's ParallelLeiden (as scanpy PR #4170 calls it) and GVE-Leiden (the conda package
  `gve-leiden` from the almost-conductor channel: upstream at f741a40, the commit benchmarked here; no
  seed, so same input) give 10 distinct partitions at every size, mean pairwise ARI 0.88–0.96 and
  0.87–0.89, about a seed change. Exact values vary from run to run, which is the point. cuGraph behaved
  the same way, measured outside this script; single-threaded igraph and leidenalg are reproducible but
  serial.

## Lineage

- **GVE-Leiden** (Sahu 2024[^sahu2024], MIT): the per-stage kernels, pruning, per-thread scratch,
  aggregation scheme, tolerance rule.
- **leiden-rs** 0.8.1 (MIT/Apache-2.0): greedy colouring and the colour-group decide/apply skeleton.
- **Jones–Plassmann** colouring, with Hasenplaugh et al. (SPAA 2014)[^hasenplaugh2014] for reproducing greedy order.
- **graspologic-native** (MIT): the portable exp.
- **Mt-KaHyPar**'s deterministic synchronous Louvain (Gottesbüren & Hamann 2022)[^gottesburen2022]: the closest prior art.
- **igraph**: `beta`/θ in edge-weight units; the `n_iterations` semantics.

Borrowed code is cited where it is used.

## References

[^traag2019]: Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. *Sci Rep*. 2019;9:5233. doi:[10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z). arXiv:[1810.08473](https://arxiv.org/abs/1810.08473).
[^weir2017]: Weir WH, Emmons S, Gibson R, Taylor D, Mucha PJ. Post-processing partitions to identify domains of modularity optimization. *Algorithms*. 2017;10(3):93. doi:[10.3390/a10030093](https://doi.org/10.3390/a10030093). arXiv:[1706.03675](https://arxiv.org/abs/1706.03675).
[^vinh2010]: Vinh NX, Epps J, Bailey J. Information theoretic measures for clusterings comparison: variants, properties, normalization and correction for chance. *J Mach Learn Res*. 2010;11:2837–2854. [jmlr.org/papers/v11/vinh10a](https://jmlr.org/papers/v11/vinh10a.html).
[^sahu2024]: Sahu S. GVE-Leiden: fast Leiden algorithm for community detection in shared memory setting. In: *Proc 53rd Int Conf Parallel Processing (ICPP)*. ACM; 2024:11–20. doi:[10.1145/3673038.3673146](https://doi.org/10.1145/3673038.3673146). arXiv:[2312.13936](https://arxiv.org/abs/2312.13936).
[^hasenplaugh2014]: Hasenplaugh W, Kaler T, Schardl TB, Leiserson CE. Ordering heuristics for parallel graph coloring. In: *Proc 26th ACM Symp Parallelism in Algorithms and Architectures (SPAA)*. ACM; 2014:166–177. doi:[10.1145/2612669.2612697](https://doi.org/10.1145/2612669.2612697).
[^gottesburen2022]: Gottesbüren L, Hamann M. Deterministic parallel hypergraph partitioning. In: *Euro-Par 2022: Parallel Processing*. LNCS 13440. Springer; 2022:301–316. doi:[10.1007/978-3-031-12597-3_19](https://doi.org/10.1007/978-3-031-12597-3_19). arXiv:[2112.12704](https://arxiv.org/abs/2112.12704).
