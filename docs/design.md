# clust-libreville-rs: design

A parallel, seeded Leiden (Traag, Waltman & van Eck 2019) for modularity with a resolution γ, built for
single-cell kNN graphs (10⁴–10⁶ vertices, degree ~10–30). This file records what is built and why; the
code is the reference for how.

## The guarantee

**Same graph, same options, same seed: the same labels, at any thread count; on the CPU, across machines by design (tested on x86_64 and aarch64).**

- Thread count: tested bit for bit (labels and per-level graph hashes) at 1, 2, 3, 4, 8 and 16 threads
  (`tests/threads.rs`, the hao2021 lock, a random-graph harness).
- Machine: no step depends on the platform's libm or on FMA contraction (below). The fixture labels in
  `tests/fixtures/` were produced on x86_64 and are reproduced on aarch64 (Apple M1, `cargo test`); CI
  compares every wheel (Linux and macOS, x86_64 and aarch64) against them (`python/tests/test_fixtures.py`).
- GPU: with `Apply::OrderFree` the CUDA backend gives the CPU's labels bit for bit. Metal/WGSL have no
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
| apply rule | — | `Apply::Sequential` (default): in-order re-check against the updated totals. `Apply::OrderFree`: each move accepted against the group's starting totals made pessimistic by all earlier proposals (per-community prefix sums), so the result is independent of order and a GPU can reproduce it |
| pruning | affected flags on neighbours of moved vertices | the same, set after each group, so each group's work set is deterministic |
| refinement | greedy, singletons only, atomic compare-and-undo | parallel across communities, sequential inside each; singletons only, constrained to the parent community (Algorithm 3); draw ∝ exp(Δ/θ) from the counter RNG; θ = 0 is GVE's greedy rule |
| exp in refinement | — | graspologic-native's `(1 + x/256)^256` by eight squarings: IEEE multiplies only, bit-identical on every platform (`rng::exp_neg`) |
| community members | atomic slot counter | counting sort in vertex-id order |
| aggregation | one coarse row per community via per-thread hashtables | the same, with members in id order and neighbours in CSR order, so each row's sums and column order are fixed |
| totals for OrderFree / GPU | atomic double adds | fixed-point integers (scale 2²⁴, `moving::FX`): integer sums are exact and associative, so atomics in any order give the same bits |
| final labels | — | 0.. by community size (largest first), then smallest member id |

Debugging aid: `DETLEIDEN_TRACE=1` prints per level a hash of the coarse graph (structure and exact weight
bits), colours, rounds, moves and phase times. The first level whose hash differs between two thread counts
is where a determinism bug lives.

## Structure

- **`Graph`** (`src/graph.rs`): a validated symmetric CSR. Weights stored as f32 when every input weight is
  exactly an f32 (scanpy connectivities are), else f64; all arithmetic is f64. Validation runs in parallel;
  symmetry is checked by a mirror lookup on canonical rows, else an O(m) transpose.
  `Graph::from_csr_trusted` skips only the symmetry check, for graphs symmetric by construction.
- **`Backend`** (`src/backend.rs`): `colour` / `move_nodes` / `refine` / `aggregate`. Contract: every
  kernel's output is a pure function of its inputs, never of thread count or scheduling. The level loop,
  fixed point and canonical labels (`src/lib.rs`) are shared by all backends.
  - `Cpu`: rayon.
  - `gpu` (cubecl, features `gpu-cuda`, `gpu-wgpu`, `gpu-metal`): level-0 local moving on the device,
    everything else on the CPU; the host reads one move counter per round; CUDA rounds are captured and
    replayed as a graph.
- **Threads** (`with_threads`, `Options.threads`): each calling thread keeps a cache of rayon pools by size,
  so repeated calls spawn no threads and an ensemble of single-threaded runs on N OS threads uses exactly N
  cores. The cache records the process id and is rebuilt in a forked child (R `mclapply`, Python
  `multiprocessing` with fork), whose copied pools have no worker threads.
- **Stopping** (`leiden_run` → `LeidenRun { labels, iterations, stop, modularity }`):
  - `n_iterations < 0` (default): iterate until an iteration changes nothing, really (igraph before C core
    0.10.17 ran one iteration for `n_iterations = -1`). Capped at 1000 iterations.
  - `min_gain`: also stop when an iteration gains less modularity than this. The fast mode; non-inferior
    to strict convergence (below).
  - `Stop::{Converged, MinGain, IterationCap}` says which.

## Ensembles: CHAMP and agreement (experimental)

The interface and the scores may change.

`champ` (`src/champ.rs`) implements CHAMP (Weir, Emmons, Gibson, Taylor & Mucha 2017,
doi:10.3390/a10030093). For a fixed partition, modularity is a line in γ, Q(γ) = a − γb
(`modularity_line`); the best partition of a pool at each γ is the upper envelope of the lines. `champ_grid`
runs Leiden over resolutions × seeds and returns each envelope partition with its exact γ interval, which
is optimal relative to the pool only.

Each kept partition is scored by agreement (`Agreement`) under a `Metric`:

- `to_prev` / `to_neighbours`: with the adjacent envelope partitions (a plateau vs a boundary);
- `to_seeds`: with the other seeds' partitions at the same grid resolution (algorithmic noise).

Metrics (`src/compare.rs`):

- **NVI** (default) = VI / H(a, b), PyGenStability's normalisation. A metric (triangle inequality), so steps
  along the envelope add up; not corrected for chance.
- **AMI** with max normalisation (Vinh, Epps & Bailey 2010; as in the CHAMP paper): corrected for chance
  agreement between partitions into many small communities.

Determinism makes the whole ensemble, and therefore the envelope and its scores, reproducible.

## Validation

- **Quality:** modularity equals the converged references (leidenalg; igraph iterated to a fixed point) on
  hao2021 10k/100k (e.g. 100k: mean Q 0.89479 vs 0.89482 / 0.89480), and agreement with them sits on their
  own seed-to-seed floor.
- **Non-inferiority** (ARI to the converged references, jackknife over seed sets, δ = 0.024): strict
  convergence, `min_gain = 1e-5` and `Apply::OrderFree` are each non-inferior at 10k, 100k and 154k.
- **Reproducibility of other parallel Leidens** (same seed, repeated runs at 8 threads): GVE-Leiden,
  networkit's ParallelLeiden and cuGraph give different partitions from run to run (same-seed ARI
  0.87–0.99, about their seed-to-seed spread); single-threaded igraph and leidenalg are reproducible but
  serial.

## Lineage

- **GVE-Leiden** (Sahu 2023, arXiv:2312.13936, MIT): the per-stage kernels, pruning, per-thread scratch,
  aggregation scheme, tolerance rule.
- **leiden-rs** 0.8.1 (MIT/Apache-2.0): greedy colouring and the colour-group decide/apply skeleton.
- **Jones–Plassmann** colouring, with Hasenplaugh et al. (SPAA 2014) for reproducing greedy order.
- **graspologic-native** (MIT): the portable exp.
- **Mt-KaHyPar**'s deterministic synchronous Louvain (arXiv:2112.12704): the closest prior art.
- **igraph**: `beta`/θ in edge-weight units; the `n_iterations` semantics.

Borrowed code is cited where it is used.
