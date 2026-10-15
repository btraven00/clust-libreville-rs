//! clust-libreville-rs: seeded, deterministic Leiden community detection (modularity, resolution γ).
//!
//! Same graph, same options, same seed: the same labels. Every random choice is a pure function
//! of (seed, step, vertex) and every floating-point sum runs in a fixed order, so the result
//! does not depend on the order in which work is done (phase 2 relies on it for thread-count
//! invariance). See docs/design.md.

mod aggregate;
mod assess;
mod backend;
mod champ;
mod compare;
#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
pub mod gpu;
mod graph;
mod moving;
mod refine;
mod rng;

pub use champ::{champ, champ_grid, Agreement, Domain, Scored};
pub use assess::{block_stats, dcsbm_loglik, description_length, label_surprise, ln_gamma, neosbm, poisson_log_tail, significance, BlockStats, NeoSbm};
pub use compare::{ami, ecs, nvi, Metric};
pub use graph::Graph;
use backend::{Backend, Cpu};
use graph::{Csr, Repr, Weight};

/// How local moving applies the moves decided (in parallel) for a colour group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Apply {
    /// In order, each re-checked against the totals as they are after the earlier moves: the
    /// CPU fast path (default).
    #[default]
    Sequential,
    /// Rule C: each accepted against the totals frozen at the group's start, made pessimistic by
    /// all earlier proposals (prefix sums per community). Order-free, so a GPU reproduces it in
    /// parallel bit for bit: the GPU backend uses it, and on the CPU it gives the GPU's labels.
    /// Non-inferior to `Sequential` (docs/design.md, Validation); ~5% slower on the CPU.
    OrderFree,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Modularity resolution γ.
    pub resolution: f64,
    pub seed: u64,
    /// Full Leiden iterations, each starting from the previous result; negative: until an
    /// iteration changes nothing (really, unlike igraph < 0.10.17).
    pub n_iterations: i64,
    /// Refinement randomness (igraph's `beta`), in edge-weight units; 0: greedy (GVE).
    pub theta: f64,
    /// Local moving stops when a round gains at most this much modularity; the tolerance is
    /// divided by 10 at each level (GVE-Leiden's rule, which starts at 1e-2). 0: move until
    /// nothing moves.
    pub tolerance: f64,
    /// With `n_iterations < 0`: also stop when an iteration gains less than this modularity.
    /// 0 (default): strictly until an iteration changes nothing (comparable to igraph/leidenalg).
    pub min_gain: f64,
    /// See [`Apply`].
    pub apply: Apply,
    /// Worker threads; None: `RAYON_NUM_THREADS`, else `std::thread::available_parallelism()`
    /// (which respects cgroup quotas). The labels do not depend on it.
    ///
    /// Ensembles in shared memory (many runs over one graph, e.g. Markov stability scans of
    /// scales × seeds): run each call on its own OS thread with `threads: Some(1)` (or a small
    /// count) instead of nesting calls inside a rayon pool. Every calling thread keeps its own
    /// pool, built on its first call and reused afterwards, so repeated calls cost no thread spawns
    /// and the ensemble uses exactly (callers × threads) workers. Calls from inside a rayon pool
    /// also work, but a caller blocked on its own pool may pick up other ensemble jobs: the
    /// thread count is not respected and memory grows with the number of runs in flight.
    pub threads: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options { resolution: 1.0, seed: 0, n_iterations: -1, theta: 0.01, tolerance: 0.0, min_gain: 0.0, apply: Apply::Sequential, threads: None }
    }
}

const MAX_ITERATIONS: i64 = 1000; // igraph/igraph#2934: fail rather than loop forever
const MAX_LEVELS: u64 = 1000;

/// `DETLEIDEN_TRACE` is set: per-iteration, per-level and per-phase diagnostics on stderr
/// (docs/design.md). Read once per process.
pub(crate) fn trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("DETLEIDEN_TRACE").is_some())
}

/// Community labels 0.., ordered by community size (largest first), then by smallest member.
pub fn leiden(g: &Graph, o: &Options) -> Vec<u32> {
    run(g, o, None).0
}

/// Why a run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The last iteration changed nothing: a fixed point.
    Converged,
    /// `min_gain`: the last iteration gained less modularity than that.
    MinGain,
    /// `n_iterations` iterations ran, the last of which still changed something.
    IterationCap,
}

/// A run's labels and how it ended.
#[derive(Clone, Debug)]
pub struct LeidenRun {
    pub labels: Vec<u32>,
    /// Leiden iterations run (the last one included).
    pub iterations: usize,
    pub stop: Stop,
    /// Modularity of `labels` at the run's resolution.
    pub modularity: f64,
}

impl LeidenRun {
    /// Whether the run reached a fixed point (its last iteration changed nothing).
    pub fn converged(&self) -> bool {
        self.stop == Stop::Converged
    }
}

/// `leiden`, with how the run ended: iterations, why it stopped, modularity.
pub fn leiden_run(g: &Graph, o: &Options) -> LeidenRun {
    let (labels, iterations, stop) = run(g, o, None);
    let modularity = modularity(g, &labels, o.resolution);
    LeidenRun { labels, iterations, stop, modularity }
}

/// `leiden`, plus the hash of every level's graph (structure and exact weight bits), in order:
/// for checking that nothing, not even a last bit, depends on the thread count.
#[doc(hidden)]
pub fn leiden_traced(g: &Graph, o: &Options) -> (Vec<u32>, Vec<u64>) {
    let mut h = Vec::new();
    let l = run(g, o, Some(&mut h)).0;
    (l, h)
}

/// `leiden` with level-0 local moving on the GPU (cubecl; the build's runtime), always with
/// `Apply::OrderFree`: on CUDA (f64) the labels equal `leiden` with `apply: Apply::OrderFree`.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
pub fn leiden_gpu(g: &Graph, o: &Options) -> Vec<u32> {
    run_gpu(g, o, gpu::Gpu::new(), None).0
}

/// `leiden_gpu` with every level's graph hash (as `leiden_traced`), running on the device the
/// levels with at least `min_n` (level 0) or `coarse_min_n` (coarse levels) vertices; the rest run
/// on the host. For the GPU tests (tests/gpu.rs), which force every path.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
#[doc(hidden)]
pub fn leiden_gpu_traced(g: &Graph, o: &Options, min_n: usize, coarse_min_n: usize) -> (Vec<u32>, Vec<u64>) {
    let mut h = Vec::new();
    let l = run_gpu(g, o, gpu::Gpu::with_thresholds(min_n, coarse_min_n), Some(&mut h)).0;
    (l, h)
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
fn run_gpu(g: &Graph, o: &Options, gpu: gpu::Gpu, hashes: Option<&mut Vec<u64>>) -> (Vec<u32>, usize, Stop) {
    let o = &Options { apply: Apply::OrderFree, ..o.clone() }; // the rule the device runs
    with_pool(o, || match &g.0 {
        Repr::F32(c) => iterate(&gpu, c, o, hashes),
        Repr::F64(c) => iterate(&gpu, c, o, hashes),
    })
}

/// Run `f` on `threads` worker threads (None: as `Options::threads`): the same cached pool `leiden`
/// uses. For bindings, so that building the graph (its validation is parallel) also stays within
/// the caller's thread count instead of rayon's global pool.
pub fn with_threads<T: Send>(threads: Option<usize>, f: impl FnOnce() -> T + Send) -> T {
    with_pool(&Options { threads, ..Default::default() }, f)
}

fn with_pool<T: Send>(o: &Options, f: impl FnOnce() -> T + Send) -> T {
    let threads = o.threads.unwrap_or_else(|| {
        std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&t| t > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    });
    if rayon::current_thread_index().is_some() && rayon::current_num_threads() == threads {
        // already on a worker of a pool of that size (`with_threads` around `leiden`, as the
        // bindings do; an ensemble worker's own pool): run there, don't nest a second pool
        return f();
    }
    type Pools = (u32, Vec<(usize, std::rc::Rc<rayon::ThreadPool>)>);
    thread_local! {
        // this thread's pools by size, and the process they were built in; ponytail: never freed (one
        // per size a thread asks for)
        static POOLS: std::cell::RefCell<Pools> = const { std::cell::RefCell::new((0, Vec::new())) };
    }
    let pool = POOLS.with(|p| {
        let mut p = p.borrow_mut();
        let pid = std::process::id();
        if p.0 != pid {
            // fork() (R's mclapply, Python's multiprocessing) copies this cache but not the pools'
            // worker threads, so a cached pool would hang the child. Leak the copies (dropping one
            // may touch state the parent's threads held at the fork) and build new pools.
            std::mem::forget(std::mem::replace(&mut *p, (pid, Vec::new())));
        }
        let p = &mut p.1;
        if let Some((_, pool)) = p.iter().find(|(t, _)| *t == threads) {
            return pool.clone();
        }
        let pool = std::rc::Rc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool"));
        p.push((threads, pool.clone()));
        pool
    });
    pool.install(f)
}

fn run(g: &Graph, o: &Options, hashes: Option<&mut Vec<u64>>) -> (Vec<u32>, usize, Stop) {
    with_pool(o, || match &g.0 {
        Repr::F32(c) => iterate(&Cpu, c, o, hashes),
        Repr::F64(c) => iterate(&Cpu, c, o, hashes),
    })
}

/// The iterations; returns the labels, the iterations run and why it stopped.
fn iterate<B: Backend, W: Weight>(b: &B, g: &Csr<W>, o: &Options, mut hashes: Option<&mut Vec<u64>>) -> (Vec<u32>, usize, Stop) {
    let up = b.upload(g);
    let mut labels: Vec<u32> = (0..g.n() as u32).collect();
    let cap = if o.n_iterations < 0 { MAX_ITERATIONS } else { o.n_iterations };
    let watch_gain = o.n_iterations < 0 && o.min_gain > 0.0;
    let mut q = f64::NEG_INFINITY;
    let mut stop = Stop::IterationCap;
    for it in 0..cap {
        let t = std::time::Instant::now();
        let next = canonical(&iteration(b, &up, &labels, o, it as u64, hashes.as_deref_mut()));
        let unchanged = next == labels;
        let mut done = unchanged;
        if watch_gain || trace() {
            let q_next = modularity_csr(g, &next, o.resolution);
            if trace() {
                let changed = next.iter().zip(&labels).filter(|(a, b)| a != b).count();
                eprintln!("ITER {it} time {:.4} changed {changed} Q {q_next:.7}", t.elapsed().as_secs_f64());
            }
            done |= watch_gain && q_next - q < o.min_gain;
            q = q_next;
        }
        labels = next;
        stop = if unchanged { Stop::Converged } else if done { Stop::MinGain } else { Stop::IterationCap };
        if done && o.n_iterations < 0 {
            return (labels, it as usize + 1, stop);
        }
    }
    assert!(o.n_iterations >= 0, "Leiden did not converge in {MAX_ITERATIONS} iterations");
    (labels, cap as usize, stop)
}

/// One Leiden iteration from `start`: move, refine, aggregate, until no level aggregates.
/// Level 0 runs on the input graph (f32 or f64 weights), every coarse level on f64.
fn iteration<B: Backend, W: Weight>(b: &B, g: &B::Graph<'_, W>, start: &[u32], o: &Options, it: u64,
                                    mut hashes: Option<&mut Vec<u64>>) -> Vec<u32> {
    let mut node_of: Vec<u32> = (0..b.n(g) as u32).collect(); // coarse node of each vertex
    let mut comm = start.to_vec();
    let mut next = level(b, g, &mut comm, o, it, 0, hashes.as_deref_mut());
    for lvl in 1..MAX_LEVELS {
        let Some((agg, coarse)) = next else { break };
        for x in node_of.iter_mut() {
            *x = agg[*x as usize];
        }
        next = level(b, &coarse, &mut comm, o, it, lvl, hashes.as_deref_mut());
    }
    node_of.iter().map(|&x| comm[x as usize]).collect()
}

/// One level: move nodes, then refine and aggregate. `comm` becomes the partition of the next
/// level's nodes. None when every community is one node (nothing to aggregate); otherwise the
/// map from this level's nodes to the next level's, and the next level's graph.
fn level<B: Backend, W: Weight>(b: &B, cur: &B::Graph<'_, W>, comm: &mut Vec<u32>, o: &Options, it: u64, level: u64,
                                hashes: Option<&mut Vec<u64>>) -> Option<(Vec<u32>, B::Graph<'static, f64>)> {
    let trace = trace();
    let step = it << 32 | level;
    if let Some(h) = hashes {
        h.push(b.hash(cur));
    }
    let t0 = std::time::Instant::now();
    let tol = o.tolerance / 10f64.powi(level as i32);
    let col = b.colour(cur, o.seed, step);
    let (colours, rounds) = b.move_nodes(cur, comm, o.resolution, tol, &col, o.apply);
    let (dcomm, n_comm) = refine::dense(comm);
    let move_s = t0.elapsed().as_secs_f64();
    if trace {
        eprintln!("it {it} level {level} n {} m {} hash {:016x} colours {colours} rounds {rounds} communities {n_comm} move_s {move_s:.4}",
                  b.n(cur), b.edges(cur), b.hash(cur));
    }
    let t1 = std::time::Instant::now(); // after the trace's hash, so refine_s doesn't include it
    if n_comm == b.n(cur) {
        return None; // every community is one node: nothing to aggregate
    }
    let (mut agg, mut n_agg) = b.refine(cur, &dcomm, o.resolution, o.theta, o.seed, step, o.apply, &col);
    let t2 = std::time::Instant::now();
    if n_agg == b.n(cur) {
        (agg, n_agg) = (dcomm.clone(), n_comm); // refinement merged nothing: aggregate the moved partition (igraph)
    }
    let mut next_comm = vec![0u32; n_agg];
    for v in 0..b.n(cur) {
        next_comm[agg[v] as usize] = dcomm[v]; // each refined cluster lies in one community
    }
    let next = b.aggregate(cur, &agg, n_agg, o.apply);
    if trace {
        eprintln!("it {it} level {level}   refined {n_agg} refine_s {:.4} aggregate_s {:.4}",
                  (t2 - t1).as_secs_f64(), t2.elapsed().as_secs_f64());
    }
    *comm = next_comm;
    Some((agg, next))
}

/// Labels ordered by community size (largest first), then by smallest member id.
fn canonical(labels: &[u32]) -> Vec<u32> {
    let (d, k) = refine::dense(labels); // first occurrence = smallest member
    let mut size = vec![0u32; k];
    for &l in &d {
        size[l as usize] += 1;
    }
    let mut ids: Vec<u32> = (0..k as u32).collect();
    ids.sort_by_key(|&c| (std::cmp::Reverse(size[c as usize]), c));
    let mut rank = vec![0u32; k];
    for (r, &c) in ids.iter().enumerate() {
        rank[c as usize] = r as u32;
    }
    d.iter().map(|&l| rank[l as usize]).collect()
}

/// Modularity of `labels` at resolution γ, summed in a fixed order.
pub fn modularity(g: &Graph, labels: &[u32], resolution: f64) -> f64 {
    match &g.0 {
        Repr::F32(c) => modularity_csr(c, labels, resolution),
        Repr::F64(c) => modularity_csr(c, labels, resolution),
    }
}

fn modularity_csr<W: Weight>(g: &Csr<W>, labels: &[u32], resolution: f64) -> f64 {
    if g.total <= 0.0 {
        return 0.0;
    }
    let (inside, tot) = community_sums(g, labels);
    (0..inside.len()).map(|c| inside[c] - resolution * tot[c] * tot[c] / g.total).sum::<f64>() / g.total
}

/// Modularity of `labels` as a line in γ: `(a, b)` with Q(γ) = a − γ·b, where a is the fraction of the
/// edge weight inside communities and b = Σ_c (K_c / 2m)². Equals `modularity` up to rounding.
pub fn modularity_line(g: &Graph, labels: &[u32]) -> (f64, f64) {
    fn line<W: Weight>(g: &Csr<W>, labels: &[u32]) -> (f64, f64) {
        if g.total <= 0.0 {
            return (0.0, 0.0);
        }
        let (inside, tot) = community_sums(g, labels);
        (inside.iter().sum::<f64>() / g.total, tot.iter().map(|t| t * t).sum::<f64>() / (g.total * g.total))
    }
    match &g.0 {
        Repr::F32(c) => line(c, labels),
        Repr::F64(c) => line(c, labels),
    }
}

/// Per community, numbered by first appearance (so the same partition always sums in the same
/// order): the edge weight inside it (both directions) and its total strength.
fn community_sums<W: Weight>(g: &Csr<W>, labels: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let (d, k) = refine::dense(labels);
    let (mut inside, mut tot) = (vec![0.0f64; k], vec![0.0f64; k]);
    for v in 0..g.n() {
        let c = d[v] as usize;
        tot[c] += g.strength[v];
        let (t, w) = g.row(v);
        for (&u, &w) in t.iter().zip(w) {
            if d[u as usize] as usize == c {
                inside[c] += Into::<f64>::into(w);
            }
        }
    }
    (inside, tot)
}
