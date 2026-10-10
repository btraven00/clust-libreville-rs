//! detleiden: seeded, deterministic Leiden community detection (modularity, resolution γ).
//!
//! Same graph, same options, same seed: the same labels. Every random choice is a pure function
//! of (seed, step, vertex) and every floating-point sum runs in a fixed order, so the result
//! does not depend on the order in which work is done (phase 2 relies on it for thread-count
//! invariance). Phase 1: sequential. See PLAN.md.

mod aggregate;
mod graph;
mod moving;
mod refine;
mod rng;

pub use graph::Graph;

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
    /// Colour the input graph once per run instead of once per iteration (faster iterations;
    /// different, still deterministic, trajectories).
    pub reuse_colouring: bool,
    /// Worker threads; None: `RAYON_NUM_THREADS`, else `std::thread::available_parallelism()`
    /// (which respects cgroup quotas). The labels do not depend on it.
    pub threads: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options { resolution: 1.0, seed: 0, n_iterations: -1, theta: 0.01, tolerance: 0.0, min_gain: 0.0, reuse_colouring: false, threads: None }
    }
}

const MAX_ITERATIONS: i64 = 1000; // igraph/igraph#2934: fail rather than loop forever
const MAX_LEVELS: u64 = 1000;

/// Community labels 0.., ordered by community size (largest first), then by smallest member.
pub fn leiden(g: &Graph, o: &Options) -> Vec<u32> {
    run(g, o, None)
}

/// `leiden`, plus the hash of every level's graph (structure and exact weight bits), in order:
/// for checking that nothing, not even a last bit, depends on the thread count.
#[doc(hidden)]
pub fn leiden_traced(g: &Graph, o: &Options) -> (Vec<u32>, Vec<u64>) {
    let mut h = Vec::new();
    let l = run(g, o, Some(&mut h));
    (l, h)
}

fn run(g: &Graph, o: &Options, hashes: Option<&mut Vec<u64>>) -> Vec<u32> {
    let threads = o.threads.unwrap_or_else(|| {
        std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&t| t > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    });
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool");
    pool.install(|| iterate(g, o, hashes))
}

fn iterate(g: &Graph, o: &Options, mut hashes: Option<&mut Vec<u64>>) -> Vec<u32> {
    let mut labels: Vec<u32> = (0..g.n() as u32).collect();
    let cap = if o.n_iterations < 0 { MAX_ITERATIONS } else { o.n_iterations };
    let colours0 = o.reuse_colouring.then(|| moving::colouring(g, o.seed, 0));
    let watch_gain = o.n_iterations < 0 && o.min_gain > 0.0;
    let mut q = f64::NEG_INFINITY;
    for it in 0..cap {
        let t = std::time::Instant::now();
        let next = canonical(&iteration(g, &labels, o, it as u64, colours0.as_deref(), hashes.as_deref_mut()));
        let mut done = next == labels;
        if watch_gain || std::env::var_os("DETLEIDEN_TRACE").is_some() {
            let q_next = modularity(g, &next, o.resolution);
            if std::env::var_os("DETLEIDEN_TRACE").is_some() {
                let changed = next.iter().zip(&labels).filter(|(a, b)| a != b).count();
                eprintln!("ITER {it} time {:.4} changed {changed} Q {q_next:.7}", t.elapsed().as_secs_f64());
            }
            done |= watch_gain && q_next - q < o.min_gain;
            q = q_next;
        }
        labels = next;
        if done && o.n_iterations < 0 {
            return labels;
        }
    }
    assert!(o.n_iterations >= 0, "Leiden did not converge in {MAX_ITERATIONS} iterations");
    labels
}

/// One Leiden iteration from `start`: move, refine, aggregate, until no level aggregates.
fn iteration(g: &Graph, start: &[u32], o: &Options, it: u64, colours0: Option<&[Vec<u32>]>,
             mut hashes: Option<&mut Vec<u64>>) -> Vec<u32> {
    let trace = std::env::var_os("DETLEIDEN_TRACE").is_some();
    let mut node_of: Vec<u32> = (0..g.n() as u32).collect(); // coarse node of each vertex
    let mut comm = start.to_vec();
    let mut coarse: Option<Graph> = None;
    for level in 0..MAX_LEVELS {
        let cur = coarse.as_ref().unwrap_or(g);
        let step = it << 32 | level;
        if let Some(h) = hashes.as_deref_mut() {
            h.push(cur.hash());
        }
        let t0 = std::time::Instant::now();
        let tol = o.tolerance / 10f64.powi(level as i32);
        let fresh;
        let groups = match colours0 {
            Some(c) if level == 0 => c,
            _ => {
                fresh = moving::colouring(cur, o.seed, step);
                &fresh
            }
        };
        let (colours, rounds) = moving::move_nodes(cur, &mut comm, o.resolution, tol, groups);
        let (dcomm, n_comm) = refine::dense(&comm);
        let t1 = std::time::Instant::now();
        if trace {
            eprintln!("it {it} level {level} n {} m {} hash {:016x} colours {colours} rounds {rounds} communities {n_comm} move_s {:.4}",
                      cur.n(), cur.targets.len(), cur.hash(), (t1 - t0).as_secs_f64());
        }
        if n_comm == cur.n() {
            break; // every community is one node: nothing to aggregate
        }
        let (mut agg, mut n_agg) = refine::refine(cur, &dcomm, o.resolution, o.theta, o.seed, step);
        let t2 = std::time::Instant::now();
        if n_agg == cur.n() {
            (agg, n_agg) = (dcomm.clone(), n_comm); // refinement merged nothing: aggregate the moved partition (igraph)
        }
        let mut next_comm = vec![0u32; n_agg];
        for v in 0..cur.n() {
            next_comm[agg[v] as usize] = dcomm[v]; // each refined cluster lies in one community
        }
        let next = aggregate::aggregate(cur, &agg, n_agg);
        if trace {
            eprintln!("it {it} level {level}   refined {n_agg} refine_s {:.4} aggregate_s {:.4}",
                      (t2 - t1).as_secs_f64(), t2.elapsed().as_secs_f64());
        }
        for x in node_of.iter_mut() {
            *x = agg[*x as usize];
        }
        comm = next_comm;
        coarse = Some(next);
    }
    node_of.iter().map(|&x| comm[x as usize]).collect()
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
    if g.total <= 0.0 {
        return 0.0;
    }
    let (d, k) = refine::dense(labels);
    let (mut inside, mut tot) = (vec![0.0f64; k], vec![0.0f64; k]);
    for v in 0..g.n() {
        let c = d[v] as usize;
        tot[c] += g.strength[v];
        let (t, w) = g.row(v);
        for (&u, &w) in t.iter().zip(w) {
            if d[u as usize] as usize == c {
                inside[c] += w;
            }
        }
    }
    (0..k).map(|c| inside[c] - resolution * tot[c] * tot[c] / g.total).sum::<f64>() / g.total
}
