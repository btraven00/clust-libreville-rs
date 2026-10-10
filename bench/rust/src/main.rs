//! ldbench: run leiden-rs (stock / trace / seqagg) or single-clustering on one graph.
//!
//!   ldbench run  --graph G --impl I [--threads T] [--seed S] [--niter N] [--out PREFIX]
//!                [--seqmove] [--aggseq] [--louvain] [--move-thr N] [--agg-thr N]
//!   ldbench prop [--graphs K] [--threads T]      # H3: disconnected communities on random graphs
//!
//! G: a .npz (indptr, indices, data: full symmetric CSR), a whitespace edge list (u v [w]),
//!    or a generator: er:N:AVGDEG:SEED | sbm:K:SIZE:DIN:DOUT:SEED | knn2d:N:K:SEED | cliques
//! I: stock | trace | seqagg | sc   (sc = single-clustering, always 1 thread)
//! Writes PREFIX.npy (int64 labels, renumbered by first occurrence) and PREFIX.json.
//! Parameter mapping to igraph community_leiden(objective="modularity", resolution=1,
//! beta=0.01, n_iterations=N) is in README.md.
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

struct Csr {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    data: Vec<f64>,
}

impl Csr {
    /// From an undirected edge list (each edge once); mirrors it.
    fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Csr {
        let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for &(u, v, w) in edges {
            adj[u].push((v as u32, w));
            if u != v {
                adj[v].push((u as u32, w));
            }
        }
        let mut c = Csr { n, indptr: vec![0], indices: vec![], data: vec![] };
        for row in &mut adj {
            row.sort_by_key(|e| e.0);
            for &(v, w) in row.iter() {
                c.indices.push(v);
                c.data.push(w);
            }
            c.indptr.push(c.indices.len());
        }
        c
    }
    fn row(&self, u: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        (self.indptr[u]..self.indptr[u + 1]).map(|i| (self.indices[i] as usize, self.data[i]))
    }
}

fn read_npz(path: &str) -> Csr {
    let mut z = npyz::npz::NpzArchive::open(path).expect("open npz");
    let mut ints = |name: &str| -> Vec<i64> {
        let p = z.by_name(name).expect("read npz").unwrap_or_else(|| panic!("npz has no '{name}'"));
        match p.dtype().descr().as_str() {
            "'<i8'" => p.into_vec::<i64>().unwrap(),
            _ => p.into_vec::<i32>().unwrap().into_iter().map(i64::from).collect(),
        }
    };
    let indptr: Vec<usize> = ints("indptr").into_iter().map(|x| x as usize).collect();
    let indices: Vec<u32> = ints("indices").into_iter().map(|x| x as u32).collect();
    let p = z.by_name("data").expect("read npz").expect("npz has no 'data'");
    let data: Vec<f64> = match p.dtype().descr().as_str() {
        "'<f4'" => p.into_vec::<f32>().unwrap().into_iter().map(f64::from).collect(),
        _ => p.into_vec::<f64>().unwrap(),
    };
    Csr { n: indptr.len() - 1, indptr, indices, data }
}

fn load_graph(spec: &str) -> Csr {
    let parts: Vec<&str> = spec.split(':').collect();
    let num = |i: usize| parts[i].parse::<f64>().unwrap();
    match parts[0] {
        "er" => {
            let (n, deg, seed) = (num(1) as usize, num(2), num(3) as u64);
            let mut rng = StdRng::seed_from_u64(seed);
            let m = (n as f64 * deg / 2.0) as usize;
            let mut e: Vec<(usize, usize, f64)> = (0..m)
                .map(|_| (rng.random_range(0..n), rng.random_range(0..n), rng.random_range(0.1..1.0)))
                .filter(|&(u, v, _)| u != v)
                .map(|(u, v, w)| (u.min(v), u.max(v), w))
                .collect();
            e.sort_by_key(|x| (x.0, x.1));
            e.dedup_by_key(|x| (x.0, x.1));
            Csr::from_edges(n, &e)
        }
        "sbm" => {
            let (k, size, din, dout, seed) = (num(1) as usize, num(2) as usize, num(3), num(4), num(5) as u64);
            let n = k * size;
            let mut rng = StdRng::seed_from_u64(seed);
            let (pin, pout) = (din / size as f64, dout / (n - size) as f64);
            let mut e = vec![];
            for u in 0..n {
                for v in u + 1..n {
                    let p = if u / size == v / size { pin } else { pout };
                    if rng.random::<f64>() < p {
                        e.push((u, v, rng.random_range(0.1..1.0)));
                    }
                }
            }
            Csr::from_edges(n, &e)
        }
        "knn2d" => {
            // ponytail: O(n^2) brute-force kNN, fine for property-test sizes (n <= ~5000).
            let (n, k, seed) = (num(1) as usize, num(2) as usize, num(3) as u64);
            let mut rng = StdRng::seed_from_u64(seed);
            let pts: Vec<(f64, f64)> = (0..n).map(|_| (rng.random(), rng.random())).collect();
            let mut e = vec![];
            for u in 0..n {
                let mut d: Vec<(f64, usize)> = (0..n)
                    .filter(|&v| v != u)
                    .map(|v| ((pts[u].0 - pts[v].0).powi(2) + (pts[u].1 - pts[v].1).powi(2), v))
                    .collect();
                d.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for &(dist, v) in &d[..k] {
                    e.push((u.min(v), u.max(v), (-dist * n as f64).exp()));
                }
            }
            e.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            e.dedup_by_key(|x| (x.0, x.1));
            Csr::from_edges(n, &e)
        }
        "cliques" => {
            let mut e = vec![];
            for base in [0, 5] {
                for i in 0..5 {
                    for j in i + 1..5 {
                        e.push((base + i, base + j, 1.0));
                    }
                }
            }
            e.push((0, 5, 1.0));
            Csr::from_edges(10, &e)
        }
        _ if spec.ends_with(".npz") => read_npz(spec),
        _ => {
            let txt = std::fs::read_to_string(spec).expect("read edge list");
            let e: Vec<(usize, usize, f64)> = txt
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
                .map(|l| {
                    let f: Vec<&str> = l.split_whitespace().collect();
                    let (u, v): (usize, usize) = (f[0].parse().unwrap(), f[1].parse().unwrap());
                    (u.min(v), u.max(v), f.get(2).map_or(1.0, |w| w.parse().unwrap()))
                })
                .collect();
            let n = e.iter().map(|x| x.1 + 1).max().unwrap_or(0);
            Csr::from_edges(n, &e)
        }
    }
}

#[derive(Clone, Default)]
struct Opts {
    imp: String,
    seed: u64,
    niter: i64,
    seqmove: bool,
    aggseq: bool,
    louvain: bool,
    move_thr: Option<usize>,
    agg_thr: Option<usize>,
}

struct Res {
    labels: Vec<usize>,
    quality: f64,
    passes: usize,
    build_s: f64,
    run_s: f64,
}

/// One leiden-rs implementation (the three crates are distinct types, hence a macro).
/// niter = 1: one `run()` (all levels). niter = k: up to k passes, each warm-started from the
/// previous labels with seed+pass. niter = -1: passes until the labels stop changing (cap 1000).
macro_rules! run_lrs {
    ($k:ident, $g:expr, $o:expr) => {{
        let (g, o): (&Csr, &Opts) = ($g, $o);
        let t = Instant::now();
        let mut b = $k::GraphDataBuilder::new(g.n);
        for u in 0..g.n {
            for (v, w) in g.row(u) {
                if v >= u {
                    b.add_edge(u, v, w).unwrap();
                }
            }
        }
        let data = b.build().unwrap();
        let build_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let max_pass = if o.niter < 0 { 1000 } else { o.niter as usize };
        let mut prev: Option<$k::Partition> = None;
        let mut quality = 0.0;
        let mut passes = 0;
        for pass in 0..max_pass {
            let mut cfg = $k::LeidenConfig::builder()
                .resolution(1.0)
                .quality($k::QualityType::Modularity)
                .seed(o.seed.wrapping_add(pass as u64))
                .skip_refinement(o.louvain)
                .build();
            if o.seqmove {
                cfg.parallel_local_moving_threshold = Some(usize::MAX);
            } else if let Some(x) = o.move_thr {
                cfg.parallel_local_moving_threshold = Some(x);
            }
            if o.aggseq {
                cfg.parallel_aggregation_threshold = Some(usize::MAX);
            } else if let Some(x) = o.agg_thr {
                cfg.parallel_aggregation_threshold = Some(x);
            }
            let l = $k::Leiden::new(cfg);
            let out = match &prev {
                None => l.run(&data).unwrap(),
                Some(p) => l.run_with_initial_partition(&data, p.clone()).unwrap(),
            };
            passes += 1;
            quality = out.quality;
            let same = prev.as_ref().is_some_and(|p| p.as_slice() == out.partition.as_slice());
            prev = Some(out.partition);
            if same {
                break;
            }
        }
        let run_s = t.elapsed().as_secs_f64();
        Res { labels: prev.unwrap().as_slice().to_vec(), quality, passes, build_s, run_s }
    }};
}

fn run_sc(g: &Csr, o: &Opts) -> Res {
    use single_clustering::community_search::leiden::{leiden, modularity, LeidenConfig, ObjectiveKind};
    let t = Instant::now();
    let net = single_clustering::network::CSRNetwork::from_csr_parts(
        g.indptr.clone(),
        g.indices.clone(),
        g.data.iter().map(|&w| w as f32).collect(),
        None,
    )
    .unwrap();
    let build_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let cfg = LeidenConfig {
        objective: ObjectiveKind::Rb { resolution: 1.0 },
        seed: Some(o.seed),
        // igraph n_iterations=-1 ~ passes until one finds nothing (patience 1)
        n_iterations: if o.niter < 0 { 1000 } else { o.niter as usize },
        patience: 1,
        refine: !o.louvain,
        refine_randomness: 0.01,
        ..Default::default()
    };
    let c = leiden(&net, &cfg).unwrap();
    let run_s = t.elapsed().as_secs_f64();
    let quality = modularity(&net, c.labels(), 1.0);
    Res { labels: c.labels().to_vec(), quality, passes: 0, build_s, run_s }
}

fn run_impl(g: &Csr, o: &Opts) -> Res {
    match o.imp.as_str() {
        "stock" => run_lrs!(leiden_stock, g, o),
        "trace" => run_lrs!(leiden_trace, g, o),
        "seqagg" => run_lrs!(leiden_seqagg, g, o),
        "sc" => run_sc(g, o),
        x => panic!("unknown --impl {x}"),
    }
}

/// Renumber labels by first occurrence, so equal partitions give equal arrays.
fn canon(labels: &[usize]) -> Vec<i64> {
    let mut map = std::collections::HashMap::new();
    labels
        .iter()
        .map(|l| {
            let k = map.len() as i64;
            *map.entry(*l).or_insert(k)
        })
        .collect()
}

/// Number of communities whose induced subgraph is disconnected (Leiden guarantees 0).
fn disconnected(g: &Csr, labels: &[i64]) -> usize {
    let k = labels.iter().max().map_or(0, |&m| m as usize + 1);
    let mut seen = vec![false; g.n];
    let mut parts = vec![0usize; k];
    for s in 0..g.n {
        if seen[s] {
            continue;
        }
        parts[labels[s] as usize] += 1;
        seen[s] = true;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for (v, _) in g.row(u) {
                if !seen[v] && labels[v] == labels[u] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
    }
    parts.iter().filter(|&&p| p > 1).count()
}

fn hash_of(x: &[i64]) -> u64 {
    let mut h = DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

fn pool(threads: Option<usize>) -> rayon::ThreadPool {
    let t = threads
        .or_else(|| std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(1);
    rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let has = |k: &str| args.iter().any(|a| a == k);
    let threads = get("--threads").map(|s| s.parse().unwrap());
    let pool = pool(threads);
    let nt = pool.current_num_threads();

    if args.get(1).map(String::as_str) == Some("prop") {
        prop(get("--graphs").map_or(50, |s| s.parse().unwrap()), &pool);
        return;
    }

    let gspec = get("--graph").expect("--graph");
    let o = Opts {
        imp: get("--impl").unwrap_or("stock".into()),
        seed: get("--seed").map_or(0, |s| s.parse().unwrap()),
        niter: get("--niter").map_or(-1, |s| s.parse().unwrap()),
        seqmove: has("--seqmove"),
        aggseq: has("--aggseq"),
        louvain: has("--louvain"),
        move_thr: get("--move-thr").map(|s| s.parse().unwrap()),
        agg_thr: get("--agg-thr").map(|s| s.parse().unwrap()),
    };
    let t = Instant::now();
    let g = load_graph(&gspec);
    let load_s = t.elapsed().as_secs_f64();
    let r = if o.imp == "sc" { run_impl(&g, &o) } else { pool.install(|| run_impl(&g, &o)) };
    let labels = canon(&r.labels);
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    let threads_used = if o.imp == "sc" { 1 } else { nt };
    let json = format!(
        "{{\"graph\":\"{gspec}\",\"impl\":\"{}\",\"seqmove\":{},\"aggseq\":{},\"louvain\":{},\"threads\":{threads_used},\"seed\":{},\"niter\":{},\"n\":{},\"nnz\":{},\"passes\":{},\"load_s\":{load_s:.4},\"build_s\":{:.4},\"time_s\":{:.4},\"n_clusters\":{k},\"quality_impl\":{:.12},\"disconnected\":{},\"labels_hash\":\"{:016x}\"}}",
        o.imp, o.seqmove, o.aggseq, o.louvain, o.seed, o.niter, g.n, g.indices.len(), r.passes,
        r.build_s, r.run_s, r.quality, disconnected(&g, &labels), hash_of(&labels)
    );
    println!("{json}");
    if let Some(p) = get("--out") {
        if let Some(d) = std::path::Path::new(&p).parent() {
            std::fs::create_dir_all(d).unwrap();
        }
        npyz::to_file_1d(format!("{p}.npy"), labels).unwrap();
        std::fs::write(format!("{p}.json"), json + "\n").unwrap();
    }
}

/// H3 property test: random graphs x seeds; count disconnected final communities per impl.
/// Thresholds forced to 0 so leiden-rs takes its parallel paths even on small graphs.
fn prop(ngraphs: usize, pool: &rayon::ThreadPool) {
    let mut specs = vec![];
    for i in 0..ngraphs as u64 {
        specs.push(format!("er:{}:{}:{i}", 200 + 37 * (i % 20), 3 + i % 6));
        specs.push(format!("sbm:{}:{}:6:{}:{i}", 4 + i % 5, 30 + 5 * (i % 7), 1 + i % 4));
        specs.push(format!("knn2d:{}:{}:{i}", 300 + 50 * (i % 10), 5 + i % 10));
    }
    for imp in ["stock", "seqagg", "sc"] {
        let (mut bad_runs, mut bad_comms, mut comms, mut runs) = (0, 0, 0, 0);
        for s in &specs {
            let g = load_graph(s);
            for seed in 0..5 {
                let o = Opts { imp: imp.into(), seed, niter: -1, move_thr: Some(0), agg_thr: Some(0), ..Default::default() };
                let r = pool.install(|| run_impl(&g, &o));
                let l = canon(&r.labels);
                let d = disconnected(&g, &l);
                runs += 1;
                comms += l.iter().max().map_or(0, |&m| m as usize + 1);
                bad_comms += d;
                if d > 0 {
                    bad_runs += 1;
                    if bad_runs <= 5 {
                        println!("  {imp} {s} seed={seed}: {d} disconnected communities");
                    }
                }
            }
        }
        println!("PROP impl={imp} runs={runs} graphs={} runs_with_disconnected={bad_runs} disconnected_comms={bad_comms} total_comms={comms}", specs.len());
    }
}
