//! cargo run --release --example run -- GRAPH.npz [--seed S] [--theta T] [--niter N] [--threads T] [--tol E] [--min-gain G] [--repeat R] [--out LABELS.npy]
//! GRAPH.npz: full symmetric CSR (indptr, indices, data), as bench/py/export_graph.py writes it.
//! Prints one line: n, time (leiden() alone, the "algo only" time), build_s (Graph::from_csr), clusters, modularity. LABELS.npy: int64, graph row order.

use detleiden::{leiden, modularity, Graph, Options};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let mut o = Options::default();
    if let Some(s) = flag("--seed") { o.seed = s.parse().unwrap(); }
    if let Some(s) = flag("--theta") { o.theta = s.parse().unwrap(); }
    if let Some(s) = flag("--niter") { o.n_iterations = s.parse().unwrap(); }
    if let Some(s) = flag("--threads") { o.threads = Some(s.parse().unwrap()); }
    if let Some(s) = flag("--tol") { o.tolerance = s.parse().unwrap(); }
    if let Some(s) = flag("--min-gain") { o.min_gain = s.parse().unwrap(); }

    let mut z = npyz::npz::NpzArchive::open(&args[0]).expect("open npz");
    let mut ints = |name: &str| -> Vec<i64> {
        let p = z.by_name(name).unwrap().unwrap_or_else(|| panic!("no '{name}'"));
        if p.dtype().descr() == "'<i8'" { p.into_vec::<i64>().unwrap() } else { p.into_vec::<i32>().unwrap().into_iter().map(i64::from).collect() }
    };
    let indptr: Vec<usize> = ints("indptr").into_iter().map(|x| x as usize).collect();
    let indices: Vec<u32> = ints("indices").into_iter().map(|x| x as u32).collect();
    let p = z.by_name("data").unwrap().expect("no 'data'");
    let data: Vec<f64> = if p.dtype().descr() == "'<f4'" { p.into_vec::<f32>().unwrap().into_iter().map(f64::from).collect() } else { p.into_vec::<f64>().unwrap() };
    let t = Instant::now();
    let g = Graph::from_csr(&indptr, &indices, &data).expect("valid graph");
    let build_s = t.elapsed().as_secs_f64();

    // --repeat R: R calls in this process, report the fastest (as GVE's harness does: best of 3 in one process)
    let repeat: usize = flag("--repeat").map_or(1, |s| s.parse().unwrap());
    let (mut labels, mut secs) = (Vec::new(), f64::INFINITY);
    for _ in 0..repeat {
        let t = Instant::now();
        labels = leiden(&g, &o);
        secs = secs.min(t.elapsed().as_secs_f64());
    }
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    println!("n {} time_s {secs:.3} build_s {build_s:.3} k {k} Q {:.6} seed {} theta {} niter {} threads {:?} tol {} min_gain {}", g.n(), modularity(&g, &labels, o.resolution), o.seed, o.theta, o.n_iterations, o.threads, o.tolerance, o.min_gain);

    if let Some(path) = flag("--out") {
        let v: Vec<i64> = labels.iter().map(|&x| x as i64).collect();
        npyz::to_file_1d(&path, v).expect("write npy");
    }
}
