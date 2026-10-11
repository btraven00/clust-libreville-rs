//! cargo run --release --example run -- GRAPH.npz [--seed S] [--resolution R] [--theta T] [--niter N] [--threads T] [--tol E] [--min-gain G] [--apply sequential|orderfree] [--repeat R] [--gpu] [--trusted] [--out LABELS.npy]
//! GRAPH.npz: full symmetric CSR (indptr, indices, data), as bench/py/export_graph.py writes it.
//! Prints one line: n, time_s (fastest leiden() call, the "algo only" time), median_s and times (every call, in order),
//! build_s (Graph::from_csr), clusters, modularity. LABELS.npy: int64, graph row order.

use clust_libreville_rs::{leiden, modularity, Graph, Options};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let mut o = Options::default();
    if let Some(s) = flag("--seed") { o.seed = s.parse().unwrap(); }
    if let Some(s) = flag("--resolution") { o.resolution = s.parse().unwrap(); }
    if let Some(s) = flag("--theta") { o.theta = s.parse().unwrap(); }
    if let Some(s) = flag("--niter") { o.n_iterations = s.parse().unwrap(); }
    if let Some(s) = flag("--threads") { o.threads = Some(s.parse().unwrap()); }
    if let Some(s) = flag("--tol") { o.tolerance = s.parse().unwrap(); }
    if let Some(s) = flag("--min-gain") { o.min_gain = s.parse().unwrap(); }
    if flag("--apply").as_deref() == Some("orderfree") { o.apply = clust_libreville_rs::Apply::OrderFree; }

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
    // --trusted: skip the symmetry check (inputs symmetric by construction)
    let build = if args.iter().any(|a| a == "--trusted") { Graph::from_csr_trusted } else { Graph::from_csr };
    let g = build(&indptr, &indices, &data).expect("valid graph");
    let build_s = t.elapsed().as_secs_f64();

    // --repeat R: R calls in this process; fastest, median and all of them (no call is discarded as warm-up)
    let repeat: usize = flag("--repeat").map_or(1, |s| s.parse().unwrap());
    let gpu = args.iter().any(|a| a == "--gpu"); // leiden_gpu (needs a gpu-* feature)
    let (mut labels, mut times) = (Vec::new(), Vec::new());
    for _ in 0..repeat {
        let t = Instant::now();
        labels = if gpu { gpu_leiden(&g, &o) } else { leiden(&g, &o) };
        times.push(t.elapsed().as_secs_f64());
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let (secs, median) = (sorted[0], (sorted[(repeat - 1) / 2] + sorted[repeat / 2]) / 2.0);
    let all: Vec<String> = times.iter().map(|t| format!("{t:.4}")).collect();
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    println!("n {} time_s {secs:.3} median_s {median:.3} times {} build_s {build_s:.3} k {k} Q {:.6} seed {} theta {} niter {} threads {:?} tol {} min_gain {} apply {:?}", g.n(), all.join(","), modularity(&g, &labels, o.resolution), o.seed, o.theta, o.n_iterations, o.threads, o.tolerance, o.min_gain, o.apply);

    if let Some(path) = flag("--out") {
        let v: Vec<i64> = labels.iter().map(|&x| x as i64).collect();
        npyz::to_file_1d(&path, v).expect("write npy");
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
fn gpu_leiden(g: &Graph, o: &clust_libreville_rs::Options) -> Vec<u32> {
    clust_libreville_rs::leiden_gpu(g, o)
}
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal")))]
fn gpu_leiden(_: &Graph, _: &clust_libreville_rs::Options) -> Vec<u32> {
    panic!("--gpu needs a build with --features gpu-cuda, gpu-metal or gpu-wgpu")
}
