//! Performance non-regression: `cargo bench -- --save-baseline main`, then after a change
//! `cargo bench -- --baseline main`. Uses the hao2021 graphs in bench/graphs (git-ignored; skipped
//! if absent). Hold the CPU lock: `flock bench/.cpu.lock cargo bench ...`. With a GPU feature
//! (`--features gpu-cuda`, `gpu-metal`, `gpu-wgpu`) also the GPU path and its CPU reference
//! (`Apply::OrderFree`), in-process, so kernel compilation falls in criterion's warm-up.

use criterion::{criterion_group, criterion_main, Criterion};
use clust_libreville_rs::{leiden, Graph, Options};

fn load(name: &str) -> Option<Graph> {
    let path = format!("{}/bench/graphs/{name}.npz", env!("CARGO_MANIFEST_DIR"));
    let mut z = npyz::npz::NpzArchive::open(&path).ok()?;
    let mut ints = |k: &str| -> Vec<i64> {
        let p = z.by_name(k).unwrap().unwrap();
        if p.dtype().descr() == "'<i8'" { p.into_vec::<i64>().unwrap() } else { p.into_vec::<i32>().unwrap().into_iter().map(i64::from).collect() }
    };
    let indptr: Vec<usize> = ints("indptr").into_iter().map(|x| x as usize).collect();
    let indices: Vec<u32> = ints("indices").into_iter().map(|x| x as u32).collect();
    let p = z.by_name("data").unwrap().unwrap();
    let data: Vec<f64> = if p.dtype().descr() == "'<f4'" { p.into_vec::<f32>().unwrap().into_iter().map(f64::from).collect() } else { p.into_vec::<f64>().unwrap() };
    Some(Graph::from_csr(&indptr, &indices, &data).unwrap())
}

fn bench(c: &mut Criterion) {
    for name in ["hao2021_10k", "hao2021_100k"] {
        let Some(g) = load(name) else { continue };
        let mut grp = c.benchmark_group(name);
        grp.sample_size(10);
        for threads in [1, 8] {
            for (mode, n_iterations) in [("1iter", 1), ("converged", -1)] {
                let o = Options { seed: 100, n_iterations, threads: Some(threads), ..Default::default() };
                grp.bench_function(format!("{mode}/{threads}t"), |b| b.iter(|| leiden(&g, &o)));
            }
        }
        #[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
        for (mode, n_iterations) in [("1iter", 1), ("converged", -1)] {
            let o = Options { seed: 100, n_iterations, apply: clust_libreville_rs::Apply::OrderFree, ..Default::default() };
            grp.bench_function(format!("{mode}/orderfree"), |b| b.iter(|| leiden(&g, &o)));
            grp.bench_function(format!("{mode}/gpu"), |b| b.iter(|| clust_libreville_rs::leiden_gpu(&g, &o)));
        }
        grp.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
