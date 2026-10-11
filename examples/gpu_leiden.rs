//! cargo run --release --features gpu-cuda --example gpu_leiden -- GRAPH.npz [--niter N] [--threads T]
//! leiden_gpu (level-0 local moving on the GPU, OrderFree) against leiden with OrderFree (its CPU
//! reference: same labels?) and against the CPU default (Sequential), with times.
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal")))]
fn main() {
    eprintln!("build with --features gpu-cuda, gpu-metal or gpu-wgpu");
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let mut z = npyz::npz::NpzArchive::open(&args[0]).unwrap();
    let mut ints = |k: &str| -> Vec<i64> {
        let p = z.by_name(k).unwrap().unwrap();
        if p.dtype().descr() == "'<i8'" { p.into_vec::<i64>().unwrap() } else { p.into_vec::<i32>().unwrap().into_iter().map(i64::from).collect() }
    };
    let indptr: Vec<usize> = ints("indptr").into_iter().map(|x| x as usize).collect();
    let indices: Vec<u32> = ints("indices").into_iter().map(|x| x as u32).collect();
    let p = z.by_name("data").unwrap().unwrap();
    let data: Vec<f64> = if p.dtype().descr() == "'<f4'" { p.into_vec::<f32>().unwrap().into_iter().map(f64::from).collect() } else { p.into_vec::<f64>().unwrap() };
    let g = clust_libreville_rs::Graph::from_csr(&indptr, &indices, &data).unwrap();
    let o = clust_libreville_rs::Options {
        seed: 100,
        n_iterations: flag("--niter").map_or(1, |s| s.parse().unwrap()),
        threads: flag("--threads").map(|s| s.parse().unwrap()),
        ..Default::default()
    };
    // best of 3; the third element: all three calls gave the same labels
    let best = |f: &dyn Fn() -> Vec<u32>| {
        let mut r = (Vec::new(), f64::INFINITY, true);
        for i in 0..3 {
            let t = std::time::Instant::now();
            let l = f();
            let same = i == 0 || l == r.0;
            r = (l, r.1.min(t.elapsed().as_secs_f64()), r.2 && same);
        }
        r
    };
    if std::env::var_os("ONCE").is_some() {
        let _ = clust_libreville_rs::leiden_gpu(&g, &o); // warm-up (JIT)
        eprintln!("--- traced GPU run");
        let _ = clust_libreville_rs::leiden_gpu(&g, &o);
        return;
    }
    let cpu_ref = clust_libreville_rs::Options { apply: clust_libreville_rs::Apply::OrderFree, ..o.clone() }; // the GPU's rule
    let (cpu, t_cpu, _) = best(&|| clust_libreville_rs::leiden(&g, &cpu_ref));
    let t_default = best(&|| clust_libreville_rs::leiden(&g, &o)).1;
    let (gpu, t_gpu, repeat) = best(&|| clust_libreville_rs::leiden_gpu(&g, &o));
    println!("n {} niter {} threads {:?}: CPU default {t_default:.3} s | CPU OrderFree {t_cpu:.3} s | GPU {t_gpu:.3} s (best of 3; upload, not JIT) | GPU == CPU OrderFree: {} | GPU repeatable: {repeat} | Q GPU {:.6} CPU OrderFree {:.6}",
             g.n(), o.n_iterations, o.threads, cpu == gpu, clust_libreville_rs::modularity(&g, &gpu, 1.0), clust_libreville_rs::modularity(&g, &cpu, 1.0));
}
