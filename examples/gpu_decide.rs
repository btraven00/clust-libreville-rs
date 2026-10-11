//! cargo run --release --features gpu-cuda --example gpu_decide -- GRAPH.npz
//! Debug: the GPU decide kernels (incl. hubs) against the CPU's best_move at singletons.
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal")))]
fn main() {
    eprintln!("build with --features gpu-cuda");
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
fn main() {
    let path = std::env::args().nth(1).expect("GRAPH.npz");
    let mut z = npyz::npz::NpzArchive::open(&path).unwrap();
    let mut ints = |k: &str| -> Vec<i64> {
        let p = z.by_name(k).unwrap().unwrap();
        if p.dtype().descr() == "'<i8'" { p.into_vec::<i64>().unwrap() } else { p.into_vec::<i32>().unwrap().into_iter().map(i64::from).collect() }
    };
    let indptr: Vec<usize> = ints("indptr").into_iter().map(|x| x as usize).collect();
    let indices: Vec<u32> = ints("indices").into_iter().map(|x| x as u32).collect();
    let p = z.by_name("data").unwrap().unwrap();
    let data: Vec<f64> = if p.dtype().descr() == "'<f4'" { p.into_vec::<f32>().unwrap().into_iter().map(f64::from).collect() } else { p.into_vec::<f64>().unwrap() };
    let g = detleiden::Graph::from_csr(&indptr, &indices, &data).unwrap();
    let singletons: Vec<u32> = (0..g.n() as u32).collect();
    let (n, hubs, mism) = detleiden::gpu::check_decide(&g, &singletons, 1.0);
    println!("singletons: n {n}, hubs {hubs}, mismatches {mism}");
}
