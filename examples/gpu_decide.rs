//! cargo run --release --features gpu-cuda --example gpu_decide -- GRAPH.npz
//! Debug: the GPU decide kernels (incl. hubs) against the CPU's best_move at singletons.
#![allow(unused)]

#[path = "../tests/common/mod.rs"]
mod common;
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal")))]
fn main() {
    eprintln!("build with --features gpu-cuda");
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]
fn main() {
    let path = std::env::args().nth(1).expect("GRAPH.npz");
    let g = common::load_npz(&path).expect("open npz");
    let singletons: Vec<u32> = (0..g.n() as u32).collect();
    let (n, hubs, mism) = clust_libreville_rs::gpu::check_decide(&g, &singletons, 1.0);
    println!("singletons: n {n}, hubs {hubs}, mismatches {mism}");
}
