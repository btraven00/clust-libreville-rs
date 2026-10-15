//! The GPU backend against its CPU reference (`leiden` with `Apply::OrderFree`), on whichever
//! cubecl runtime the build has:
//!
//!     cargo test --release --features gpu-cuda --test gpu              # or gpu-metal, gpu-wgpu
//!     cargo test --release --features gpu-metal --test gpu -- --ignored --nocapture   # + bench graphs, times
//!
//! Devices with f64 (CUDA): labels and every level's graph hash equal the CPU's bit for bit, so
//! a failure names the first level that differs. With `gpu-f32` (Metal, WGSL): two runs must give
//! the same labels and hashes (repeatable on the device), and the partition must be close to the
//! CPU's (modularity within 1e-3, NVI below 0.1; both printed). Synthetic graphs force every
//! device path: level 0 on the device, coarse levels on the device, hub rows (the shared-memory
//! sort kernels), a row too long for them (host fallback). The ignored test runs the hao2021 and
//! syn_1M graphs from bench/graphs (`DETLEIDEN_GRAPHS`), skipping absent ones, and prints best-of-3
//! times of the CPU default, the CPU reference and the GPU (kernel compilation falls in run 1).
#![cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal"))]

mod common;

use clust_libreville_rs::{leiden, leiden_gpu, leiden_gpu_traced, leiden_traced, modularity, nvi, Apply, Graph, Options};

const F32: bool = cfg!(feature = "gpu-f32");
/// One GPU run at a time: concurrent `leiden_gpu` calls share the default client, and a stream
/// capture started by one is invalidated by the other's launches (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED).
static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// The crate's own thresholds (gpu.rs GPU_MIN_N / GPU_COARSE_MIN_N): what `leiden_gpu` uses.
const DEFAULT: (usize, usize) = (50_000, 500_000);
/// Everything on the device, whatever the level's size.
const ALL_ON_DEVICE: (usize, usize) = (0, 0);

/// The GPU run of `o` on `g` (device thresholds `th`) against the CPU reference; returns the GPU labels.
fn check(name: &str, g: &Graph, o: &Options, th: (usize, usize)) -> Vec<u32> {
    let _one = DEVICE.lock().unwrap_or_else(|e| e.into_inner());
    let cpu = leiden_traced(g, &Options { apply: Apply::OrderFree, ..o.clone() });
    let gpu = leiden_gpu_traced(g, o, th.0, th.1);
    let again = leiden_gpu_traced(g, o, th.0, th.1);
    assert_eq!(gpu.1, again.1, "{name}: level hashes differ between two GPU runs (not repeatable)");
    assert_eq!(gpu.0, again.0, "{name}: labels differ between two GPU runs (not repeatable)");
    let (q_cpu, q_gpu) = (modularity(g, &cpu.0, o.resolution), modularity(g, &gpu.0, o.resolution));
    if F32 {
        let d = nvi(&cpu.0, &gpu.0);
        eprintln!("{name}: niter {} f32 kernels: Q GPU {q_gpu:.6} CPU {q_cpu:.6}, NVI {d:.4}, labels equal {}", o.n_iterations, cpu.0 == gpu.0);
        assert!((q_cpu - q_gpu).abs() < 1e-3, "{name}: Q GPU {q_gpu} vs CPU {q_cpu}");
        assert!(d < 0.1, "{name}: NVI to the CPU reference {d}");
    } else {
        if let Some(l) = cpu.1.iter().zip(&gpu.1).position(|(a, b)| a != b) {
            panic!("{name}: level {l} (of {}) of the first differing iteration hashes {:016x} on the CPU, {:016x} on the GPU",
                   cpu.1.len(), cpu.1[l], gpu.1[l]);
        }
        assert_eq!(cpu.1.len(), gpu.1.len(), "{name}: different number of levels");
        assert_eq!(cpu.0, gpu.0, "{name}: labels differ from the CPU reference (Q GPU {q_gpu} CPU {q_cpu})");
    }
    gpu.0
}

fn opts(n_iterations: i64) -> Options {
    Options { seed: 100, n_iterations, ..Default::default() }
}

/// 60k vertices in 40 blocks, kNN-like rows, 12 hubs of ~700 edges: level 0 on the device with the
/// crate's thresholds (coarse levels on the host), one iteration and converged; hub rows take the
/// shared-memory sort kernels of local moving and refinement.
#[test]
fn level_0_on_the_device() {
    let g = common::sparse_planted(1, 60_000, 40, 12, 0.8, 12, 700);
    check("synthetic 60k, 1 iteration", &g, &opts(1), DEFAULT);
    check("synthetic 60k, converged", &g, &opts(-1), DEFAULT);
}

/// Every level on the device, including the coarse ones (dense rows, self-loops, aggregation and
/// colouring of coarse graphs on the device), on a 30k graph that would otherwise stay on the host.
#[test]
fn coarse_levels_on_the_device() {
    let g = common::sparse_planted(2, 30_000, 25, 12, 0.8, 6, 600);
    check("synthetic 30k all on device, 1 iteration", &g, &opts(1), ALL_ON_DEVICE);
    check("synthetic 30k all on device, converged", &g, &opts(-1), ALL_ON_DEVICE);
}

/// A row longer than the hub kernels' shared memory (HUB_P = 1024 edges): local moving takes the
/// quadratic rescan, refinement of that level falls back to the host. Same labels.
#[test]
fn rows_longer_than_the_hub_kernels_fall_back() {
    let g = common::sparse_planted(3, 20_000, 20, 12, 0.8, 1, 1500);
    check("synthetic 20k with a 1500-edge row", &g, &opts(1), ALL_ON_DEVICE);
}

/// f64 weights (not f32-exact) take the other upload path.
#[test]
fn f64_weights() {
    let (n, e) = common::planted(4, 40, 50, 0.3, 0.01);
    let e: Vec<_> = e.into_iter().map(|(u, v, w)| (u, v, w + 1e-9)).collect(); // no longer f32-exact
    let g = common::graph(n, &e);
    check("planted 2k, f64 weights, all on device", &g, &opts(1), ALL_ON_DEVICE);
}

/// Below the thresholds `leiden_gpu` is `leiden` with `Apply::OrderFree`, exactly, on any device.
#[test]
fn small_graphs_run_on_the_host() {
    let _one = DEVICE.lock().unwrap_or_else(|e| e.into_inner());
    let g = common::sparse_planted(5, 5_000, 10, 12, 0.8, 0, 0);
    let o = opts(-1);
    assert_eq!(leiden_gpu(&g, &o), leiden(&g, &Options { apply: Apply::OrderFree, ..o }));
}

/// The bench graphs: `-- --ignored --nocapture`. One iteration on every graph present, converged on
/// those up to 200k vertices; then best-of-3 times (one iteration) of the CPU default, the CPU
/// reference and the GPU.
#[test]
#[ignore]
fn bench_graphs() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    for name in ["hao2021_10k", "hao2021_50k", "hao2021_100k", "hao2021_154k", "syn_1M"] {
        let Some(g) = common::bench_graph(name) else { eprintln!("skip {name}: no graph"); continue };
        check(&format!("{name}, 1 iteration"), &g, &opts(1), DEFAULT);
        if g.n() <= 200_000 {
            check(&format!("{name}, converged"), &g, &opts(-1), DEFAULT);
        }
        let best = |f: &dyn Fn() -> Vec<u32>| {
            (0..3).map(|_| { let t = std::time::Instant::now(); std::hint::black_box(f()); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min)
        };
        let o = opts(1);
        let _one = DEVICE.lock().unwrap_or_else(|e| e.into_inner()); // `check` locks it itself
        let t_default = best(&|| leiden(&g, &o));
        let t_ref = best(&|| leiden(&g, &Options { apply: Apply::OrderFree, ..o.clone() }));
        let t_gpu = best(&|| leiden_gpu(&g, &o));
        eprintln!("{name}: n {} one iteration, best of 3: CPU default {t_default:.3} s | CPU OrderFree {t_ref:.3} s | GPU {t_gpu:.3} s ({threads} host threads)", g.n());
    }
}
