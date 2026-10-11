//! Phase 2's gate: nothing, not even a last bit of a coarse weight, depends on the thread count.

use detleiden::{leiden_traced, Apply, Graph, Options};

mod common;
use common::{graph, planted};

const THREADS: [usize; 6] = [1, 2, 3, 4, 8, 16];

fn fnv(labels: &[u32]) -> u64 {
    labels.iter().flat_map(|x| x.to_le_bytes()).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

fn same_at_every_thread_count(g: &Graph, o: &Options) -> (Vec<u32>, Vec<u64>) {
    let base = leiden_traced(g, &Options { threads: Some(1), ..o.clone() });
    for t in THREADS {
        let r = leiden_traced(g, &Options { threads: Some(t), ..o.clone() });
        assert_eq!(r.1, base.1, "level hashes differ at {t} threads");
        assert_eq!(r.0, base.0, "labels differ at {t} threads");
    }
    base
}

#[test]
fn random_graphs() {
    for seed in 0..12u64 {
        let (n, e) = planted(seed, 8 + seed as usize % 5, 250 + 40 * seed as usize, 0.04, 0.002);
        let g = graph(n, &e);
        for theta in [0.01, 0.0] {
            for n_iterations in [1, -1] {
                for apply in [Apply::Sequential, Apply::OrderFree] {
                    same_at_every_thread_count(&g, &Options { seed, theta, n_iterations, apply, ..Default::default() });
                }
            }
        }
    }
}

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

/// The hao2021 graphs (bench/graphs, git-ignored): every thread count, and the locked labels
/// exactly. Sequential: phase 1 (4783e5d) and every output-preserving change since. OrderFree:
/// rule C moving + fixed-point sorted-column aggregation + synchronous refinement (the GPU
/// reference; NI vs Sequential and both converged references, PLAN.md).
/// `cargo test --release -- --ignored`
#[test]
#[ignore]
fn hao2021_matches_lock_at_every_thread_count() {
    let lock = [
        ("hao2021_10k", 100, Apply::Sequential, 0x87ded254a27d8377u64), ("hao2021_10k", 101, Apply::Sequential, 0x927306921a2dcc27),
        ("hao2021_50k", 100, Apply::Sequential, 0x2e14cd1b4085cba9), ("hao2021_50k", 101, Apply::Sequential, 0x6cd3a3991e9d2160),
        ("hao2021_100k", 100, Apply::Sequential, 0x7ec81e3557da7095), ("hao2021_100k", 101, Apply::Sequential, 0x12c848f6a5f824ac),
        ("hao2021_10k", 100, Apply::OrderFree, 0x8d7a414343580a2b), ("hao2021_10k", 101, Apply::OrderFree, 0x20f940b597024e5e),
        ("hao2021_50k", 100, Apply::OrderFree, 0x2d93ed96feb7741b), ("hao2021_50k", 101, Apply::OrderFree, 0x64e2c1dc466c0b35),
        ("hao2021_100k", 100, Apply::OrderFree, 0x880c906aa8088169), ("hao2021_100k", 101, Apply::OrderFree, 0x26bfb8b9d38a8315),
    ];
    for (name, seed, apply, want) in lock {
        let Some(g) = load(name) else { eprintln!("skip {name}: no graph"); continue };
        let (labels, _) = same_at_every_thread_count(&g, &Options { seed, apply, ..Default::default() });
        assert_eq!(fnv(&labels), want, "{name} seed {seed} {apply:?}: not the locked labels");
    }
}
