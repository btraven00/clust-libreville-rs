//! Phase 2's gate: nothing, not even a last bit of a coarse weight, depends on the thread count.

use clust_libreville_rs::{leiden_traced, Apply, Graph, Options};

mod common;
use common::{bench_graph, fnv, graph, planted, sparse_planted};

const THREADS: [usize; 6] = [1, 2, 3, 4, 8, 16];

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

#[test]
fn parallel_colouring_at_every_thread_count() {
    // n >= JP_MIN_N (100k), the size real data runs at: more than one thread colours by
    // Jones–Plassmann, one thread sequentially, and both must give the same labels and levels
    let g = sparse_planted(1, 120_000, 40, 12, 0.8, 0, 0);
    for apply in [Apply::Sequential, Apply::OrderFree] {
        let o = Options { seed: 3, n_iterations: 1, apply, ..Default::default() };
        let base = leiden_traced(&g, &Options { threads: Some(1), ..o.clone() });
        for t in [2, 8] {
            assert_eq!(leiden_traced(&g, &Options { threads: Some(t), ..o.clone() }), base, "{apply:?} at {t} threads");
        }
    }
}

/// The hao2021 graphs (bench/graphs, git-ignored; `DETLEIDEN_GRAPHS`): every thread count, and the locked labels
/// exactly. Sequential: phase 1 (e3c0295) and every output-preserving change since. OrderFree:
/// rule C moving + fixed-point sorted-column aggregation + synchronous refinement (the GPU
/// reference; NI vs Sequential and both converged references, docs/design.md).
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
        let Some(g) = bench_graph(name) else { eprintln!("skip {name}: no graph"); continue };
        let (labels, _) = same_at_every_thread_count(&g, &Options { seed, apply, ..Default::default() });
        assert_eq!(fnv(&labels), want, "{name} seed {seed} {apply:?}: not the locked labels");
    }
}
