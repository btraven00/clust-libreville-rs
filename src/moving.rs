//! Local moving, deterministic by construction.
//!
//! Vertices are coloured so that each colour is an independent set (greedy first-fit over a
//! seeded order; adapted from leiden-rs 0.8.1, `parallel.rs::greedy_coloring`, MIT). Colour by
//! colour, every vertex's best move is *decided* against the state frozen before its group (no
//! neighbour of it moves inside the group, so its edge weights to communities are exact), then
//! moves are *applied* in group order with a gain re-check against the updated community totals.
//! The decide step runs in parallel (order-preserving collect); the apply step is sequential,
//! O(moves); marking the moved vertices' neighbours runs in parallel after it.
//!
//! Pruning as in GVE-Leiden: only vertices whose neighbourhood changed are re-examined. A flag
//! is set for each neighbour of a moved vertex that is now in another community. Neighbours of a
//! group never move inside the group, so marking after the whole group equals marking during
//! its apply, and the work set of every group is deterministic. The flag writes race, but they
//! all write `true`, so the result doesn't depend on their order.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use rayon::prelude::*;

use crate::graph::Graph;
use crate::rng;

const MIN_LEN: usize = 256; // vertices per rayon task; any value gives the same output

thread_local! {
    // per-thread scratch, kept across calls (GVE's per-thread hashtables): weight from the
    // current vertex to each community, and the communities touched. All zero between uses.
    static SCRATCH: RefCell<(Vec<f64>, Vec<u32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

const MAX_ROUNDS: usize = 1000; // ponytail: safety cap; GVE stops at 20 with a gain tolerance (phase 3)
const NEW: u32 = u32::MAX; // "move to an empty community", resolved at apply time

struct Move {
    v: u32,
    to: u32,
    w_to: f64,
    w_from: f64,
}

/// Colour groups of `g` from a seeded vertex order.
pub(crate) fn colouring(g: &Graph, seed: u64, step: u64) -> Vec<Vec<u32>> {
    let t = std::time::Instant::now();
    let groups = colour(g, &rng::order(g.n(), seed, rng::COLOUR_ORDER, step));
    if std::env::var_os("DETLEIDEN_TRACE").is_some() {
        eprintln!("    colouring n {}: {:.4} s", g.n(), t.elapsed().as_secs_f64());
    }
    groups
}

fn colour(g: &Graph, order: &[u32]) -> Vec<Vec<u32>> {
    let mut colour = vec![u32::MAX; g.n()];
    let mut used: Vec<bool> = Vec::new();
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for &v in order {
        let (t, _) = g.row(v as usize);
        for &u in t {
            let c = colour[u as usize];
            if c != u32::MAX && u != v {
                if used.len() <= c as usize {
                    used.resize(c as usize + 1, false);
                }
                used[c as usize] = true;
            }
        }
        let c = used.iter().position(|&x| !x).unwrap_or(used.len());
        for &u in t {
            let c = colour[u as usize];
            if c != u32::MAX && (c as usize) < used.len() {
                used[c as usize] = false;
            }
        }
        colour[v as usize] = c as u32;
        if groups.len() <= c {
            groups.push(Vec::new());
        }
        groups[c].push(v);
    }
    groups
}

/// Moves vertices of `g` between communities `comm` (ids < n) to increase modularity with
/// resolution `gamma`. Stops when a round moves nothing, or when its modularity gain (summed in
/// apply order) is at most `tol` (GVE-Leiden's convergence rule; 0: until nothing moves).
/// Returns (colours, rounds) for the trace.
pub(crate) fn move_nodes(g: &Graph, comm: &mut [u32], gamma: f64, tol: f64, groups: &[Vec<u32>]) -> (usize, usize) {
    let n = g.n();
    if n == 0 || g.total <= 0.0 {
        return (0, 0);
    }
    let scale = gamma / g.total;
    let mut tot = vec![0.0f64; n];
    let mut size = vec![0u32; n];
    for v in 0..n {
        tot[comm[v] as usize] += g.strength[v];
        size[comm[v] as usize] += 1;
    }
    let mut empty: Vec<u32> = (0..n as u32).rev().filter(|&c| size[c as usize] == 0).collect();


    let affected: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(true)).collect();
    let mut moved: Vec<u32> = Vec::new();
    let mut rounds = 0;

    for _ in 0..MAX_ROUNDS {
        rounds += 1;
        let mut round_moves = 0usize;
        let mut round_gain = 0.0f64; // weight units; modularity gain = 2 * round_gain / 2m
        for group in groups {
            // decide, in parallel, against the frozen state
            let frozen: &[u32] = comm;
            let moves: Vec<Move> = group
                .par_iter()
                .with_min_len(MIN_LEN)
                .filter_map(|&v| {
                    if !affected[v as usize].swap(false, Relaxed) {
                        return None;
                    }
                    SCRATCH.with(|s| {
                        let (wt, touched) = &mut *s.borrow_mut();
                        if wt.len() < n {
                            wt.resize(n, 0.0);
                        }
                        best_move(g, v, frozen, &tot, &size, scale, wt, touched)
                    })
                })
                .collect();
            // apply, in order, re-checking the gain against the totals as they are now
            moved.clear();
            for m in &moves {
                let (v, k) = (m.v as usize, g.strength[m.v as usize]);
                let from = comm[v] as usize;
                let stay = m.w_from - scale * k * (tot[from] - k);
                let go = if m.to == NEW { 0.0 } else { m.w_to - scale * k * tot[m.to as usize] };
                if go <= stay || (m.to == NEW && size[from] == 1) {
                    continue; // no longer an improvement, or already alone
                }
                round_gain += go - stay;
                let to = if m.to == NEW {
                    empty.pop().expect("an empty community exists while one has 2+ members")
                } else {
                    m.to
                } as usize;
                tot[from] -= k;
                size[from] -= 1;
                if size[from] == 0 {
                    empty.push(from as u32);
                }
                tot[to] += k;
                size[to] += 1;
                comm[v] = to as u32;
                moved.push(v as u32);
            }
            round_moves += moved.len();
            // mark, in parallel, the neighbours now in another community
            let comm: &[u32] = comm;
            moved.par_iter().with_min_len(MIN_LEN).for_each(|&v| {
                let c = comm[v as usize];
                for &u in g.row(v as usize).0 {
                    if comm[u as usize] != c {
                        affected[u as usize].store(true, Relaxed);
                    }
                }
            });
        }
        if round_moves == 0 || 2.0 * round_gain / g.total <= tol {
            break;
        }
    }
    (groups.len(), rounds)
}

/// The best community for `v` against the current state, or None to stay. Ties keep the
/// earlier candidate in scan order (own community first, then first-touched neighbours).
#[allow(clippy::too_many_arguments)]
fn best_move(g: &Graph, v: u32, comm: &[u32], tot: &[f64], size: &[u32], scale: f64,
             wt: &mut [f64], touched: &mut Vec<u32>) -> Option<Move> {
    let (t, w) = g.row(v as usize);
    let own = comm[v as usize];
    for (&u, &w) in t.iter().zip(w) {
        if u == v {
            continue;
        }
        let c = comm[u as usize];
        if wt[c as usize] == 0.0 {
            touched.push(c); // a zero-weight edge can push c twice: harmless, re-evaluated and cleared
        }
        wt[c as usize] += w;
    }
    let k = g.strength[v as usize];
    let w_from = wt[own as usize];
    let mut best = (own, w_from - scale * k * (tot[own as usize] - k));
    if size[own as usize] > 1 && 0.0 > best.1 {
        best = (NEW, 0.0);
    }
    for &c in touched.iter() {
        if c == own {
            continue;
        }
        let gain = wt[c as usize] - scale * k * tot[c as usize];
        if gain > best.1 {
            best = (c, gain);
        }
    }
    let w_to = if best.0 == NEW { 0.0 } else { wt[best.0 as usize] };
    for &c in touched.iter() {
        wt[c as usize] = 0.0;
    }
    touched.clear();
    (best.0 != own).then_some(Move { v, to: best.0, w_to, w_from })
}
