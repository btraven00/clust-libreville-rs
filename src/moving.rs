//! Local moving, deterministic by construction.
//!
//! Vertices are coloured so that each colour is an independent set: greedy first-fit over a
//! seeded order (the colour-group scheme follows leiden-rs 0.8.1, `parallel.rs::greedy_coloring`
//! and `local_moving_parallel`, MIT), computed in parallel by Jones–Plassmann (see `colour`). Colour by
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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};

use rayon::prelude::*;

use crate::graph::{Csr, Weight};
use crate::Apply;
use crate::rng;

const MIN_LEN: usize = 64; // vertices per rayon task; any value gives the same output. 64 beat 256 by 7-10% (100k-1M, 8 threads): per-vertex work is uneven
const JP_MIN_N: usize = 100_000; // below this, colour sequentially; either way gives the same colouring

thread_local! {
    // per-thread scratch, kept across calls (GVE's per-thread hashtables): weight from the
    // current vertex to each community, and the communities touched. All zero between uses.
    static SCRATCH: RefCell<(Vec<f64>, Vec<u32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    // colours used by a vertex's earlier neighbours; all false between uses
    static USED: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
}

const MAX_ROUNDS: usize = 1000; // ponytail: safety cap; GVE stops at 20 with a gain tolerance (phase 3)
pub(crate) const NEW: u32 = u32::MAX; // "move to an empty community", resolved at apply time

/// Per-community state of the apply step, valid while `stamp` is the current group's id.
#[derive(Clone, Copy, Default)]
struct ApplyState {
    tot0: f64, // total at the group's start
    inc: f64,  // weight of this group's proposals into it, so far (group order)
    outg: f64, // weight of this group's proposals out of it, so far
    size0: u32,
    outc: u32, // number of proposals out of it, so far
    stamp: u32,
}

pub(crate) struct Move {
    pub(crate) v: u32,
    pub(crate) to: u32,
    pub(crate) w_to: f64,
    pub(crate) w_from: f64,
}

/// Colour groups of `g` from a seeded vertex order.
pub(crate) fn colouring<W: Weight>(g: &Csr<W>, seed: u64, step: u64) -> Vec<Vec<u32>> {
    let t = std::time::Instant::now();
    let groups = colour(g, &rng::order(g.n(), seed, rng::COLOUR_ORDER, step));
    if std::env::var_os("DETLEIDEN_TRACE").is_some() {
        eprintln!("    colouring n {}: {:.4} s", g.n(), t.elapsed().as_secs_f64());
    }
    groups
}

/// The sequential greedy pass (`colour` reproduces it exactly): each vertex in `order` takes the
/// smallest colour no earlier-coloured neighbour has.
fn colour_sequential<W: Weight>(g: &Csr<W>, order: &[u32]) -> Vec<Vec<u32>> {
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

/// Greedy first-fit colouring in `order`: in parallel (Jones–Plassmann) on large graphs, sequentially
/// otherwise; the colouring is identical either way.
///
/// Jones–Plassmann: a vertex is coloured once all its neighbours that come earlier in `order`
/// are, taking the smallest colour none of them has. Each vertex then sees exactly what the
/// sequential greedy pass would show it (its earlier neighbours, already coloured), so the
/// colouring is the same, and grouping in `order` gives the same groups in the same order. The
/// vertices coloured in one round are independent (of two adjacent ones, the later waits for the
/// earlier), so a round runs in parallel. Rounds are bounded by the longest chain of increasing
/// priority, which is short for random priorities.
///
/// - M. T. Jones and P. E. Plassmann, "A parallel graph coloring heuristic", SIAM J. Sci. Comput.
///   14(3):654–669, 1993, doi:10.1137/0914041 (the algorithm; random priorities).
/// - W. Hasenplaugh, T. Kaler, T. B. Schardl and C. E. Leiserson, "Ordering heuristics for parallel
///   graph coloring", SPAA 2014, doi:10.1145/2612669.2612697 (JP with a priority order computes the
///   serial greedy colouring of that order; the dependency-counter, frontier-based formulation).
///
/// No implementation was used as a reference; the dependency counters are atomic integers (the
/// result does not depend on the order of the decrements).
fn colour<W: Weight>(g: &Csr<W>, order: &[u32]) -> Vec<Vec<u32>> {
    // Jones–Plassmann's rounds cost more than they save on small (dense, coarse) graphs: at 1M
    // the level-0 colouring drops 0.35 -> 0.14 s at 8 threads; at 55k it doubles.
    if g.n() < JP_MIN_N || rayon::current_num_threads() == 1 {
        colour_sequential(g, order)
    } else {
        colour_parallel(g, order)
    }
}

/// The Jones–Plassmann colouring (see `colour` for the idea and references).
fn colour_parallel<W: Weight>(g: &Csr<W>, order: &[u32]) -> Vec<Vec<u32>> {
    let n = g.n();
    let mut rank = vec![0u32; n];
    for (i, &v) in order.iter().enumerate() {
        rank[v as usize] = i as u32;
    }
    let earlier = |v: u32, u: u32| u != v && rank[u as usize] < rank[v as usize];
    // waiting[v]: earlier neighbours not yet coloured
    let waiting: Vec<AtomicU32> = (0..n as u32)
        .into_par_iter()
        .map(|v| AtomicU32::new(g.row(v as usize).0.iter().filter(|&&u| earlier(v, u)).count() as u32))
        .collect();
    let waiting = &waiting;
    let colour: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(u32::MAX)).collect();
    let mut frontier: Vec<u32> = (0..n as u32).into_par_iter().filter(|&v| waiting[v as usize].load(Relaxed) == 0).collect();
    while !frontier.is_empty() {
        frontier.par_iter().with_min_len(MIN_LEN).for_each(|&v| {
            USED.with(|used| {
                let used = &mut *used.borrow_mut();
                let t = g.row(v as usize).0;
                for &u in t {
                    if earlier(v, u) {
                        let c = colour[u as usize].load(Relaxed) as usize;
                        if used.len() <= c {
                            used.resize(c + 1, false);
                        }
                        used[c] = true;
                    }
                }
                let c = used.iter().position(|&x| !x).unwrap_or(used.len());
                for &u in t {
                    if earlier(v, u) {
                        used[colour[u as usize].load(Relaxed) as usize] = false;
                    }
                }
                colour[v as usize].store(c as u32, Relaxed);
            })
        });
        frontier = frontier
            .par_iter()
            .with_min_len(MIN_LEN)
            .flat_map_iter(|&v| {
                g.row(v as usize).0.iter().copied().filter(move |&u| earlier(u, v) && waiting[u as usize].fetch_sub(1, Relaxed) == 1)
            })
            .collect();
    }
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for &v in order {
        let c = colour[v as usize].load(Relaxed) as usize;
        if groups.len() <= c {
            groups.resize(c + 1, Vec::new());
        }
        groups[c].push(v);
    }
    groups
}

/// Moves vertices of `g` between communities `comm` (ids < n) to increase modularity with
/// resolution `gamma`. Stops when a round moves nothing, or when its modularity gain (summed in
/// apply order) is at most `tol` (GVE-Leiden's convergence rule; 0: until nothing moves).
/// Returns (colours, rounds) for the trace.
///
/// `apply` chooses how a group's decided moves are applied (`crate::Apply`): `Sequential` (the
/// CPU fast path, in-order re-check against the live totals) or `OrderFree` (rule C: what a GPU
/// can reproduce in parallel; selecting it on the CPU gives the GPU's labels).
pub(crate) fn move_nodes<W: Weight>(g: &Csr<W>, comm: &mut [u32], gamma: f64, tol: f64, groups: &[Vec<u32>], apply: Apply) -> (usize, usize) {
    let n = g.n();
    if n == 0 || g.total <= 0.0 {
        return (0, 0);
    }
    let scale = gamma / g.total;
    // community ids: < n (Sequential: an empty-community stack); < 2n (OrderFree: a vertex v
    // that leaves for a new community takes id n + v)
    let ids = if apply == Apply::OrderFree { 2 * n } else { n };
    let mut tot = vec![0.0f64; ids];
    let mut size = vec![0u32; ids];
    for v in 0..n {
        tot[comm[v] as usize] += g.strength[v];
        size[comm[v] as usize] += 1;
    }
    // per community, lazily reset per group (generation stamp): proposal prefix sums in group
    // order, and the totals and sizes frozen at the group's start
    // (one struct per community: the apply step touches all fields of a community together, so
    // this is one cache line instead of six)
    let mut cs = vec![ApplyState::default(); if apply == Apply::OrderFree { ids } else { 0 }];
    let mut gid = 0u32;
    let mut empty: Vec<u32> = if apply == Apply::Sequential { (0..n as u32).rev().filter(|&c| size[c as usize] == 0).collect() } else { Vec::new() };


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
                        if wt.len() < ids {
                            wt.resize(ids, 0.0);
                        }
                        best_move(g, v, frozen, &tot, &size, scale, wt, touched)
                    })
                })
                .collect();
            // Apply. A move is accepted against the frozen totals, made pessimistic by every
            // *earlier proposal* of this group (group order): the target counts all proposals into
            // it, the source loses all proposals out of it. Only prefix sums over proposals, so no
            // accept depends on another accept: parallel-friendly (a GPU does it with a stable
            // sort and per-community scans in the same order) and bit-identical either way.
            moved.clear();
            if apply == Apply::Sequential {
                // apply, in order, re-checking the gain against the totals as they are now
                for m in &moves {
                    let (v, k) = (m.v as usize, g.strength[m.v as usize]);
                    let from = comm[v] as usize;
                    let stay = m.w_from - scale * k * (tot[from] - k);
                    let go = if m.to == NEW { 0.0 } else { m.w_to - scale * k * tot[m.to as usize] };
                    if go <= stay || (m.to == NEW && size[from] == 1) {
                        continue; // no longer an improvement, or already alone
                    }
                    round_gain += go - stay;
                    let to = if m.to == NEW { empty.pop().expect("an empty community exists while one has 2+ members") } else { m.to } as usize;
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
            } else {
                // One pass: acceptance reads the frozen snapshot plus the prefixes, while the live
                // totals update per community in group order.
                gid += 1;
                for m in &moves {
                    let (v, k) = (m.v as usize, g.strength[m.v as usize]);
                    let from = comm[v] as usize;
                    let to = if m.to == NEW { n + v } else { m.to as usize };
                    for x in [from, to] {
                        if cs[x].stamp != gid {
                            cs[x] = ApplyState { stamp: gid, tot0: tot[x], size0: size[x], ..Default::default() };
                        }
                    }
                    let (f, t) = (cs[from], cs[to]);
                    let stay = m.w_from - scale * k * (f.tot0 - k - f.outg);
                    let (ok, go) = if m.to == NEW {
                        // a new community must be empty now and v must not be left alone already
                        (t.size0 == 0 && f.size0 - f.outc > 1 && 0.0 > stay, 0.0)
                    } else {
                        let go = m.w_to - scale * k * (t.tot0 + t.inc);
                        (go > stay, go)
                    };
                    cs[to].inc += k;
                    cs[from].outg += k;
                    cs[from].outc += 1;
                    if ok {
                        round_gain += go - stay;
                        tot[from] -= k;
                        size[from] -= 1;
                        tot[to] += k;
                        size[to] += 1;
                        comm[v] = to as u32;
                        moved.push(v as u32);
                    }
                }
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
pub(crate) fn best_move<W: Weight>(g: &Csr<W>, v: u32, comm: &[u32], tot: &[f64], size: &[u32], scale: f64,
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
        wt[c as usize] += Into::<f64>::into(w);
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

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn parallel_colouring_equals_sequential_greedy() {
        for seed in 0..60u64 {
            let r = |i: u64| rng::unit(rng::key(seed, 77, i, 0));
            let n = 50 + (r(0) * 900.0) as usize;
            let p = 2.0 + r(1) * 30.0; // mean degree
            let mut rows: Vec<Vec<(u32, f64)>> = vec![vec![]; n];
            for u in 0..n {
                for v in u + 1..n {
                    if r(10 + (u * n + v) as u64) < p / n as f64 {
                        rows[u].push((v as u32, 1.0));
                        rows[v].push((u as u32, 1.0));
                    }
                }
                if r(3 + u as u64) < 0.05 {
                    rows[u].push((u as u32, 1.0)); // self-loops, as on coarse graphs
                }
            }
            let mut off = vec![0];
            let (mut t, mut w) = (vec![], vec![]);
            for row in &rows {
                t.extend(row.iter().map(|x| x.0));
                w.extend(row.iter().map(|x| x.1));
                off.push(t.len());
            }
            let g = Csr::from_parts(off, t, w);
            let order = rng::order(n, seed, rng::COLOUR_ORDER, 0);
            for threads in [1, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                assert_eq!(pool.install(|| colour_parallel(&g, &order)), colour_sequential(&g, &order), "seed {seed}, {threads} threads");
            }
        }
    }
}
