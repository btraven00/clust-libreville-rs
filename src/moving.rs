//! Local moving, deterministic by construction.
//!
//! Vertices are coloured so that each colour is an independent set (greedy first-fit over a
//! seeded order; adapted from leiden-rs 0.8.1, `parallel.rs::greedy_coloring`, MIT). Colour by
//! colour, every vertex's best move is *decided* against the state frozen before its group (no
//! neighbour of it moves inside the group, so its edge weights to communities are exact), then
//! moves are *applied* in group order with a gain re-check against the updated community totals.
//! The decide step is the part phase 2 runs in parallel; the apply step stays sequential.
//!
//! Pruning as in GVE-Leiden: only vertices whose neighbourhood changed are re-examined. The
//! flags are set in the apply step, so the work set of every group is deterministic.

use crate::graph::Graph;
use crate::rng;

const MAX_ROUNDS: usize = 1000; // ponytail: safety cap; GVE stops at 20 with a gain tolerance (phase 3)
const NEW: u32 = u32::MAX; // "move to an empty community", resolved at apply time

struct Move {
    v: u32,
    to: u32,
    w_to: f64,
    w_from: f64,
}

pub(crate) fn colour(g: &Graph, order: &[u32]) -> Vec<Vec<u32>> {
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
/// resolution `gamma`. Returns (colours, rounds) for the trace.
pub(crate) fn move_nodes(g: &Graph, comm: &mut [u32], gamma: f64, seed: u64, step: u64) -> (usize, usize) {
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
    let groups = colour(g, &rng::order(n, seed, rng::COLOUR_ORDER, step));

    let mut affected = vec![true; n];
    let mut wt = vec![0.0f64; n]; // scratch: weight from v to each community
    let mut touched: Vec<u32> = Vec::new();
    let mut moves: Vec<Move> = Vec::new();
    let mut rounds = 0;

    for _ in 0..MAX_ROUNDS {
        rounds += 1;
        let mut round_moves = 0usize;
        for group in &groups {
            // decide, against the frozen state
            moves.clear();
            for &v in group {
                if !affected[v as usize] {
                    continue;
                }
                affected[v as usize] = false;
                if let Some(m) = best_move(g, v, comm, &tot, &size, scale, &mut wt, &mut touched) {
                    moves.push(m);
                }
            }
            // apply, in order, re-checking the gain against the totals as they are now
            for m in &moves {
                let (v, k) = (m.v as usize, g.strength[m.v as usize]);
                let from = comm[v] as usize;
                let stay = m.w_from - scale * k * (tot[from] - k);
                let to = if m.to == NEW {
                    if size[from] == 1 {
                        continue; // already alone
                    }
                    if 0.0 <= stay {
                        continue;
                    }
                    empty.pop().expect("an empty community exists while one has 2+ members")
                } else {
                    if m.w_to - scale * k * tot[m.to as usize] <= stay {
                        continue;
                    }
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
                let (t, _) = g.row(v);
                for &u in t {
                    if comm[u as usize] as usize != to {
                        affected[u as usize] = true;
                    }
                }
                round_moves += 1;
            }
        }
        if round_moves == 0 {
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
