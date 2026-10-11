//! Refinement (Traag, Waltman & van Eck 2019, Algorithm 3), deterministic by construction.
//!
//! Inside each community S of the moved partition, start from singletons. Visit the
//! well-connected vertices of S in a seeded order. A vertex still alone may join a
//! well-connected refined cluster C of S with gain >= 0, drawn with probability
//! proportional to exp(gain / theta) (theta = 0: the best gain, i.e. GVE's greedy rule).
//!
//! Gains are in edge-weight units, as in igraph (`mergenodes`, beta = theta):
//! gain = w(v, C) - gamma * k_v * K_C / 2m. "Well-connected": E(X, S - X) >= gamma * K_X * (K_S - K_X) / 2m.
//! Unlike igraph 0.10.16, the external weight of a vertex that stays alone is not
//! double-counted (ext_C <- ext_C + ext_v - 2 w(v, C) on every insert).
//!
//! Communities are independent, so they are refined in parallel (largest first, for balance),
//! each with its own arrays indexed by position in the community. Inside a community the visit
//! order and every sum are fixed, so the result doesn't depend on the thread count. Refined
//! cluster ids are the ids of their founding vertices; no O(n) reset per community (leiden-rs).

use rayon::prelude::*;

use std::cell::RefCell;

use crate::graph::{Csr, Weight};
use crate::moving::{fx, to_fx};
use crate::{rng, Apply};

/// Refined labels (dense, ordered by first occurrence) and their count. Every refined cluster
/// lies inside one community of `comm`. `Apply::OrderFree` refines synchronously over the level's
/// colour groups (`refine_orderfree`), the GPU-reproducible variant.
/// `groups`: the level's colouring (local moving's), used by OrderFree only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine<W: Weight>(g: &Csr<W>, comm: &[u32], gamma: f64, theta: f64, seed: u64, step: u64, apply: Apply,
                                groups: &[Vec<u32>]) -> (Vec<u32>, usize) {
    if apply == Apply::OrderFree {
        return refine_orderfree(g, comm, groups, gamma, theta, seed, step);
    }
    let n = g.n();
    let scale = if g.total > 0.0 { gamma / g.total } else { 0.0 };
    let members = bucket(comm, n);
    let mut pos = vec![0u32; n]; // position of each vertex in its community
    for s in &members {
        for (i, &v) in s.iter().enumerate() {
            pos[v as usize] = i as u32;
        }
    }
    let mut big: Vec<&Vec<u32>> = members.iter().filter(|s| s.len() > 1).collect();
    big.sort_by_key(|s| std::cmp::Reverse(s.len())); // stable: ties keep community order
    // one task per community: rayon pre-splits a slice into ~one chunk per thread, and the chunk
    // holding the largest communities would otherwise run them all on one thread
    let results: Vec<Vec<u32>> = big
        .par_iter()
        .with_max_len(1)
        .map(|s| refine_one(g, comm, s, &pos, scale, theta, seed, step))
        .collect();
    let mut refined: Vec<u32> = (0..n as u32).collect();
    for (s, r) in big.iter().zip(&results) {
        for (&v, &c) in s.iter().zip(r) {
            refined[v as usize] = s[c as usize];
        }
    }
    dense(&refined)
}

/// One community `s` (vertices in id order): the refined cluster of each member, as a position in `s`.
#[allow(clippy::too_many_arguments)]
fn refine_one<W: Weight>(g: &Csr<W>, comm: &[u32], s: &[u32], pos: &[u32], scale: f64, theta: f64, seed: u64, step: u64) -> Vec<u32> {
    let len = s.len();
    let c_s = comm[s[0] as usize];
    let k_s: f64 = s.iter().map(|&v| g.strength[v as usize]).sum();
    let mut refined: Vec<u32> = (0..len as u32).collect();
    let mut cw: Vec<f64> = s.iter().map(|&v| g.strength[v as usize]).collect(); // weight of each refined cluster
    let mut size = vec![1u32; len];
    // weight from each refined cluster to the rest of its community
    let mut ext: Vec<f64> = s
        .iter()
        .map(|&v| {
            let (t, w) = g.row(v as usize);
            t.iter().zip(w).filter(|&(&u, _)| u != v && comm[u as usize] == c_s).map(|(_, &w)| Into::<f64>::into(w)).sum::<f64>()
        })
        .collect();
    let mut wt = vec![0.0f64; len];
    let mut touched: Vec<u32> = Vec::new();
    let mut cand: Vec<(u32, f64)> = Vec::new();
    let mut p: Vec<f64> = Vec::new();
    let mut order: Vec<(u64, u32)> = s.iter().map(|&v| (rng::key(seed, rng::REFINE_ORDER, step, v as u64), v)).collect();
    order.sort_unstable();

    for &(_, v) in &order {
        let (vi, k) = (v as usize, g.strength[v as usize]);
        let own = refined[pos[vi] as usize] as usize;
        let ext_v = ext[own];
        if size[own] != 1 || ext_v < scale * k * (k_s - k) {
            continue; // merged into, or not well-connected
        }
        // take v out: its singleton becomes empty
        cw[own] = 0.0;
        size[own] = 0;
        ext[own] = 0.0;
        let (t, w) = g.row(vi);
        for (&u, &w) in t.iter().zip(w) {
            if u != v && comm[u as usize] == c_s {
                let c = refined[pos[u as usize] as usize];
                if wt[c as usize] == 0.0 {
                    touched.push(c);
                }
                wt[c as usize] += Into::<f64>::into(w);
            }
        }
        cand.clear();
        cand.push((own as u32, 0.0)); // staying alone: gain 0
        for &c in &touched {
            let ci = c as usize;
            if ci == own || ext[ci] < scale * cw[ci] * (k_s - cw[ci]) {
                continue;
            }
            let gain = wt[ci] - scale * k * cw[ci];
            if gain >= 0.0 {
                cand.push((c, gain));
            }
        }
        let chosen = choose(&cand, theta, rng::key(seed, rng::REFINE_DRAW, step, v as u64), &mut p) as usize;
        let w_vc = if chosen == own { 0.0 } else { wt[chosen] };
        refined[pos[vi] as usize] = chosen as u32;
        cw[chosen] += k;
        size[chosen] += 1;
        ext[chosen] += ext_v - 2.0 * w_vc;
        for &c in &touched {
            wt[c as usize] = 0.0;
        }
        touched.clear();
    }
    refined
}

thread_local! {
    // refine_orderfree's decide scratch: fixed-point weight to each refined cluster (zero between
    // vertices), the clusters touched, candidates, cumulative probabilities
    static OF_SCRATCH: RefCell<(Vec<i64>, Vec<u32>, Vec<(u32, f64)>, Vec<f64>)> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

/// Refinement for `Apply::OrderFree`: Algorithm 3's singleton merges, made synchronous so a GPU
/// reproduces them bit for bit.
///
/// Vertices are visited once, colour group by colour group (the level's colouring: independent
/// sets, also inside every community). In a group, every still-singleton, well-connected vertex
/// decides in parallel against the state frozen at the group's start: its candidates are staying
/// alone (gain 0) and the well-connected refined clusters of its community it touches with gain
/// >= 0, in sorted cluster order, drawn as in `choose`. A proposal into C is then accepted if its
/// gain stays >= 0 against C's frozen weight plus the earlier proposals of the group into C
/// (group order), as in local moving's rule C. Weights, cluster weights and external weights are
/// fixed-point integers (moving::FX), so every sum is order-free. Two proposals of a group are
/// never adjacent, so external weights add exactly; every merge joins an adjacent cluster, so
/// refined clusters stay connected inside their community.
pub(crate) fn refine_orderfree<W: Weight>(g: &Csr<W>, comm: &[u32], groups: &[Vec<u32>], gamma: f64, theta: f64, seed: u64, step: u64) -> (Vec<u32>, usize) {
    let n = g.n();
    let scale = if g.total > 0.0 { gamma / g.total } else { 0.0 };
    let kint: Vec<i64> = g.strength.iter().map(|&k| to_fx(k)).collect();
    let mut ks = vec![0i64; n]; // community weights (community ids < n)
    for v in 0..n {
        ks[comm[v] as usize] += kint[v];
    }
    let mut refd: Vec<u32> = (0..n as u32).collect(); // refined cluster of each vertex (founder id)
    let mut cw = kint.clone(); // refined cluster weights
    let mut size = vec![1u32; n];
    // weight from each refined cluster to the rest of its community
    let mut ext: Vec<i64> = (0..n)
        .into_par_iter()
        .map(|v| {
            let (t, w) = g.row(v);
            t.iter().zip(w).filter(|&(&u, _)| u as usize != v && comm[u as usize] == comm[v]).map(|(_, &w)| to_fx(w.into())).sum()
        })
        .collect();
    let (mut inc, mut cw0, mut stamp) = (vec![0i64; n], vec![0i64; n], vec![0u32; n]);
    let mut gid = 0u32;
    for group in groups {
        let (refd_f, cw_f, size_f, ext_f) = (&refd, &cw, &size, &ext);
        let props: Vec<(u32, u32, i64, i64)> = group
            .par_iter()
            .with_min_len(64)
            .filter_map(|&v| {
                OF_SCRATCH.with(|sc| {
                    let (wt, touched, cand, p) = &mut *sc.borrow_mut();
                    if wt.len() < n {
                        wt.resize(n, 0);
                    }
                    propose(g, comm, refd_f, cw_f, size_f, ext_f, &ks, scale, theta, seed, step, v, wt, touched, cand, p)
                })
            })
            .collect();
        gid += 1;
        for &(v, c, wvc, extv) in &props {
            let (vi, ci) = (v as usize, c as usize);
            if stamp[ci] != gid {
                (stamp[ci], inc[ci], cw0[ci]) = (gid, 0, cw[ci]);
            }
            let ok = fx(wvc) - scale * g.strength[vi] * fx(cw0[ci] + inc[ci]) >= 0.0;
            inc[ci] += kint[vi];
            if ok {
                let own = refd[vi] as usize;
                (cw[own], size[own], ext[own]) = (0, 0, 0);
                refd[vi] = c;
                cw[ci] += kint[vi];
                size[ci] += 1;
                ext[ci] += extv - 2 * wvc;
            }
        }
    }
    dense(&refd)
}

/// One vertex's refinement proposal against the frozen state: (v, target cluster, fixed-point
/// weight from v to it, v's external weight), or None to stay.
#[allow(clippy::too_many_arguments)]
fn propose<W: Weight>(g: &Csr<W>, comm: &[u32], refd: &[u32], cw: &[i64], size: &[u32], ext: &[i64], ks: &[i64], scale: f64,
                      theta: f64, seed: u64, step: u64, v: u32, wt: &mut [i64], touched: &mut Vec<u32>,
                      cand: &mut Vec<(u32, f64)>, p: &mut Vec<f64>) -> Option<(u32, u32, i64, i64)> {
    let vi = v as usize;
    let own = refd[vi];
    if size[own as usize] != 1 {
        return None; // merged into: no longer a singleton
    }
    let (cs, k) = (comm[vi], g.strength[vi]);
    let k_s = fx(ks[cs as usize]);
    let ext_v = ext[own as usize];
    if fx(ext_v) < scale * k * (k_s - k) {
        return None; // not well-connected
    }
    let (t, w) = g.row(vi);
    for (&u, &w) in t.iter().zip(w) {
        if u != v && comm[u as usize] == cs {
            let c = refd[u as usize];
            if wt[c as usize] == 0 {
                touched.push(c);
            }
            wt[c as usize] += to_fx(w.into());
        }
    }
    touched.sort_unstable();
    touched.dedup(); // a zero-weight edge can push a cluster twice
    cand.clear();
    cand.push((own, 0.0)); // staying alone: gain 0
    for &c in touched.iter() {
        let cwf = fx(cw[c as usize]);
        if c != own && fx(ext[c as usize]) >= scale * cwf * (k_s - cwf) {
            let gain = fx(wt[c as usize]) - scale * k * cwf;
            if gain >= 0.0 {
                cand.push((c, gain));
            }
        }
    }
    let chosen = choose(cand, theta, rng::key(seed, rng::REFINE_DRAW, step, v as u64), p);
    let w_vc = if chosen == own { 0 } else { wt[chosen as usize] };
    for &c in touched.iter() {
        wt[c as usize] = 0;
    }
    touched.clear();
    (chosen != own).then_some((v, chosen, w_vc, ext_v))
}

/// Draw from candidates (cluster, gain >= 0), the first being "stay". theta <= 0: the best gain
/// (the first among equals). Otherwise proportional to exp((gain - max) / theta).
fn choose(cand: &[(u32, f64)], theta: f64, key: u64, p: &mut Vec<f64>) -> u32 {
    let best = cand.iter().fold(cand[0], |b, &c| if c.1 > b.1 { c } else { b });
    if theta <= 0.0 || cand.len() == 1 {
        return best.0;
    }
    p.clear();
    let mut sum = 0.0;
    for &(_, gain) in cand {
        sum += rng::exp_neg((gain - best.1) / theta);
        p.push(sum);
    }
    let r = rng::unit(key) * sum;
    cand[p.iter().position(|&x| x > r).unwrap_or(cand.len() - 1)].0
}

/// Vertices grouped by label, each group in vertex-id order (a counting sort).
pub(crate) fn bucket(labels: &[u32], n_labels: usize) -> Vec<Vec<u32>> {
    let mut b: Vec<Vec<u32>> = vec![Vec::new(); n_labels];
    for (v, &l) in labels.iter().enumerate() {
        b[l as usize].push(v as u32);
    }
    b
}

/// Labels renumbered 0.. by first occurrence, and their count.
pub(crate) fn dense(labels: &[u32]) -> (Vec<u32>, usize) {
    let mut map = vec![u32::MAX; labels.len().max(1 + *labels.iter().max().unwrap_or(&0) as usize)];
    let mut next = 0u32;
    let out = labels
        .iter()
        .map(|&l| {
            if map[l as usize] == u32::MAX {
                map[l as usize] = next;
                next += 1;
            }
            map[l as usize]
        })
        .collect();
    (out, next as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random graphs and random (hence often disconnected) partitions: every refined cluster must
    /// lie in one community and be connected through edges inside that community. Skipping
    /// refinement (Louvain) fails this at once.
    #[test]
    fn refined_clusters_are_connected_inside_their_community() {
        for seed in 0..300u64 {
            let r = |i: u64| rng::unit(rng::key(seed, 99, i, 0));
            let n = 20 + (r(0) * 60.0) as usize;
            let (k, p) = (1 + (r(1) * 6.0) as u32, 0.03 + r(2) * 0.2);
            let mut rows: Vec<Vec<(u32, f64)>> = vec![vec![]; n];
            let mut i = 10;
            for u in 0..n {
                for v in u + 1..n {
                    i += 2;
                    if r(i) < p {
                        let w = 0.1 + r(i + 1);
                        rows[u].push((v as u32, w));
                        rows[v].push((u as u32, w));
                    }
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
            let comm: Vec<u32> = (0..n).map(|v| (r(1_000_000 + v as u64) * k as f64) as u32).collect();
            for (theta, apply) in [(0.01, Apply::Sequential), (0.0, Apply::Sequential), (0.01, Apply::OrderFree), (0.0, Apply::OrderFree)] {
                let (refd, nr) = refine(&g, &comm, 1.0, theta, seed, 0, apply, &crate::moving::colouring(&g, seed, 0));
                assert!(nr <= n);
                for c in bucket(&refd, nr) {
                    assert!(c.iter().all(|&v| comm[v as usize] == comm[c[0] as usize]), "seed {seed}: crosses communities");
                    // BFS over edges inside the cluster
                    let inside: std::collections::HashSet<u32> = c.iter().copied().collect();
                    let mut seen = std::collections::HashSet::from([c[0]]);
                    let mut stack = vec![c[0]];
                    while let Some(x) = stack.pop() {
                        for &y in g.row(x as usize).0 {
                            if inside.contains(&y) && seen.insert(y) {
                                stack.push(y);
                            }
                        }
                    }
                    assert_eq!(seen.len(), c.len(), "seed {seed} theta {theta}: refined cluster {c:?} is disconnected");
                }
            }
        }
    }
}
