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

use crate::graph::Graph;
use crate::rng;

/// Refined labels (dense, ordered by first occurrence) and their count. Every refined cluster
/// lies inside one community of `comm`.
pub(crate) fn refine(g: &Graph, comm: &[u32], gamma: f64, theta: f64, seed: u64, step: u64) -> (Vec<u32>, usize) {
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
fn refine_one(g: &Graph, comm: &[u32], s: &[u32], pos: &[u32], scale: f64, theta: f64, seed: u64, step: u64) -> Vec<u32> {
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
            t.iter().zip(w).filter(|&(&u, _)| u != v && comm[u as usize] == c_s).map(|(_, &w)| w).sum()
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
                wt[c as usize] += w;
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
            let g = Graph::from_parts(off, t, w);
            let comm: Vec<u32> = (0..n).map(|v| (r(1_000_000 + v as u64) * k as f64) as u32).collect();
            for theta in [0.01, 0.0] {
                let (refd, nr) = refine(&g, &comm, 1.0, theta, seed, 0);
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
