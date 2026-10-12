//! CHAMP (Weir, Emmons, Gibson, Taylor & Mucha 2017, "Post-processing partitions to identify domains of
//! modularity optimization"): which partition of a pool has the highest modularity at each resolution γ.
//! Each partition's modularity is a line in γ (`modularity_line`), so the answer is the upper envelope
//! of the pool's lines: a few partitions, each optimal on an interval of γ. Optimal relative to the pool
//! only; the pool is the caller's (e.g. `leiden` over a grid of resolutions × seeds).

use crate::{leiden, modularity_line, with_pool, Graph, Metric, Options};
use rayon::prelude::*;

/// `pool[index]` has the highest modularity in the pool for every γ in `[lo, hi]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain {
    pub index: usize,
    pub lo: f64,
    pub hi: f64,
}

/// The upper envelope of the pool's modularity over `[gamma_min, gamma_max]`, in increasing γ.
/// Intervals tile the range. Ties: the flatter line (smaller null term, so it stays ahead as γ grows),
/// then the lower pool index, so duplicate partitions resolve to their first occurrence.
pub fn champ(g: &Graph, pool: &[Vec<u32>], gamma_min: f64, gamma_max: f64) -> Vec<Domain> {
    assert!(0.0 <= gamma_min && gamma_min <= gamma_max, "need 0 <= gamma_min <= gamma_max");
    assert!(pool.iter().all(|l| l.len() == g.n()), "every partition needs one label per vertex");
    let lines: Vec<(f64, f64)> = pool.iter().map(|l| modularity_line(g, l)).collect();
    envelope(&lines, gamma_min, gamma_max)
}

/// How a domain's partition agrees with the partitions around it, in `metric`'s units (NVI: distance,
/// lower = closer; AMI: similarity, higher = closer). NaN where there is nothing to compare with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agreement {
    /// With the previous domain's partition (NaN for the first domain).
    pub prev: f64,
    /// Mean with the previous and the next domain's partitions: a partition close to its neighbours sits
    /// on a plateau, a large step marks a boundary.
    // ponytail: unweighted; weight by neighbour widths if sliver domains turn out to dominate
    pub neighbours: f64,
    /// Mean with the other seeds' partitions at the grid resolution that produced it: how often another
    /// seed would have given (about) the same partition. Algorithmic noise, not structure.
    pub seeds: f64,
}

/// `champ` over the pool of `leiden` at every (resolution, seed), in grid order (pool index =
/// i_resolution · seeds.len() + i_seed, numpy's `meshgrid(indexing="ij")`); the other options from `o`.
/// Returns each domain with its partition and its `Agreement` under `metric`.
#[allow(clippy::too_many_arguments)]
pub fn champ_grid(g: &Graph, resolutions: &[f64], seeds: &[u64], o: &Options, gamma_min: f64, gamma_max: f64,
                  metric: Metric) -> Vec<(Domain, Vec<u32>, Agreement)> {
    let pool: Vec<Vec<u32>> = resolutions.iter()
        .flat_map(|&resolution| seeds.iter().map(move |&seed| Options { resolution, seed, ..o.clone() }))
        .map(|o| leiden(g, &o))
        .collect();
    let domains = champ(g, &pool, gamma_min, gamma_max);
    let s = seeds.len();
    // the caller's cached pool, not rayon's global one (which a fork would leave dead)
    with_pool(o, || {
        let prev: Vec<f64> = (0..domains.len()).into_par_iter()
            .map(|d| if d == 0 { f64::NAN } else { metric.eval(&pool[domains[d - 1].index], &pool[domains[d].index]) })
            .collect();
        let mean = |v: Vec<f64>| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
        domains.par_iter().enumerate().map(|(d, dom)| {
            let neighbours = mean([prev[d], prev.get(d + 1).copied().unwrap_or(f64::NAN)].into_iter()
                .filter(|x| !x.is_nan()).collect());
            let row = dom.index / s * s;
            let mates = mean((row..row + s).filter(|&j| j != dom.index).map(|j| metric.eval(&pool[dom.index], &pool[j])).collect());
            (*dom, pool[dom.index].clone(), Agreement { prev: prev[d], neighbours, seeds: mates })
        }).collect()
    })
}

// ponytail: gift wrapping, O(pool × envelope); sort + stack (O(P log P)) if pools reach ~1e5 partitions
fn envelope(lines: &[(f64, f64)], lo: f64, hi: f64) -> Vec<Domain> {
    let q = |i: usize, x: f64| lines[i].0 - x * lines[i].1;
    // "i before j": higher at x, then flatter (it stays ahead to the right), then lower index
    let better = |x: f64| move |i: &usize, j: &usize| {
        q(*i, x).total_cmp(&q(*j, x)).then(lines[*j].1.total_cmp(&lines[*i].1)).then(j.cmp(i))
    };
    let Some(mut cur) = (0..lines.len()).max_by(better(lo)) else { return vec![] };
    let (mut x, mut out) = (lo, vec![]);
    loop {
        let (a, b) = lines[cur];
        // the first flatter line to overtake cur right of x; slopes strictly decrease, so this ends
        let next = (0..lines.len())
            .filter(|&j| lines[j].1 < b)
            .map(|j| (((a - lines[j].0) / (b - lines[j].1)).max(x), j))
            .min_by(|(xi, i), (xj, j)| xi.total_cmp(xj).then(better(*xi)(j, i)));
        match next {
            Some((y, j)) if y < hi => {
                if y > x {
                    out.push(Domain { index: cur, lo: x, hi: y });
                }
                (cur, x) = (j, y);
            }
            _ => {
                out.push(Domain { index: cur, lo: x, hi });
                return out;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_of_hand_made_lines() {
        // Q = 1 − γ·0.5 (coarse), 2 − γ·1.5 (fine), 0.5 − γ·0.6 (never best), and a duplicate of 0
        let lines = [(1.0, 0.5), (2.0, 1.5), (0.5, 0.6), (1.0, 0.5)];
        let d = envelope(&lines, 0.0, 3.0);
        assert_eq!(d, vec![Domain { index: 1, lo: 0.0, hi: 1.0 }, Domain { index: 0, lo: 1.0, hi: 3.0 }]);
        assert_eq!(envelope(&lines, 2.0, 3.0), vec![Domain { index: 0, lo: 2.0, hi: 3.0 }]);
        assert!(envelope(&[], 0.0, 1.0).is_empty());
    }
}
