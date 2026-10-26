//! Agreement between two partitions of the same vertices, from their contingency table.
//!
//! NVI = (H(a) + H(b) − 2 I(a; b)) / H(a, b), a metric on partitions (Kraskov, Stögbauer, Andrzejak &
//! Grassberger 2005, "Hierarchical clustering using mutual information", Europhys. Lett. 70(2); Vinh, Epps &
//! Bailey 2010, "Information theoretic measures for clusterings comparison", JMLR 11). Same normalisation as
//! PyGenStability's `evaluate_NVI`, so values compare with Markov-stability scans. 0 for identical partitions.
//!
//! AMI = (I − E[I]) / (max(H(a), H(b)) − E[I]), E[I] under the hypergeometric (permutation) model (Vinh et
//! al. 2010, eq. 24a; the "max" form is the one Weir et al. 2017 use with CHAMP). Corrects NVI's blind spot:
//! partitions into many small communities share information by chance. 1 for identical partitions.
//!
//! ECS, per vertex (Gates, Wood, Hetrick & Ahn 2019, "Element-centric clustering comparison unifies overlaps and
//! hierarchy", Sci. Rep. 9:8574; reference implementation CluSim): which vertices the two partitions move.
//!
//! Deterministic: counts come from sorting, every sum runs in sorted order. Natural logarithms.

use std::collections::HashMap;

/// Contingency table of `a` × `b`: the nonzero cells (row, column, count) sorted by (row, column), and the
/// row and column totals, rows and columns numbered by sorted label.
struct Table {
    cells: Vec<(usize, usize, u64)>,
    rows: Vec<u64>,
    cols: Vec<u64>,
    n: u64,
}

fn table(a: &[u32], b: &[u32]) -> Table {
    assert_eq!(a.len(), b.len(), "partitions of different sizes");
    let ids = |l: &[u32]| {
        let mut s = l.to_vec();
        s.sort_unstable();
        s.dedup();
        s
    };
    let (ra, rb) = (ids(a), ids(b));
    let mut keys: Vec<u64> = a.iter().zip(b).map(|(&x, &y)| (x as u64) << 32 | y as u64).collect();
    keys.sort_unstable();
    let (mut rows, mut cols) = (vec![0u64; ra.len()], vec![0u64; rb.len()]);
    let mut cells: Vec<(usize, usize, u64)> = vec![];
    for k in keys {
        let (i, j) = (ra.binary_search(&((k >> 32) as u32)).unwrap(), rb.binary_search(&(k as u32)).unwrap());
        rows[i] += 1;
        cols[j] += 1;
        match cells.last_mut() {
            Some(c) if (c.0, c.1) == (i, j) => c.2 += 1,
            _ => cells.push((i, j, 1)),
        }
    }
    Table { cells, rows, cols, n: a.len() as u64 }
}

/// Entropy of the counts, which sum to `n`: ln n − (1/n) Σ c ln c.
fn entropy(counts: impl Iterator<Item = u64>, n: u64) -> f64 {
    let n = n as f64;
    n.ln() - counts.map(|c| c as f64 * (c as f64).ln()).sum::<f64>() / n
}

/// (H(a), H(b), H(a, b)).
fn entropies(t: &Table) -> (f64, f64, f64) {
    (entropy(t.rows.iter().copied(), t.n), entropy(t.cols.iter().copied(), t.n),
     entropy(t.cells.iter().map(|c| c.2), t.n))
}

/// Normalised variation of information between `a` and `b` (same length, any labels), in [0, 1].
pub fn nvi(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let (ha, hb, hab) = entropies(&table(a, b));
    if hab < 1e-12 {
        return 0.0; // both one community
    }
    ((2.0 * hab - ha - hb) / hab).max(0.0) // H(a) + H(b) − 2I = 2H(a, b) − H(a) − H(b)
}

/// Adjusted mutual information between `a` and `b` (same length, any labels); ≤ 1, ≈ 0 for unrelated.
pub fn ami(a: &[u32], b: &[u32]) -> f64 {
    let t = table(a, b);
    let (ha, hb, hab) = entropies(&t);
    if ha.max(hb) < 1e-12 {
        return 1.0; // both one community (or empty)
    }
    let mi = ha + hb - hab;
    let emi = expected_mi(&t);
    let d = ha.max(hb) - emi;
    if d.abs() < 1e-12 { 0.0 } else { (mi - emi) / d }
}

/// E[I(a; b)] over random partitions with the same row and column totals (Vinh et al. 2010, eq. 24a).
// Σ over all row × column pairs, O(n · min(#rows, #cols)) worst case; fine for scoring a few
// dozen CHAMP domains, prune or parallelise if it's ever called on thousands of pairs at 1e6 vertices
fn expected_mi(t: &Table) -> f64 {
    let n = t.n as usize;
    let mut lf = vec![0.0f64; n + 1]; // ln k!
    for k in 1..=n {
        lf[k] = lf[k - 1] + (k as f64).ln();
    }
    let nf = n as f64;
    let mut e = 0.0;
    for &ai in &t.rows {
        for &bj in &t.cols {
            let (ai, bj) = (ai as usize, bj as usize);
            let base = lf[ai] + lf[bj] + lf[n - ai] + lf[n - bj] - lf[n];
            for nij in (ai + bj).saturating_sub(n).max(1)..=ai.min(bj) {
                let p = (base - lf[nij] - lf[ai - nij] - lf[bj - nij] - lf[n + nij - ai - bj]).exp();
                e += nij as f64 / nf * (nf * nij as f64 / (ai as f64 * bj as f64)).ln() * p;
            }
        }
    }
    e
}

/// Element-centric similarity of each vertex between `a` and `b` (same length, any labels), in [0, 1]. For hard
/// partitions the personalised-PageRank definition collapses to |A_i ∩ B_i| / max(|A_i|, |B_i|), A_i and B_i
/// being the vertex's communities in `a` and `b`, for any restart probability (checked against the PageRank form
/// to 1e-15). The mean is the global ECS; the per-vertex values show which vertices move, which NVI/AMI can't.
pub fn ecs(a: &[u32], b: &[u32]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "partitions of different sizes");
    let (mut sa, mut sb, mut sab) = (HashMap::new(), HashMap::new(), HashMap::new());
    for (&x, &y) in a.iter().zip(b) {
        *sa.entry(x).or_insert(0u64) += 1;
        *sb.entry(y).or_insert(0u64) += 1;
        *sab.entry((x, y)).or_insert(0u64) += 1;
    }
    a.iter().zip(b).map(|(&x, &y)| sab[&(x, y)] as f64 / sa[&x].max(sb[&y]) as f64).collect()
}

/// Which agreement measure CHAMP's neighbour and seed scores use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Metric {
    /// `nvi`: a distance, 0 = identical.
    #[default]
    Nvi,
    /// `ami`: a similarity, 1 = identical.
    Ami,
}

impl Metric {
    pub fn eval(self, a: &[u32], b: &[u32]) -> f64 {
        match self {
            Metric::Nvi => nvi(a, b),
            Metric::Ami => ami(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_up_to_relabelling() {
        let (a, relabelled) = ([0, 0, 0, 1, 1, 1], [7, 7, 7, 3, 3, 3]);
        assert_eq!(nvi(&a, &relabelled), 0.0);
        assert!((ami(&a, &relabelled) - 1.0).abs() < 1e-12);
        assert_eq!(nvi(&[0; 4], &[5; 4]), 0.0);
        assert_eq!(ami(&[0; 4], &[5; 4]), 1.0);
        assert!(ecs(&a, &relabelled).iter().all(|&s| s == 1.0));
    }

    #[test]
    fn ecs_is_overlap_over_larger_community() {
        // a: {0,1,2,3} {4,5}; b: {0,1} {2,3,4,5}
        let (a, b) = ([0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1]);
        assert_eq!(ecs(&a, &b), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let b = [0, 0, 0, 1, 1, 1];
        assert_eq!(ecs(&a, &b), [0.75, 0.75, 0.75, 0.25, 2.0 / 3.0, 2.0 / 3.0]);
    }

    #[test]
    fn matches_sklearn_and_pygenstability() {
        // sklearn 1.9.1 adjusted_mutual_info_score(average_method="max"); NVI as PyGenStability evaluate_NVI
        let (x, y) = ([0, 0, 1, 1, 2, 2, 2, 0], [0, 0, 1, 1, 1, 2, 2, 2]);
        assert!((ami(&x, &y) - 0.31967265056964705).abs() < 1e-12, "{}", ami(&x, &y));
        assert!((nvi(&x, &y) - 0.612197222702993).abs() < 1e-12, "{}", nvi(&x, &y));
        let a: Vec<u32> = (0..200).map(|i| (i * 7 + 3) % 11).collect();
        let b: Vec<u32> = (0..200).map(|i| (i * i) % 13).collect();
        assert!((ami(&a, &b) - -0.06206420254915781).abs() < 1e-12, "{}", ami(&a, &b));
        assert!((nvi(&a, &b) - 0.9925362075663795).abs() < 1e-12, "{}", nvi(&a, &b));
        assert!((nvi(&a, &b) - nvi(&b, &a)).abs() < 1e-15 && (ami(&a, &b) - ami(&b, &a)).abs() < 1e-12);
    }
}
