//! Undirected weighted graph as a symmetric CSR.
//!
//! Convention: the graph is its symmetric adjacency matrix A. Row v lists every neighbour u with
//! A[v][u]. A self-loop is one entry A[v][v] in its row. k_v = sum of row v, 2m = sum of all k_v.
//! Coarse graphs follow the same rule: the self-loop of a coarse node holds the full double sum
//! of its internal weights, so k_c = sum of its members' k_v.
//!
//! Weights are stored as f32 when every input weight is exactly an f32 (scanpy's connectivities
//! are), else f64; coarse graphs are always f64. All arithmetic is f64: f32 -> f64 is exact, so
//! the storage type changes memory traffic (8 instead of 12 bytes per edge), never a result.

use rayon::prelude::*;

pub(crate) trait Weight: Copy + Send + Sync + Into<f64> + 'static {
    /// The weights as f32, when that is their storage type (for GPU upload).
    #[cfg_attr(not(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-metal")), allow(dead_code))]
    fn as_f32(w: &[Self]) -> Option<&[f32]>;
}
impl Weight for f32 {
    fn as_f32(w: &[f32]) -> Option<&[f32]> {
        Some(w)
    }
}
impl Weight for f64 {
    fn as_f32(_: &[f64]) -> Option<&[f32]> {
        None
    }
}

#[derive(Clone)] // for Cow; never cloned in practice (the CPU backend borrows level 0)
pub(crate) struct Csr<W> {
    pub(crate) offsets: Vec<usize>,
    pub(crate) targets: Vec<u32>,
    pub(crate) weights: Vec<W>,
    pub(crate) strength: Vec<f64>,
    pub(crate) total: f64, // 2m
}

pub(crate) enum Repr {
    F32(Csr<f32>),
    F64(Csr<f64>),
}

/// An undirected weighted graph (validated symmetric CSR).
pub struct Graph(pub(crate) Repr);

impl Graph {
    /// A graph from a full symmetric CSR (both directions of every edge, e.g. scanpy's
    /// connectivities). Validated: offsets, bounds, finite non-negative weights, symmetry.
    pub fn from_csr(indptr: &[usize], indices: &[u32], weights: &[f64]) -> Result<Graph, String> {
        Self::build(indptr, indices, weights, true)
    }

    /// As [`Graph::from_csr`], without the symmetry check (O(m), about as long as validating
    /// everything else): for inputs symmetric by construction, such as scanpy's or Seurat's
    /// neighbour graphs. Offsets, bounds and weights are still validated. An asymmetric input
    /// gives meaningless labels, never undefined behaviour.
    pub fn from_csr_trusted(indptr: &[usize], indices: &[u32], weights: &[f64]) -> Result<Graph, String> {
        Self::build(indptr, indices, weights, false)
    }

    fn build(indptr: &[usize], indices: &[u32], weights: &[f64], check_symmetry: bool) -> Result<Graph, String> {
        let n = indptr.len().checked_sub(1).ok_or("indptr is empty")?;
        if n > u32::MAX as usize {
            return Err(format!("{n} vertices, more than u32 ids"));
        }
        if indptr[0] != 0 || indptr[n] != indices.len() || indices.len() != weights.len() {
            return Err("indptr, indices and data lengths disagree".into());
        }
        if indptr.par_windows(2).any(|w| w[0] > w[1]) {
            return Err("indptr is not monotone".into());
        }
        if indices.par_iter().any(|&u| u as usize >= n) {
            return Err("index out of range".into());
        }
        if weights.par_iter().any(|&w| !(w.is_finite() && w >= 0.0)) {
            return Err("weights must be finite and non-negative".into());
        }
        let symmetric;
        let repr = if weights.par_iter().all(|&w| (w as f32) as f64 == w) {
            let g = Csr::from_parts(indptr.to_vec(), indices.to_vec(), weights.par_iter().map(|&w| w as f32).collect());
            symmetric = !check_symmetry || g.is_symmetric();
            Repr::F32(g)
        } else {
            let g = Csr::from_parts(indptr.to_vec(), indices.to_vec(), weights.to_vec());
            symmetric = !check_symmetry || g.is_symmetric();
            Repr::F64(g)
        };
        if !symmetric {
            return Err("matrix is not symmetric".into());
        }
        Ok(Graph(repr))
    }

    pub fn n(&self) -> usize {
        match &self.0 {
            Repr::F32(g) => g.n(),
            Repr::F64(g) => g.n(),
        }
    }
}

impl<W: Weight> Csr<W> {
    /// No validation: for graphs we build ourselves (aggregation, tests).
    pub(crate) fn from_parts(offsets: Vec<usize>, targets: Vec<u32>, weights: Vec<W>) -> Csr<W> {
        let n = offsets.len() - 1;
        // each row summed in order (parallel over rows: the same bits at any thread count)
        let strength: Vec<f64> = (0..n).into_par_iter().map(|v| weights[offsets[v]..offsets[v + 1]].iter().map(|&w| w.into()).sum()).collect();
        let total = strength.iter().sum();
        Csr { offsets, targets, weights, strength, total }
    }

    pub(crate) fn n(&self) -> usize {
        self.offsets.len() - 1
    }

    pub(crate) fn row(&self, v: usize) -> (&[u32], &[W]) {
        let r = self.offsets[v]..self.offsets[v + 1];
        (&self.targets[r.clone()], &self.weights[r])
    }

    /// FNV-1a over the structure and the exact weight bits (as f64, so the storage type doesn't
    /// change the hash): equal hashes, identical graphs.
    pub(crate) fn hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |x: u64| {
            for b in x.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
            }
        };
        self.offsets.iter().for_each(|&o| eat(o as u64));
        self.targets.iter().for_each(|&t| eat(t as u64));
        self.weights.iter().for_each(|&w| eat(Into::<f64>::into(w).to_bits()));
        h
    }

    /// Every row equals the same row of the transpose, as a multiset of (target, weight bits).
    /// Canonical CSR: a parallel mirror lookup per entry. Otherwise the transpose, a counting
    /// sort by target (O(m)) whose rows list sources in increasing order: a row with strictly
    /// increasing targets is compared element-wise, any other row after sorting both sides.
    fn is_symmetric(&self) -> bool {
        let n = self.n();
        // canonical CSR (every row strictly increasing, as scanpy's): each row is a set, so the
        // matrix is symmetric iff every entry (v, u, w) has its mirror (u, v, w), found by binary
        // search: O(m log d), parallel, no allocation
        let canonical = (0..n).into_par_iter().all(|v| self.row(v).0.windows(2).all(|p| p[0] < p[1]));
        if canonical {
            return (0..n).into_par_iter().with_min_len(1024).all(|v| {
                let (t, w) = self.row(v);
                t.iter().zip(w).all(|(&u, &x)| {
                    let (tu, wu) = self.row(u as usize);
                    tu.binary_search(&(v as u32)).is_ok_and(|i| Into::<f64>::into(wu[i]).to_bits() == Into::<f64>::into(x).to_bits())
                })
            });
        }
        let mut start = vec![0usize; n + 1];
        for &u in &self.targets {
            start[u as usize + 1] += 1;
        }
        for v in 0..n {
            start[v + 1] += start[v];
        }
        if (0..n).any(|v| start[v + 1] - start[v] != self.offsets[v + 1] - self.offsets[v]) {
            return false; // in-degree != out-degree
        }
        let (mut src, mut bits) = (vec![0u32; self.targets.len()], vec![0u64; self.targets.len()]);
        let mut fill = start[..n].to_vec();
        for v in 0..n {
            let (t, w) = self.row(v);
            for (&u, &w) in t.iter().zip(w) {
                let p = &mut fill[u as usize];
                (src[*p], bits[*p]) = (v as u32, Into::<f64>::into(w).to_bits());
                *p += 1;
            }
        }
        (0..n).into_par_iter().with_min_len(1024).all(|v| {
            let (t, w) = self.row(v);
            let r = start[v]..start[v + 1];
            let wb = |w: &W| Into::<f64>::into(*w).to_bits();
            if t.windows(2).all(|p| p[0] < p[1]) {
                t == &src[r.clone()] && w.iter().map(wb).eq(bits[r].iter().copied())
            } else {
                let mut a: Vec<(u32, u64)> = t.iter().copied().zip(w.iter().map(wb)).collect();
                let mut b: Vec<(u32, u64)> = src[r.clone()].iter().copied().zip(bits[r].iter().copied()).collect();
                a.sort_unstable();
                b.sort_unstable();
                a == b
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng;

    /// The previous check (sort all triples, both directions): the reference for the O(m) one.
    fn symmetric_by_sorting(g: &Csr<f64>) -> bool {
        let mut fwd: Vec<(u32, u32, u64)> = Vec::new();
        for v in 0..g.n() {
            let (t, w) = g.row(v);
            fwd.extend(t.iter().zip(w).map(|(&u, &w)| (v as u32, u, w.to_bits())));
        }
        let mut rev: Vec<(u32, u32, u64)> = fwd.iter().map(|&(v, u, w)| (u, v, w)).collect();
        fwd.sort_unstable();
        rev.sort_unstable();
        fwd == rev
    }

    /// Random graphs (unsorted rows, duplicate edges, self-loops), symmetric or perturbed by one
    /// entry: the two checks agree on every one.
    #[test]
    fn symmetry_check_agrees_with_sorting() {
        let mut seen = [0usize; 4]; // (general, canonical) x (asymmetric, symmetric)
        for seed in 0..2000u64 {
            let r = |i: u64| rng::unit(rng::key(seed, 77, i, 0));
            let n = 1 + (r(0) * 30.0) as usize;
            let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
            for e in 0..(r(1) * 3.0 * n as f64) as u64 {
                let (u, v) = ((r(10 + 3 * e) * n as f64) as usize, (r(11 + 3 * e) * n as f64) as usize);
                let w = [1.0, 0.5, 0.1 + r(12 + 3 * e)][(r(12 + 3 * e) * 3.0) as usize];
                rows[u].push((v as u32, w));
                if u != v {
                    rows[v].push((u as u32, w));
                }
            }
            // shuffle some rows (unsorted), sort others (canonical)
            for (v, row) in rows.iter_mut().enumerate() {
                if r(5000 + v as u64) < 0.5 {
                    row.sort_by_key(|e| e.0);
                } else {
                    row.reverse();
                }
            }
            let (kind, kv) = (r(9000), r(9001));
            let nonempty: Vec<usize> = (0..n).filter(|&v| !rows[v].is_empty()).collect();
            if kind < 0.5 && !nonempty.is_empty() {
                let v = nonempty[(kv * nonempty.len() as f64) as usize];
                let e = &mut rows[v][0];
                if kind < 0.25 {
                    e.1 = f64::from_bits(e.1.to_bits() + 1); // one weight bit off
                } else {
                    e.0 = ((e.0 as usize + 1) % n) as u32; // one target moved
                }
            }
            let mut offsets = vec![0usize];
            let (mut targets, mut weights) = (Vec::new(), Vec::new());
            for row in &rows {
                targets.extend(row.iter().map(|e| e.0));
                weights.extend(row.iter().map(|e| e.1));
                offsets.push(targets.len());
            }
            let g = Csr::from_parts(offsets, targets, weights);
            let want = symmetric_by_sorting(&g);
            assert_eq!(g.is_symmetric(), want, "seed {seed}");
            seen[want as usize] += 1;
            // the canonical form (rows sorted, duplicate targets summed): the mirror-lookup path
            let mut offsets = vec![0usize];
            let (mut targets, mut weights) = (Vec::new(), Vec::new());
            for row in &rows {
                let mut r = row.clone();
                r.sort_by_key(|e| e.0);
                for (u, w) in r {
                    if targets.len() > *offsets.last().unwrap() && *targets.last().unwrap() == u {
                        *weights.last_mut().unwrap() += w;
                    } else {
                        targets.push(u);
                        weights.push(w);
                    }
                }
                offsets.push(targets.len());
            }
            let c = Csr::from_parts(offsets, targets, weights);
            let want = symmetric_by_sorting(&c);
            assert_eq!(c.is_symmetric(), want, "seed {seed} (canonical)");
            seen[2 + want as usize] += 1;
        }
        assert!(seen.iter().all(|&k| k > 200), "{seen:?}");
    }
}
