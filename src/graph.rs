//! Undirected weighted graph as a symmetric CSR.
//!
//! Convention: the graph is its symmetric adjacency matrix A. Row v lists every neighbour u with
//! A[v][u]. A self-loop is one entry A[v][v] in its row. k_v = sum of row v, 2m = sum of all k_v.
//! Coarse graphs follow the same rule: the self-loop of a coarse node holds the full double sum
//! of its internal weights, so k_c = sum of its members' k_v.

pub struct Graph {
    pub(crate) offsets: Vec<usize>,
    pub(crate) targets: Vec<u32>,
    pub(crate) weights: Vec<f64>,
    pub(crate) strength: Vec<f64>,
    pub(crate) total: f64, // 2m
}

impl Graph {
    /// A graph from a full symmetric CSR (both directions of every edge, e.g. scanpy's
    /// connectivities). Validated: offsets, bounds, finite non-negative weights, symmetry.
    pub fn from_csr(indptr: &[usize], indices: &[u32], weights: &[f64]) -> Result<Graph, String> {
        let n = indptr.len().checked_sub(1).ok_or("indptr is empty")?;
        if n > u32::MAX as usize {
            return Err(format!("{n} vertices, more than u32 ids"));
        }
        if indptr[0] != 0 || indptr[n] != indices.len() || indices.len() != weights.len() {
            return Err("indptr, indices and data lengths disagree".into());
        }
        if indptr.windows(2).any(|w| w[0] > w[1]) {
            return Err("indptr is not monotone".into());
        }
        if indices.iter().any(|&u| u as usize >= n) {
            return Err("index out of range".into());
        }
        if weights.iter().any(|&w| !(w.is_finite() && w >= 0.0)) {
            return Err("weights must be finite and non-negative".into());
        }
        let g = Graph::from_parts(indptr.to_vec(), indices.to_vec(), weights.to_vec());
        if !g.is_symmetric() {
            return Err("matrix is not symmetric".into());
        }
        Ok(g)
    }

    /// No validation: for graphs we build ourselves (aggregation, tests).
    pub(crate) fn from_parts(offsets: Vec<usize>, targets: Vec<u32>, weights: Vec<f64>) -> Graph {
        let n = offsets.len() - 1;
        let strength: Vec<f64> = (0..n).map(|v| weights[offsets[v]..offsets[v + 1]].iter().sum()).collect();
        let total = strength.iter().sum();
        Graph { offsets, targets, weights, strength, total }
    }

    pub fn n(&self) -> usize {
        self.offsets.len() - 1
    }

    pub(crate) fn row(&self, v: usize) -> (&[u32], &[f64]) {
        let r = self.offsets[v]..self.offsets[v + 1];
        (&self.targets[r.clone()], &self.weights[r])
    }

    /// FNV-1a over the structure and the exact weight bits: equal hashes, identical graphs.
    pub(crate) fn hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |x: u64| {
            for b in x.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
            }
        };
        self.offsets.iter().for_each(|&o| eat(o as u64));
        self.targets.iter().for_each(|&t| eat(t as u64));
        self.weights.iter().for_each(|&w| eat(w.to_bits()));
        h
    }

    fn is_symmetric(&self) -> bool {
        // every row sorted by target (with weight bits as tiebreak) must equal the transpose's
        let n = self.n();
        let mut fwd: Vec<(u32, u32, u64)> = Vec::with_capacity(self.targets.len());
        for v in 0..n {
            let (t, w) = self.row(v);
            fwd.extend(t.iter().zip(w).map(|(&u, &w)| (v as u32, u, w.to_bits())));
        }
        let mut rev: Vec<(u32, u32, u64)> = fwd.iter().map(|&(v, u, w)| (u, v, w)).collect();
        fwd.sort_unstable();
        rev.sort_unstable();
        fwd == rev
    }
}
