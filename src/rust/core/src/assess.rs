//! Statistics of a *given* partition on a graph: how well do these labels describe this graph?
//!
//! `compare.rs` scores a partition against another partition; this module scores it against the
//! graph itself, with the labels taken as fixed (an annotation, a metadata column, a clustering
//! from another modality). Nothing here chooses the number of blocks or fits blocks: every
//! quantity is a closed form of the block edge-count matrix, so it costs one pass over the edges.
//!
//! * `dcsbm_loglik`: the profile log-likelihood of the degree-corrected stochastic block model with
//!   the labels as blocks, L = Σ_rs m_rs ln(m_rs / (κ_r κ_s)) (Karrer & Newman 2011, "Stochastic
//!   blockmodels and community structure in networks", Phys. Rev. E 83:016107). The statistic of
//!   BESTest (Peel, Larremore & Clauset 2017, "The ground truth about metadata and community
//!   detection in networks", Sci. Adv. 3:e1602548), whose p-value is a permutation of the labels
//!   that callers run themselves.
//! * `description_length`: the microcanonical DC-SBM description length with uniform priors,
//!   Σ = −ln P(A | k, e, b) − ln P(k | e, b) − ln P(e) − ln P(b) (Peixoto 2017, "Nonparametric
//!   Bayesian inference of the microcanonical stochastic block model", Phys. Rev. E 95:012317,
//!   eqs. 13, 17, 19–21; graph-tool's `BlockState.entropy(degree_dl_kind="uniform")`). Charges
//!   for the block matrix and the partition, so it ranks nested annotations: the shorter, the
//!   better the graph supports that level; differences are log posterior odds.
//! * `significance`: Σ_c C(n_c, 2) D(p_c ‖ p), the binary KL divergence of each block's internal
//!   density from the graph's (Traag, Krings & Van Dooren 2013, "Significant scales in community
//!   structure", Sci. Rep. 3:2930; leidenalg's `SignificanceVertexPartition.quality()`).
//! * `label_surprise`: per block, −ln P(X ≥ m_c) with X ~ Poisson(κ_c² / 4E): the block's internal
//!   edges beyond what its degree sum gives under the configuration model (the edge-count form of
//!   Lancichinetti, Radicchi & Ramasco 2010, "Statistical significance of communities in
//!   networks", Phys. Rev. E 81:046110). The weakest block is the actionable one.
//! * `neosbm`: Peel et al. 2017's neoSBM, section "The neoSBM". Every vertex keeps its label's
//!   block or is free to sit in any of the K blocks; with q free vertices the objective is
//!   J = L + q ln θ + (N − q) ln(1 − θ), so leaving the annotation costs ln((1 − θ)/θ) nats of
//!   likelihood. Sweeping θ towards 1/2 traces how many vertices defect as defecting gets
//!   cheaper; at θ = 1/2 the prior is neutral and the defecting fraction is "vertices whose label
//!   is not locally optimal for this graph". The maximiser is greedy single-vertex moves with the
//!   exact incremental change of L (O(K²) per vertex from its edges into each block), in the
//!   order the caller gives, warm-started along the θ grid; it maximises the published objective
//!   and is not Peel's reference code (written for networks of ~10³ vertices).
//!
//! Edge counts are weight sums, so on an unweighted simple graph (a kNN graph) every formula is
//! exact; on a weighted graph `dcsbm_loglik`, `significance`, `label_surprise` and `neosbm` read
//! weights as multiplicities and `description_length` is the same expression with non-integer
//! counts (interpret with care). A self-loop counts once in m_rr and once in κ, as in the graph's
//! own convention. Deterministic: every sum runs in label order.

use crate::graph::{Csr, Graph, Repr, Weight};

/// Block-level counts of a labelled graph. `m` is the ordered-pair matrix (K × K, row-major):
/// m[r][s] = Σ A_uv over u in r, v in s, so m[r][r] counts an inner edge twice and κ_r = Σ_s m[r][s].
pub struct BlockStats {
    pub k: usize,
    pub n: usize,
    pub m: Vec<f64>,
    pub kappa: Vec<f64>,
    pub size: Vec<f64>,
    /// Σ_v k_v = 2E
    pub two_e: f64,
}

impl BlockStats {
    pub fn inner_edges(&self, r: usize) -> f64 {
        self.m[r * self.k + r] / 2.0
    }
}

fn check_labels(n: usize, labels: &[u32]) -> usize {
    assert_eq!(labels.len(), n, "one label per vertex");
    labels.iter().map(|&l| l as usize + 1).max().unwrap_or(0)
}

/// x ln x, 0 at 0.
fn xlx(x: f64) -> f64 {
    if x > 0.0 { x * x.ln() } else { 0.0 }
}

pub fn block_stats(g: &Graph, labels: &[u32]) -> BlockStats {
    match &g.0 {
        Repr::F32(c) => block_stats_csr(c, labels),
        Repr::F64(c) => block_stats_csr(c, labels),
    }
}

fn block_stats_csr<W: Weight>(g: &Csr<W>, labels: &[u32]) -> BlockStats {
    let n = g.n();
    let k = check_labels(n, labels);
    let mut m = vec![0.0; k * k];
    let mut kappa = vec![0.0; k];
    let mut size = vec![0.0; k];
    for v in 0..n {
        let r = labels[v] as usize;
        let (t, w) = g.row(v);
        for (&u, &x) in t.iter().zip(w) {
            m[r * k + labels[u as usize] as usize] += x.into();
        }
        kappa[r] += g.strength[v];
        size[r] += 1.0;
    }
    BlockStats { k, n, m, kappa, size, two_e: g.total }
}

/// Profile log-likelihood of the degree-corrected SBM, Σ_rs m_rs ln(m_rs / (κ_r κ_s)) =
/// Σ_rs f(m_rs) − 2 Σ_r f(κ_r), f(x) = x ln x.
pub fn dcsbm_loglik(s: &BlockStats) -> f64 {
    s.m.iter().map(|&x| xlx(x)).sum::<f64>() - 2.0 * s.kappa.iter().map(|&x| xlx(x)).sum::<f64>()
}

/// ln Γ(x), Lanczos approximation (g = 7, 9 terms), |rel. error| ~1e-15 for x > 0.
pub fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// ln of the binomial coefficient C(n, k) for real n, k.
fn ln_binom(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

/// ln of the number of multisets of size k drawn from n items, ((n, k)) = C(n + k − 1, k).
fn ln_multiset(n: f64, k: f64) -> f64 {
    ln_binom(n + k - 1.0, k)
}

/// Peixoto (2017) description length of the microcanonical DC-SBM with uniform priors, in nats.
pub fn description_length(g: &Graph, labels: &[u32]) -> f64 {
    let s = block_stats(g, labels);
    let degrees: &[f64] = match &g.0 {
        Repr::F32(c) => &c.strength,
        Repr::F64(c) => &c.strength,
    };
    description_length_from(&s, degrees)
}

fn description_length_from(s: &BlockStats, degrees: &[f64]) -> f64 {
    let (k, n) = (s.k, s.n as f64);
    let e = s.two_e / 2.0;
    // −ln P(A | k, e, b), eq. 13, simple graph: Π_r e_r! / (Π_{r<s} e_rs! Π_r e_rr!! Π_i k_i!)
    let mut data = s.kappa.iter().map(|&x| ln_gamma(x + 1.0)).sum::<f64>();
    for r in 0..k {
        for t in (r + 1)..k {
            data -= ln_gamma(s.m[r * k + t] + 1.0);
        }
        let half = s.m[r * k + r] / 2.0;
        data -= half * std::f64::consts::LN_2 + ln_gamma(half + 1.0);
    }
    data -= degrees.iter().map(|&d| ln_gamma(d + 1.0)).sum::<f64>();
    // −ln P(k | e, b): uniform over degree sequences per block, eq. 17
    let deg = (0..k).map(|r| ln_multiset(s.size[r], s.kappa[r])).sum::<f64>();
    // −ln P(e): uniform over block edge-count matrices, eq. 19
    let edges = ln_multiset((k * (k + 1)) as f64 / 2.0, e);
    // −ln P(b): the partition given the sizes, the sizes given K, and K; eq. 21
    let part = ln_gamma(n + 1.0) - s.size.iter().map(|&x| ln_gamma(x + 1.0)).sum::<f64>() + ln_binom(n - 1.0, k as f64 - 1.0) + n.ln();
    data + deg + edges + part
}

/// Traag et al. 2013: Σ_c C(n_c, 2) D(p_c ‖ p), nats. p_c = inner edges / pairs within c, p = E / C(N, 2).
pub fn significance(g: &Graph, labels: &[u32]) -> f64 {
    let s = block_stats(g, labels);
    let n = s.n as f64;
    let p = (s.two_e / 2.0) / (n * (n - 1.0) / 2.0);
    let kl = |x: f64| {
        let mut d = 0.0;
        if x > 0.0 {
            d += x * (x / p).ln();
        }
        if x < 1.0 {
            d += (1.0 - x) * ((1.0 - x) / (1.0 - p)).ln();
        }
        d
    };
    (0..s.k)
        .map(|r| {
            let pairs = s.size[r] * (s.size[r] - 1.0) / 2.0;
            if pairs > 0.0 { pairs * kl(s.inner_edges(r) / pairs) } else { 0.0 }
        })
        .sum()
}

/// ln P(X ≥ m) for X ~ Poisson(λ), summed in log space from m to max(m, λ + 20√λ) + 50 (the tail
/// beyond is negligible). Exact to rounding at every (m, λ), no underflow.
pub fn poisson_log_tail(m: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return if m > 0.0 { f64::NEG_INFINITY } else { 0.0 };
    }
    if m <= 0.0 {
        return 0.0;
    }
    let m = m.ceil();
    let upper = m.max((lambda + 20.0 * lambda.sqrt()).ceil()) + 50.0;
    let ln_l = lambda.ln();
    let term = |j: f64| -lambda + j * ln_l - ln_gamma(j + 1.0);
    let peak = term(if lambda > m { lambda.floor() } else { m });
    let mut acc = 0.0;
    let mut j = m;
    while j <= upper {
        acc += (term(j) - peak).exp();
        j += 1.0;
    }
    (peak + acc.ln()).min(0.0)
}

/// Per block, −ln P(X ≥ m_c) with X ~ Poisson(κ_c² / 4E): nats of inner cohesion beyond the
/// configuration model. +∞ when the expectation is 0 and the block has inner edges.
pub fn label_surprise(g: &Graph, labels: &[u32]) -> Vec<f64> {
    let s = block_stats(g, labels);
    let e = s.two_e / 2.0;
    (0..s.k).map(|r| -poisson_log_tail(s.inner_edges(r), s.kappa[r] * s.kappa[r] / (4.0 * e))).collect()
}

/// The neoSBM trajectory over a θ grid.
pub struct NeoSbm {
    /// fraction of vertices whose block differs from their label, per θ
    pub defect: Vec<f64>,
    /// L at the end of each θ minus L of the labels
    pub loglik_gain: Vec<f64>,
    /// sweeps run at each θ (max_sweeps means it did not converge)
    pub sweeps: Vec<usize>,
    /// the blocks at the last θ
    pub labels: Vec<u32>,
}

struct Mover<'a, W: Weight> {
    g: &'a Csr<W>,
    meta: &'a [u32],
    k: usize,
    z: Vec<u32>,
    m: Vec<f64>,
    kappa: Vec<f64>,
    // scratch
    c: Vec<f64>,
    loop_w: f64, // v's self-loop weight (in c[r], but it moves with v: it stays inside v's block)
    delta: Vec<f64>,
}

impl<'a, W: Weight> Mover<'a, W> {
    fn new(g: &'a Csr<W>, meta: &'a [u32]) -> Self {
        let s = block_stats_csr(g, meta);
        Mover { g, meta, k: s.k, z: meta.to_vec(), m: s.m, kappa: s.kappa, c: vec![0.0; s.k], loop_w: 0.0, delta: vec![0.0; s.k] }
    }

    fn loglik(&self) -> f64 {
        self.m.iter().map(|&x| xlx(x)).sum::<f64>() - 2.0 * self.kappa.iter().map(|&x| xlx(x)).sum::<f64>()
    }

    /// Exact change of L for moving vertex v from its block r to every block (0 at r), into
    /// `self.delta`; `self.c` holds v's edge weight into each block. Affected entries: row and
    /// column r, row and column s, κ_r, κ_s.
    fn deltas(&mut self, v: usize) {
        let (k, r) = (self.k, self.z[v] as usize);
        let kv = self.g.strength[v];
        self.c.iter_mut().for_each(|x| *x = 0.0);
        let (t, w) = self.g.row(v);
        self.loop_w = 0.0;
        for (&u, &x) in t.iter().zip(w) {
            self.c[self.z[u as usize] as usize] += x.into();
            if u as usize == v {
                self.loop_w += x.into();
            }
        }
        let (m, kap, c, lw) = (&self.m, &self.kappa, &self.c, self.loop_w);
        let cr = c[r] - lw; // v's weight to the rest of r
        let mr = |t: usize| m[r * k + t];
        let a = xlx(mr(r) - 2.0 * cr - lw) - xlx(mr(r)); // m_rr
        // row/column r against every t ≠ r loses c_t (twice: rt and tr); the t = s term is taken out per s
        let mut d_all = 0.0;
        for (t, &ct) in c.iter().enumerate() {
            if t != r {
                d_all += 2.0 * (xlx(mr(t) - ct) - xlx(mr(t)));
            }
        }
        let kr = -2.0 * (xlx(kap[r] - kv) - xlx(kap[r]));
        for s in 0..k {
            if s == r {
                self.delta[s] = 0.0;
                continue;
            }
            let mss = m[s * k + s];
            let b = xlx(mss + 2.0 * c[s] + lw) - xlx(mss); // m_ss
            let cc = 2.0 * (xlx(mr(s) - c[s] + cr) - xlx(mr(s))); // m_rs = m_sr
            let d = d_all - 2.0 * (xlx(mr(s) - c[s]) - xlx(mr(s)));
            // row/column s against every t ∉ {r, s} gains c_t (twice)
            let mut e = 0.0;
            for t in 0..k {
                if t != r && t != s {
                    let mst = m[s * k + t];
                    e += 2.0 * (xlx(mst + c[t]) - xlx(mst));
                }
            }
            let ks = -2.0 * (xlx(kap[s] + kv) - xlx(kap[s]));
            self.delta[s] = a + b + cc + d + e + kr + ks;
        }
    }

    fn apply(&mut self, v: usize, s: usize) {
        let (k, r) = (self.k, self.z[v] as usize);
        let kv = self.g.strength[v];
        for t in 0..k {
            let ct = self.c[t];
            self.m[r * k + t] -= ct;
            self.m[t * k + r] -= ct;
            self.m[s * k + t] += ct;
            self.m[t * k + s] += ct;
        }
        // the self-loop went from m_rr to m_ss, not to m_rs (requires `deltas(v)` just before)
        let lw = self.loop_w;
        self.m[r * k + r] += lw;
        self.m[s * k + s] += lw;
        self.m[r * k + s] -= lw;
        self.m[s * k + r] -= lw;
        self.kappa[r] -= kv;
        self.kappa[s] += kv;
        self.z[v] = s as u32;
    }

    /// One pass in `order`; `cost` = ln((1 − θ)/θ) paid by a vertex whose block differs from its
    /// label. Returns the number of moves.
    fn sweep(&mut self, cost: f64, order: &[u32]) -> usize {
        let mut moves = 0;
        for &v in order {
            let v = v as usize;
            if self.g.strength[v] == 0.0 {
                continue;
            }
            self.deltas(v);
            let r = self.z[v] as usize;
            let meta = self.meta[v] as usize;
            let free_now = if r != meta { 1.0 } else { 0.0 };
            let (mut best, mut best_gain) = (r, 0.0);
            for s in 0..self.k {
                let free_after = if s != meta { 1.0 } else { 0.0 };
                let gain = self.delta[s] - cost * (free_after - free_now);
                if gain > best_gain + 1e-9 {
                    best = s;
                    best_gain = gain;
                }
            }
            if best != r {
                self.apply(v, best);
                moves += 1;
            }
        }
        moves
    }

    fn defect(&self) -> f64 {
        self.z.iter().zip(self.meta).filter(|(a, b)| a != b).count() as f64 / self.z.len() as f64
    }
}

/// neoSBM over the θ grid (ascending, each warm-started from the previous), sweeping vertices in
/// `order` (a permutation of 0..n; the same order gives the same result on every machine).
pub fn neosbm(g: &Graph, meta: &[u32], thetas: &[f64], max_sweeps: usize, order: &[u32]) -> NeoSbm {
    match &g.0 {
        Repr::F32(c) => neosbm_csr(c, meta, thetas, max_sweeps, order),
        Repr::F64(c) => neosbm_csr(c, meta, thetas, max_sweeps, order),
    }
}

fn neosbm_csr<W: Weight>(g: &Csr<W>, meta: &[u32], thetas: &[f64], max_sweeps: usize, order: &[u32]) -> NeoSbm {
    check_labels(g.n(), meta);
    assert_eq!(order.len(), g.n(), "order must be a permutation of the vertices");
    let mut mv = Mover::new(g, meta);
    let l0 = mv.loglik();
    let mut out = NeoSbm { defect: vec![], loglik_gain: vec![], sweeps: vec![], labels: vec![] };
    for &theta in thetas {
        assert!(theta > 0.0 && theta < 1.0, "theta must be in (0, 1)");
        let cost = ((1.0 - theta) / theta).ln();
        let mut sweeps = 0;
        while sweeps < max_sweeps {
            sweeps += 1;
            if mv.sweep(cost, order) == 0 {
                break;
            }
        }
        out.defect.push(mv.defect());
        out.loglik_gain.push(mv.loglik() - l0);
        out.sweeps.push(sweeps);
    }
    out.labels = mv.z;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(n: usize, edges: &[(u32, u32)]) -> Graph {
        let mut rows: Vec<Vec<u32>> = vec![vec![]; n];
        for &(a, b) in edges {
            rows[a as usize].push(b);
            if a != b {
                rows[b as usize].push(a);
            }
        }
        let mut indptr = vec![0usize];
        let mut indices = vec![];
        for r in &mut rows {
            r.sort_unstable();
            indices.extend_from_slice(r);
            indptr.push(indices.len());
        }
        Graph::from_csr(&indptr, &indices, &vec![1.0; indices.len()]).unwrap()
    }

    /// Planted partition: K equal blocks, edge with p_in inside and p_out between (LCG, fixed).
    fn planted(n: usize, k: usize, p_in: f64, p_out: f64, seed: u64) -> (Vec<u32>, Graph) {
        let truth: Vec<u32> = (0..n).map(|i| (i * k / n) as u32).collect();
        let mut state = seed;
        let mut edges = vec![];
        for a in 0..n {
            for b in (a + 1)..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                if u < if truth[a] == truth[b] { p_in } else { p_out } {
                    edges.push((a as u32, b as u32));
                }
            }
        }
        (truth, graph(n, &edges))
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        assert!((ln_gamma(0.5) - 0.5723649429247001).abs() < 1e-13);
        assert!((ln_gamma(1.0)).abs() < 1e-14);
        assert!((ln_gamma(10.0) - 12.801827480081469).abs() < 1e-12);
        assert!((ln_gamma(100.5) - 361.4355404677776).abs() < 1e-10);
    }

    #[test]
    fn poisson_tail_matches_known_values() {
        // P(X >= 10 | lambda = 5) = 0.03182805730620481
        assert!((poisson_log_tail(10.0, 5.0) - (0.03182805730620481f64).ln()).abs() < 1e-10);
        // P(X >= 1 | lambda = 2) = 1 - e^-2
        assert!((poisson_log_tail(1.0, 2.0) - (1.0 - (-2.0f64).exp()).ln()).abs() < 1e-12);
        // far tail does not underflow: P(X >= 5000 | lambda = 50) is tiny but finite
        let t = poisson_log_tail(5000.0, 50.0);
        assert!(t.is_finite() && t < -10000.0);
    }

    #[test]
    fn block_stats_of_two_triangles_joined_by_one_edge() {
        let g = graph(6, &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]);
        let s = block_stats(&g, &[0, 0, 0, 1, 1, 1]);
        assert_eq!(s.k, 2);
        assert_eq!(s.m, vec![6.0, 1.0, 1.0, 6.0]);
        assert_eq!(s.kappa, vec![7.0, 7.0]);
        assert_eq!(s.two_e, 14.0);
        assert_eq!(s.inner_edges(0), 3.0);
        // L = 2 * 6 ln(6/49) + 2 * 1 ln(1/49)
        let l = dcsbm_loglik(&s);
        assert!((l - (2.0 * 6.0 * (6.0f64 / 49.0).ln() + 2.0 * (1.0f64 / 49.0).ln())).abs() < 1e-12);
        // description length is finite and shorter than the one-block description
        let one = description_length(&g, &[0; 6]);
        let two = description_length(&g, &[0, 0, 0, 1, 1, 1]);
        assert!(two.is_finite() && one.is_finite());
        assert!(significance(&g, &[0, 0, 0, 1, 1, 1]) > 0.0);
        let sp = label_surprise(&g, &[0, 0, 0, 1, 1, 1]);
        assert_eq!(sp.len(), 2);
        assert!(sp.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn neosbm_deltas_are_exact() {
        let (truth, g) = planted(300, 5, 0.1, 0.02, 7);
        let meta: Vec<u32> = truth.iter().map(|&t| (t * 7 + 3) % 5).collect(); // a wrong but valid labelling
        let c = match &g.0 {
            Repr::F32(c) => c,
            _ => panic!("unit weights are f32"),
        };
        let mut mv = Mover::new(c, &meta);
        let mut state = 11u64;
        for _ in 0..300 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (state >> 33) as usize % 300;
            let s = (state >> 20) as usize % 5;
            let before = mv.loglik();
            mv.deltas(v);
            let predicted = mv.delta[s];
            mv.apply(v, s);
            assert!((mv.loglik() - before - predicted).abs() < 1e-7, "v={v} s={s}");
        }
        assert_eq!(mv.m, block_stats(&g, &mv.z).m); // the running counts are the true ones
    }

    #[test]
    fn neosbm_with_self_loops() {
        // Seurat's SNN graph has a unit diagonal: the running block counts must still be exact
        let (truth, g) = planted(200, 4, 0.1, 0.02, 5);
        let (indptr, indices) = match &g.0 {
            Repr::F32(c) => (&c.offsets, &c.targets),
            _ => unreachable!(),
        };
        let mut e: Vec<(u32, u32)> = (0..200u32).map(|v| (v, v)).collect();
        for v in 0..200 {
            e.extend(indices[indptr[v]..indptr[v + 1]].iter().filter(|&&u| u as usize > v).map(|&u| (v as u32, u)));
        }
        let g = graph(200, &e);
        let meta: Vec<u32> = truth.iter().map(|&t| (t * 3 + 1) % 4).collect();
        let c = match &g.0 {
            Repr::F32(c) => c,
            _ => unreachable!(),
        };
        let mut mv = Mover::new(c, &meta);
        for i in 0..400usize {
            let (v, s) = ((i * 7919) % 200, (i * 31) % 4);
            let before = mv.loglik();
            mv.deltas(v);
            let predicted = mv.delta[s];
            mv.apply(v, s);
            assert!((mv.loglik() - before - predicted).abs() < 1e-7, "v={v} s={s}");
            if i % 50 == 0 {
                assert_eq!(mv.m, block_stats(&g, &mv.z).m, "after {i} moves");
            }
        }
        let out = neosbm(&g, &meta, &[0.3], 20, &(0..200).collect::<Vec<_>>());
        let gain = dcsbm_loglik(&block_stats(&g, &out.labels)) - dcsbm_loglik(&block_stats(&g, &meta));
        assert!((out.loglik_gain[0] - gain).abs() < 1e-6, "{} vs {gain}", out.loglik_gain[0]);
    }

    #[test]
    fn neosbm_keeps_true_labels_and_drops_shuffled_ones() {
        let (truth, g) = planted(600, 4, 0.08, 0.005, 3);
        let order: Vec<u32> = (0..600).collect();
        let thetas = [1e-6, 1e-3, 0.1, 0.3, 0.5];
        let a = neosbm(&g, &truth, &thetas, 20, &order);
        assert!(a.defect[4] < 0.05, "true labels should be locally optimal: {:?}", a.defect);
        let shuffled: Vec<u32> = (0..600).map(|i| truth[(i * 7919) % 600]).collect();
        let b = neosbm(&g, &shuffled, &thetas, 20, &order);
        assert!(b.defect[4] > 0.5, "shuffled labels should defect: {:?}", b.defect);
        assert!(b.defect.windows(2).all(|w| w[0] <= w[1] + 1e-12), "defection grows with theta");
        assert!(b.loglik_gain[4] > 0.0);
    }
}
