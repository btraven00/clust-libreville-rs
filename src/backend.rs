//! Kernels behind a backend; the plan (level loop, fixed point, canonical labels: lib.rs) is
//! shared by every backend. See docs/road-to-cubecl.md (untracked) and PLAN.md phase 2.
//!
//! Contract: every kernel's output is a pure function of its inputs (graph, labels, options,
//! seed, step), never of thread count or scheduling. The thread-invariance and level-hash tests
//! check it at this boundary. Labels live on the host; a backend keeps graphs wherever its
//! kernels run (`Graph`).

use std::borrow::Cow;

use crate::graph::{Csr, Weight};
use crate::{aggregate, moving, refine};

pub(crate) trait Backend: Sync {
    /// A level's graph, resident where the kernels run. Level 0 borrows the input; coarse
    /// levels are owned (`'static`) and always f64.
    type Graph<'a, W: Weight>;

    fn upload<'a, W: Weight>(&self, g: &'a Csr<W>) -> Self::Graph<'a, W>;
    fn n<W: Weight>(&self, g: &Self::Graph<'_, W>) -> usize;
    /// Edge slots (for the trace).
    fn edges<W: Weight>(&self, g: &Self::Graph<'_, W>) -> usize;
    /// Structure and exact weight bits (for the determinism tests and the trace).
    fn hash<W: Weight>(&self, g: &Self::Graph<'_, W>) -> u64;
    /// Colour, then local moving on `comm` (ids < n). Returns (colours, rounds).
    fn move_nodes<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &mut [u32], gamma: f64, tol: f64, seed: u64, step: u64) -> (usize, usize);
    /// Refined labels (dense) and their count.
    fn refine<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &[u32], gamma: f64, theta: f64, seed: u64, step: u64) -> (Vec<u32>, usize);
    /// The graph of `labels` (dense, `n_labels` of them).
    fn aggregate<W: Weight>(&self, g: &Self::Graph<'_, W>, labels: &[u32], n_labels: usize) -> Self::Graph<'static, f64>;
}

/// Today's rayon kernels.
pub(crate) struct Cpu;

impl Backend for Cpu {
    type Graph<'a, W: Weight> = Cow<'a, Csr<W>>;

    fn upload<'a, W: Weight>(&self, g: &'a Csr<W>) -> Self::Graph<'a, W> {
        Cow::Borrowed(g)
    }
    fn n<W: Weight>(&self, g: &Self::Graph<'_, W>) -> usize {
        g.n()
    }
    fn edges<W: Weight>(&self, g: &Self::Graph<'_, W>) -> usize {
        g.targets.len()
    }
    fn hash<W: Weight>(&self, g: &Self::Graph<'_, W>) -> u64 {
        g.hash()
    }
    fn move_nodes<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &mut [u32], gamma: f64, tol: f64, seed: u64, step: u64) -> (usize, usize) {
        let groups = moving::colouring(g, seed, step);
        moving::move_nodes(g, comm, gamma, tol, &groups)
    }
    fn refine<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &[u32], gamma: f64, theta: f64, seed: u64, step: u64) -> (Vec<u32>, usize) {
        refine::refine(g, comm, gamma, theta, seed, step)
    }
    fn aggregate<W: Weight>(&self, g: &Self::Graph<'_, W>, labels: &[u32], n_labels: usize) -> Self::Graph<'static, f64> {
        Cow::Owned(aggregate::aggregate(g, labels, n_labels))
    }
}
