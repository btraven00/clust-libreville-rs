//! Kernels behind a backend; the plan (level loop, fixed point, canonical labels: lib.rs) is
//! shared by every backend. See docs/design.md.
//!
//! Contract: every kernel's output is a pure function of its inputs (graph, labels, options,
//! seed, step), never of thread count or scheduling. The thread-invariance and level-hash tests
//! check it at this boundary. Labels live on the host; a backend keeps graphs wherever its
//! kernels run (`Graph`).

use std::borrow::Cow;

use crate::graph::{Csr, Weight};
use crate::{aggregate, moving, refine, Apply};

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
    /// A level's colour groups (moving::colouring), wherever the kernels need them.
    type Colouring;

    /// The colouring of (graph, seed, step): computed once per level, used by local moving and
    /// by OrderFree refinement.
    fn colour<W: Weight>(&self, g: &Self::Graph<'_, W>, seed: u64, step: u64) -> Self::Colouring;
    /// Local moving on `comm` (ids < n) over the colour groups. Returns (colours, rounds).
    fn move_nodes<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &mut [u32], gamma: f64, tol: f64, col: &Self::Colouring, apply: Apply) -> (usize, usize);
    /// Refined labels (dense) and their count.
    #[allow(clippy::too_many_arguments)]
    fn refine<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &[u32], gamma: f64, theta: f64, seed: u64, step: u64, apply: Apply,
                         col: &Self::Colouring) -> (Vec<u32>, usize);
    /// The graph of `labels` (dense, `n_labels` of them).
    fn aggregate<W: Weight>(&self, g: &Self::Graph<'_, W>, labels: &[u32], n_labels: usize, apply: Apply) -> Self::Graph<'static, f64>;
}

/// Today's rayon kernels.
pub(crate) struct Cpu;

impl Backend for Cpu {
    type Graph<'a, W: Weight> = Cow<'a, Csr<W>>;
    type Colouring = Vec<Vec<u32>>;

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
    fn colour<W: Weight>(&self, g: &Self::Graph<'_, W>, seed: u64, step: u64) -> Self::Colouring {
        moving::colouring(g, seed, step)
    }
    fn move_nodes<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &mut [u32], gamma: f64, tol: f64, col: &Self::Colouring, apply: Apply) -> (usize, usize) {
        moving::move_nodes(g, comm, gamma, tol, col, apply)
    }
    fn refine<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &[u32], gamma: f64, theta: f64, seed: u64, step: u64, apply: Apply,
                         col: &Self::Colouring) -> (Vec<u32>, usize) {
        refine::refine(g, comm, gamma, theta, seed, step, apply, col)
    }
    fn aggregate<W: Weight>(&self, g: &Self::Graph<'_, W>, labels: &[u32], n_labels: usize, apply: Apply) -> Self::Graph<'static, f64> {
        Cow::Owned(aggregate::aggregate(g, labels, n_labels, apply))
    }
}
