//! Aggregation: one coarse row per cluster, as in GVE-Leiden (`leidenAggregateEdgesOmpW`), but
//! with members in vertex-id order (a counting sort, not GVE's atomic slots). Each row's weights
//! are summed, and its columns appear, in a fixed order, so the coarse graph is deterministic.
//! Rows are built in parallel (per-thread scratch) and concatenated in row order.

use std::cell::RefCell;

use rayon::prelude::*;

use crate::graph::Graph;
use crate::refine::bucket;

const MIN_ROWS: usize = 64; // rows per rayon task; any value gives the same output

thread_local! {
    // weight to each coarse column, and whether it was touched; all zero/false between rows
    static SCRATCH: RefCell<(Vec<f64>, Vec<bool>, Vec<u32>)> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

pub(crate) fn aggregate(g: &Graph, labels: &[u32], n_labels: usize) -> Graph {
    let members = bucket(labels, n_labels);
    let rows: Vec<(Vec<u32>, Vec<f64>)> = members
        .par_iter()
        .with_min_len(MIN_ROWS)
        .map(|row| {
            SCRATCH.with(|s| {
                let (wt, seen, touched) = &mut *s.borrow_mut();
                if wt.len() < n_labels {
                    wt.resize(n_labels, 0.0);
                    seen.resize(n_labels, false);
                }
                for &u in row {
                    let (t, w) = g.row(u as usize);
                    for (&v, &w) in t.iter().zip(w) {
                        let d = labels[v as usize] as usize;
                        if !seen[d] {
                            seen[d] = true;
                            touched.push(d as u32);
                        }
                        wt[d] += w;
                    }
                }
                let weights = touched.iter().map(|&d| std::mem::take(&mut wt[d as usize])).collect();
                for &d in touched.iter() {
                    seen[d as usize] = false;
                }
                (std::mem::take(touched), weights)
            })
        })
        .collect();
    let mut offsets = Vec::with_capacity(n_labels + 1);
    offsets.push(0);
    let m: usize = rows.iter().map(|r| r.0.len()).sum();
    let (mut targets, mut weights) = (Vec::with_capacity(m), Vec::with_capacity(m));
    for (t, w) in rows {
        targets.extend_from_slice(&t);
        weights.extend_from_slice(&w);
        offsets.push(targets.len());
    }
    Graph::from_parts(offsets, targets, weights)
}
