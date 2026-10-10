//! Aggregation: one coarse row per cluster, as in GVE-Leiden (`leidenAggregateEdgesOmpW`), but
//! with members in vertex-id order (a counting sort, not GVE's atomic slots). Each row's weights
//! are summed, and its columns appear, in a fixed order, so the coarse graph is deterministic.
//! Rows are independent; phase 2 builds them in parallel.

use crate::graph::Graph;
use crate::refine::bucket;

pub(crate) fn aggregate(g: &Graph, labels: &[u32], n_labels: usize) -> Graph {
    let members = bucket(labels, n_labels);
    let mut offsets = Vec::with_capacity(n_labels + 1);
    let mut targets: Vec<u32> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut wt = vec![0.0f64; n_labels];
    let mut seen = vec![false; n_labels];
    let mut touched: Vec<u32> = Vec::new();
    offsets.push(0);
    for row in &members {
        for &u in row {
            let (t, w) = g.row(u as usize);
            for (&v, &w) in t.iter().zip(w) {
                let d = labels[v as usize];
                if !seen[d as usize] {
                    seen[d as usize] = true;
                    touched.push(d);
                }
                wt[d as usize] += w;
            }
        }
        for &d in &touched {
            targets.push(d);
            weights.push(wt[d as usize]);
            wt[d as usize] = 0.0;
            seen[d as usize] = false;
        }
        touched.clear();
        offsets.push(targets.len());
    }
    Graph::from_parts(offsets, targets, weights)
}
