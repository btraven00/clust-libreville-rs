//! R bindings for detleiden (extendr). The R wrapper (R/leiden.R) converts inputs to a general
//! dgCMatrix; for a symmetric matrix its CSC arrays (p / i / x) are a CSR. Labels go back 1-based.

use extendr_api::prelude::*;

fn graph(p: &[i32], i: &[i32], x: &[f64]) -> dl::Graph {
    let indptr: Vec<usize> = p.iter().map(|&v| v as usize).collect();
    let indices: Vec<u32> = i.iter().map(|&v| v as u32).collect();
    dl::Graph::from_csr(&indptr, &indices, x).unwrap_or_else(|e| throw_r_error(format!("invalid graph: {e}")))
}

/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn leiden_rs(p: &[i32], i: &[i32], x: &[f64], resolution: f64, seed: f64, n_iterations: i32, theta: f64, min_gain: f64,
             threads: i32) -> Vec<i32> {
    let g = graph(p, i, x);
    let o = dl::Options {
        resolution,
        seed: seed as u64,
        n_iterations: n_iterations as i64,
        theta,
        min_gain,
        threads: (threads > 0).then_some(threads as usize), // 0: NULL on the R side
        ..Default::default()
    };
    dl::leiden(&g, &o).iter().map(|&l| l as i32 + 1).collect()
}

/// @keywords internal
#[extendr]
fn modularity_rs(p: &[i32], i: &[i32], x: &[f64], membership: &[i32], resolution: f64) -> f64 {
    let labels: Vec<u32> = membership.iter().map(|&l| l as u32).collect();
    dl::modularity(&graph(p, i, x), &labels, resolution)
}

extendr_module! {
    mod detleiden;
    fn leiden_rs;
    fn modularity_rs;
}
