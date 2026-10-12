//! R bindings for clust-libreville-rs (extendr). The R wrapper (R/leiden.R) converts inputs to a general
//! dgCMatrix; for a symmetric matrix its CSC arrays (p / i / x) are a CSR. Labels go back 1-based.
#![allow(non_snake_case)] // the lib is named after the R package (R_init_clusterLibreville)

use extendr_api::prelude::*;

fn graph(p: &[i32], i: &[i32], x: &[f64], check_symmetry: bool) -> dl::Graph {
    let indptr: Vec<usize> = p.iter().map(|&v| v as usize).collect();
    let indices: Vec<u32> = i.iter().map(|&v| v as u32).collect();
    let build = if check_symmetry { dl::Graph::from_csr } else { dl::Graph::from_csr_trusted };
    build(&indptr, &indices, x).unwrap_or_else(|e| throw_r_error(format!("invalid graph: {e}")))
}

/// An undirected edge list (1-based, as igraph::as_edgelist gives it) as a symmetric CSR (0-based p / i /
/// x, like a dgCMatrix's slots): each edge in both rows, a self-loop once, repeated edges summed in edge
/// order, columns sorted within each row.
/// @keywords internal
#[extendr]
fn edges_csr_rs(from: &[f64], to: &[f64], w: &[f64], n: i32) -> List {
    let n = n as usize;
    let mut start = vec![0usize; n + 1];
    for (&a, &b) in from.iter().zip(to) {
        start[a as usize] += 1;
        if a != b {
            start[b as usize] += 1;
        }
    }
    for v in 0..n {
        start[v + 1] += start[v];
    }
    let mut fill = start[..n].to_vec();
    let mut cell: Vec<(u32, f64)> = vec![(0, 0.0); start[n]];
    for ((&a, &b), &x) in from.iter().zip(to).zip(w) {
        let (a, b) = (a as usize - 1, b as usize - 1);
        cell[fill[a]] = (b as u32, x);
        fill[a] += 1;
        if a != b {
            cell[fill[b]] = (a as u32, x);
            fill[b] += 1;
        }
    }
    let (mut p, mut i, mut x) = (Vec::with_capacity(n + 1), Vec::with_capacity(cell.len()), Vec::with_capacity(cell.len()));
    p.push(0i32);
    for v in 0..n {
        let row = &mut cell[start[v]..start[v + 1]];
        row.sort_by_key(|e| e.0); // stable: repeated edges keep their order
        for &(c, wt) in row.iter() {
            if i.len() > *p.last().unwrap() as usize && *i.last().unwrap() == c as i32 {
                *x.last_mut().unwrap() += wt;
            } else {
                i.push(c as i32);
                x.push(wt);
            }
        }
        p.push(i.len() as i32);
    }
    list!(p = p, i = i, x = x)
}

/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn leiden_rs(p: &[i32], i: &[i32], x: &[f64], resolution: f64, seed: f64, n_iterations: i32, theta: f64, min_gain: f64,
             threads: i32, check_symmetry: bool) -> Vec<i32> {
    let g = graph(p, i, x, check_symmetry);
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
    dl::modularity(&graph(p, i, x, true), &labels, resolution)
}

extendr_module! {
    mod clusterLibreville;
    fn edges_csr_rs;
    fn leiden_rs;
    fn modularity_rs;
}
