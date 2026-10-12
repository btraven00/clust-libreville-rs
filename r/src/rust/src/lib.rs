//! R bindings for clust-libreville-rs (extendr). The R wrapper (R/leiden.R) converts inputs to a general
//! dgCMatrix; for a symmetric matrix its CSC arrays (p / i / x) are a CSR. Labels go back 1-based.
#![allow(non_snake_case)] // the lib is named after the R package (R_init_clusterLibreville)

use extendr_api::prelude::*;

/// The graph, or why not. No R API here: this also runs on worker threads (dl::with_threads).
fn try_graph(p: &[i32], i: &[i32], x: &[f64], check_symmetry: bool) -> Result<dl::Graph, String> {
    let indptr: Vec<usize> = p.iter().map(|&v| v as usize).collect();
    let indices: Vec<u32> = i.iter().map(|&v| v as u32).collect();
    let build = if check_symmetry { dl::Graph::from_csr } else { dl::Graph::from_csr_trusted };
    build(&indptr, &indices, x).map_err(|e| format!("invalid graph: {e}"))
}

fn graph(p: &[i32], i: &[i32], x: &[f64], check_symmetry: bool) -> dl::Graph {
    try_graph(p, i, x, check_symmetry).unwrap_or_else(|e| throw_r_error(e))
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
             threads: i32, check_symmetry: bool) -> List {

    let o = dl::Options {
        resolution,
        seed: seed as u64,
        n_iterations: n_iterations as i64,
        theta,
        min_gain,
        threads: (threads > 0).then_some(threads as usize), // 0: NULL on the R side
        ..Default::default()
    };
    let t = (threads > 0).then_some(threads as usize);
    // the graph build too runs on `threads` (its validation is parallel), not on rayon's global pool
    let r = dl::with_threads(t, || try_graph(p, i, x, check_symmetry).map(|g| dl::leiden_run(&g, &o)))
        .unwrap_or_else(|e| throw_r_error(e)); // back on R's thread
    let stop = match r.stop {
        dl::Stop::Converged => "converged",
        dl::Stop::MinGain => "min_gain",
        dl::Stop::IterationCap => "n_iterations",
    };
    let labels: Vec<i32> = r.labels.iter().map(|&l| l as i32 + 1).collect();
    list!(labels = labels, iterations = r.iterations as i32, stop = stop, modularity = r.modularity)
}

/// @keywords internal
#[extendr]
fn modularity_rs(p: &[i32], i: &[i32], x: &[f64], membership: &[i32], resolution: f64) -> f64 {
    let labels: Vec<u32> = membership.iter().map(|&l| l as u32).collect();
    dl::modularity(&graph(p, i, x, true), &labels, resolution)
}

/// Leiden at every (resolution, seed), in grid order (pool index = i_resolution * n_seeds + i_seed, 0-based),
/// then CHAMP over the pool. Labels: one column of n per domain, 1-based. Agreement under `metric` ("nvi" or
/// "ami", checked on the R side): with the previous domain, mean with both neighbours, mean with the other seeds.
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn champ_rs(p: &[i32], i: &[i32], x: &[f64], resolutions: &[f64], seeds: &[f64], gamma_min: f64, gamma_max: f64,
            n_iterations: i32, theta: f64, min_gain: f64, threads: i32, check_symmetry: bool, metric: &str) -> List {
    let metric = if metric == "ami" { dl::Metric::Ami } else { dl::Metric::Nvi };
    let t = (threads > 0).then_some(threads as usize); // 0: NULL on the R side
    let o = dl::Options { n_iterations: n_iterations as i64, theta, min_gain, threads: t, ..Default::default() };
    let seeds: Vec<u64> = seeds.iter().map(|&s| s as u64).collect();
    let domains = dl::with_threads(t, || {
        try_graph(p, i, x, check_symmetry).map(|g| dl::champ_grid(&g, resolutions, &seeds, &o, gamma_min, gamma_max, metric))
    })
    .unwrap_or_else(|e| throw_r_error(e)); // back on R's thread
    let f = |x: fn(&(dl::Domain, Vec<u32>, dl::Agreement)) -> f64| domains.iter().map(x).collect::<Vec<f64>>();
    let index: Vec<i32> = domains.iter().map(|(d, _, _)| d.index as i32).collect();
    let labels: Vec<i32> = domains.iter().flat_map(|(_, l, _)| l.iter().map(|&l| l as i32 + 1)).collect();
    list!(index = index, gamma_lo = f(|d| d.0.lo), gamma_hi = f(|d| d.0.hi), labels = labels,
          to_prev = f(|d| d.2.prev), to_neighbours = f(|d| d.2.neighbours), to_seeds = f(|d| d.2.seeds))
}

extendr_module! {
    mod clusterLibreville;
    fn edges_csr_rs;
    fn leiden_rs;
    fn modularity_rs;
    fn champ_rs;
}
