//! Python bindings for clust-libreville-rs (PyO3). python/cluster_libreville/__init__.py normalises inputs to a
//! symmetric CSR (int64 indptr, int32 indices, float64 weights); labels come back as int32.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn graph(indptr: &[i64], indices: &[i32], data: &[f64], check_symmetry: bool) -> PyResult<clust_libreville_rs::Graph> {
    if indptr.iter().any(|&x| x < 0) || indices.iter().any(|&x| x < 0) {
        return Err(PyValueError::new_err("invalid graph: negative index"));
    }
    let indptr: Vec<usize> = indptr.iter().map(|&x| x as usize).collect();
    let indices: Vec<u32> = indices.iter().map(|&x| x as u32).collect();
    let build = if check_symmetry { clust_libreville_rs::Graph::from_csr } else { clust_libreville_rs::Graph::from_csr_trusted };
    build(&indptr, &indices, data).map_err(|e| PyValueError::new_err(format!("invalid graph: {e}")))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, resolution, seed, n_iterations, theta, min_gain, threads, check_symmetry))]
#[allow(clippy::too_many_arguments)]
fn leiden_csr<'py>(py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i32>,
                   data: PyReadonlyArray1<'py, f64>, resolution: f64, seed: u64, n_iterations: i64, theta: f64, min_gain: f64,
                   threads: Option<usize>, check_symmetry: bool)
                   -> PyResult<(Bound<'py, PyArray1<i32>>, usize, &'static str, f64)> {
    let (ip, ix, d) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let o = clust_libreville_rs::Options { resolution, seed, n_iterations, theta, min_gain, threads, ..Default::default() };
    // the graph build too runs on `threads` (its validation is parallel), not on rayon's global pool
    let r = py.detach(|| clust_libreville_rs::with_threads(threads, || {
        graph(ip, ix, d, check_symmetry).map(|g| clust_libreville_rs::leiden_run(&g, &o))
    }))?;
    let stop = match r.stop {
        clust_libreville_rs::Stop::Converged => "converged",
        clust_libreville_rs::Stop::MinGain => "min_gain",
        clust_libreville_rs::Stop::IterationCap => "n_iterations",
    };
    let labels = r.labels.into_iter().map(|l| l as i32).collect::<Vec<i32>>().into_pyarray(py);
    Ok((labels, r.iterations, stop, r.modularity))
}

#[pyfunction]
fn modularity_csr<'py>(indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i32>, data: PyReadonlyArray1<'py, f64>,
                       labels: PyReadonlyArray1<'py, i64>, resolution: f64) -> PyResult<f64> {
    let g = graph(indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, true)?;
    let l: Vec<u32> = labels.as_slice()?.iter().map(|&x| x as u32).collect();
    Ok(clust_libreville_rs::modularity(&g, &l, resolution))
}

/// Per-vertex element-centric similarity of two label vectors (codes 0..k-1, same length).
#[pyfunction]
fn ecs_labels<'py>(py: Python<'py>, a: PyReadonlyArray1<'py, i64>, b: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (a, b) = (a.as_slice()?, b.as_slice()?);
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!("partitions of different sizes: {} and {}", a.len(), b.len())));
    }
    let (a, b): (Vec<u32>, Vec<u32>) = (a.iter().map(|&x| x as u32).collect(), b.iter().map(|&x| x as u32).collect());
    Ok(clust_libreville_rs::ecs(&a, &b).into_pyarray(py))
}

/// Leiden at every (resolution, seed) of the grid, in `meshgrid(indexing="ij")` order (pool index =
/// i_resolution * n_seeds + i_seed), then CHAMP over the pool. Returns the domains' pool indices, bounds and
/// labels (flat, one row of n per domain), and their agreement under `metric` ("nvi" or "ami"): with the
/// previous domain, mean with both neighbours, mean with the other seeds at their grid resolution.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, resolutions, seeds, gamma_min, gamma_max, n_iterations, theta, min_gain, threads, check_symmetry, metric))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn champ_csr<'py>(py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i32>,
                  data: PyReadonlyArray1<'py, f64>, resolutions: PyReadonlyArray1<'py, f64>, seeds: PyReadonlyArray1<'py, u64>,
                  gamma_min: f64, gamma_max: f64, n_iterations: i64, theta: f64, min_gain: f64, threads: Option<usize>,
                  check_symmetry: bool, metric: &str) -> PyResult<Bound<'py, PyTuple>> {
    let metric = match metric {
        "nvi" => clust_libreville_rs::Metric::Nvi,
        "ami" => clust_libreville_rs::Metric::Ami,
        m => return Err(PyValueError::new_err(format!("metric must be 'nvi' or 'ami', not {m:?}"))),
    };
    let (ip, ix, d) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let (rs, ss) = (resolutions.as_slice()?, seeds.as_slice()?);
    let o = clust_libreville_rs::Options { n_iterations, theta, min_gain, threads, ..Default::default() };
    let domains = py.detach(|| clust_libreville_rs::with_threads(threads, || {
        graph(ip, ix, d, check_symmetry).map(|g| clust_libreville_rs::champ_grid(&g, rs, ss, &o, gamma_min, gamma_max, metric))
    }))?;
    let f = |x: fn(&(clust_libreville_rs::Domain, Vec<u32>, clust_libreville_rs::Agreement)) -> f64| {
        domains.iter().map(x).collect::<Vec<_>>().into_pyarray(py)
    };
    let index = domains.iter().map(|(d, _, _)| d.index as i64).collect::<Vec<_>>().into_pyarray(py);
    let labels = domains.iter().flat_map(|(_, l, _)| l.iter().map(|&l| l as i32)).collect::<Vec<_>>().into_pyarray(py);
    (index, f(|d| d.0.lo), f(|d| d.0.hi), labels, f(|d| d.2.prev), f(|d| d.2.neighbours), f(|d| d.2.seeds))
        .into_pyobject(py)
}

/// Statistics of a given partition on the graph (labels 0..K-1): DC-SBM log-likelihood, description
/// length (Peixoto 2017, uniform priors), Significance (Traag 2013), per-block surprise under the
/// configuration model, and the block sizes, degree sums and inner edge counts.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn assess_csr<'py>(py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i32>,
                   data: PyReadonlyArray1<'py, f64>, labels: PyReadonlyArray1<'py, i64>, check_symmetry: bool)
                   -> PyResult<(f64, f64, f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let g = graph(indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, check_symmetry)?;
    let l: Vec<u32> = labels.as_slice()?.iter().map(|&x| x as u32).collect();
    if l.len() != g.n() {
        return Err(PyValueError::new_err(format!("{} labels for {} vertices", l.len(), g.n())));
    }
    let s = clust_libreville_rs::block_stats(&g, &l);
    let inner: Vec<f64> = (0..s.k).map(|r| s.inner_edges(r)).collect();
    Ok((clust_libreville_rs::dcsbm_loglik(&s), clust_libreville_rs::description_length(&g, &l),
        clust_libreville_rs::significance(&g, &l), clust_libreville_rs::label_surprise(&g, &l).into_pyarray(py),
        s.size.into_pyarray(py), s.kappa.into_pyarray(py), inner.into_pyarray(py)))
}

/// neoSBM (Peel, Larremore & Clauset 2017) over `thetas`, sweeping vertices in `order`: per theta the
/// defecting fraction, the log-likelihood gain and the sweeps run; the blocks at the last theta.
#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn neosbm_csr<'py>(py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i32>,
                   data: PyReadonlyArray1<'py, f64>, labels: PyReadonlyArray1<'py, i64>, thetas: PyReadonlyArray1<'py, f64>,
                   max_sweeps: usize, order: PyReadonlyArray1<'py, i64>, check_symmetry: bool)
                   -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i32>>)> {
    let (ip, ix, d) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let l: Vec<u32> = labels.as_slice()?.iter().map(|&x| x as u32).collect();
    let o: Vec<u32> = order.as_slice()?.iter().map(|&x| x as u32).collect();
    let th = thetas.as_slice()?.to_vec();
    if th.iter().any(|&t| !(t > 0.0 && t < 1.0)) {
        return Err(PyValueError::new_err("thetas must be in (0, 1)"));
    }
    let r = py.detach(|| graph(ip, ix, d, check_symmetry).map(|g| {
        if l.len() != g.n() || o.len() != g.n() {
            return Err(PyValueError::new_err("labels and order must have one entry per vertex"));
        }
        Ok(clust_libreville_rs::neosbm(&g, &l, &th, max_sweeps, &o))
    }))??;
    Ok((r.defect.into_pyarray(py), r.loglik_gain.into_pyarray(py),
        r.sweeps.into_iter().map(|x| x as i64).collect::<Vec<i64>>().into_pyarray(py),
        r.labels.into_iter().map(|x| x as i32).collect::<Vec<i32>>().into_pyarray(py)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(leiden_csr, m)?)?;
    m.add_function(wrap_pyfunction!(modularity_csr, m)?)?;
    m.add_function(wrap_pyfunction!(champ_csr, m)?)?;
    m.add_function(wrap_pyfunction!(ecs_labels, m)?)?;
    m.add_function(wrap_pyfunction!(assess_csr, m)?)?;
    m.add_function(wrap_pyfunction!(neosbm_csr, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
