//! Python bindings for clust-libreville-rs (PyO3). python/cluster_libreville/__init__.py normalises inputs to a
//! symmetric CSR (int64 indptr, int32 indices, float64 weights); labels come back as int32.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
                   threads: Option<usize>, check_symmetry: bool) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let g = graph(indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, check_symmetry)?;
    let o = clust_libreville_rs::Options { resolution, seed, n_iterations, theta, min_gain, threads, ..Default::default() };
    let labels = py.detach(|| clust_libreville_rs::leiden(&g, &o));
    Ok(labels.into_iter().map(|l| l as i32).collect::<Vec<i32>>().into_pyarray(py))
}

#[pyfunction]
fn modularity_csr<'py>(indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i32>, data: PyReadonlyArray1<'py, f64>,
                       labels: PyReadonlyArray1<'py, i64>, resolution: f64) -> PyResult<f64> {
    let g = graph(indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, true)?;
    let l: Vec<u32> = labels.as_slice()?.iter().map(|&x| x as u32).collect();
    Ok(clust_libreville_rs::modularity(&g, &l, resolution))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(leiden_csr, m)?)?;
    m.add_function(wrap_pyfunction!(modularity_csr, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
