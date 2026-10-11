"""Seeded, deterministic, parallel Leiden community detection.

The same ``seed`` gives the same labels at any number of threads. Graphs are undirected and
weighted: a symmetric sparse matrix (scipy, e.g. scanpy's ``adata.obsp["connectivities"]``), a
CSR ``(indptr, indices, data)`` triple, or an edge list (:func:`leiden_edges`, which takes
polars / pyarrow / numpy columns). Labels are ``0..k-1``, community 0 the largest.
"""

from __future__ import annotations

import numpy as np

from . import tl
from ._core import __version__, leiden_csr, modularity_csr

__all__ = ["leiden", "leiden_edges", "modularity", "tl", "__version__"]


def _csr(graph):
    """(indptr int64, indices int32, data float64) of a square sparse matrix or a CSR triple."""
    if isinstance(graph, tuple):
        indptr, indices, data = graph
    else:
        if not hasattr(graph, "indptr") or getattr(graph, "format", "csr") not in ("csr", "csc"):
            if not hasattr(graph, "tocsr"):
                raise TypeError("graph must be a scipy sparse matrix or an (indptr, indices, data) tuple")
            graph = graph.tocsr()
        if graph.shape[0] != graph.shape[1]:
            raise ValueError(f"graph must be square (symmetric), got shape {graph.shape}")
        # a symmetric matrix's CSC arrays are its CSR arrays
        indptr, indices, data = graph.indptr, graph.indices, graph.data
    # no copy when the dtypes already match (scanpy: int32 indices; float32 data is widened)
    return (np.ascontiguousarray(indptr, dtype=np.int64), np.ascontiguousarray(indices, dtype=np.int32),
            np.ascontiguousarray(data, dtype=np.float64))


def leiden(graph, *, resolution: float = 1.0, seed: int = 0, n_iterations: int = -1, theta: float = 0.01,
           min_gain: float = 0.0, threads: int | None = None) -> np.ndarray:
    """Leiden communities of an undirected weighted graph, maximising modularity.

    Parameters
    ----------
    graph
        A symmetric scipy sparse matrix (any format; CSR/CSC are used in place) or an
        ``(indptr, indices, data)`` CSR triple.
    resolution
        Resolution of the modularity (gamma).
    seed
        Seed (0 <= seed < 2**64).
    n_iterations
        Leiden iterations; -1: until an iteration changes nothing.
    theta
        Randomness of the refinement.
    min_gain
        With ``n_iterations=-1``, also stop once an iteration gains less modularity than this.
    threads
        Worker threads; None: all cores (or ``RAYON_NUM_THREADS``). Does not change the result.

    Returns
    -------
    int32 labels ``0..k-1``, community 0 the largest (ties: by smallest vertex).
    """
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1 or None")
    return leiden_csr(*_csr(graph), float(resolution), int(seed), int(n_iterations), float(theta), float(min_gain),
                      threads)


def edges_to_csr(src, dst, weight=None, n: int | None = None):
    """Undirected edge list -> symmetric CSR triple. Each row is one edge {src, dst} (listed once;
    repeated edges add up); self-loops count once. ``n``: number of vertices (default max id + 1)."""
    src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
    w = np.ones(len(src)) if weight is None else np.asarray(weight, dtype=np.float64)
    if not (len(src) == len(dst) == len(w)):
        raise ValueError("src, dst and weight must have the same length")
    if len(src) and min(src.min(), dst.min()) < 0:
        raise ValueError("vertex ids must be >= 0")
    n = int(max(src.max(initial=-1), dst.max(initial=-1)) + 1) if n is None else int(n)
    loop = src == dst
    i = np.concatenate([src, dst[~loop]])
    j = np.concatenate([dst, src[~loop]])
    w = np.concatenate([w, w[~loop]])
    order = np.lexsort((j, i))
    i, j, w = i[order], j[order], w[order]
    first = np.ones(len(i), dtype=bool)
    first[1:] = (i[1:] != i[:-1]) | (j[1:] != j[:-1])
    starts = np.flatnonzero(first)
    w = np.add.reduceat(w, starts) if len(starts) else w
    i, j = i[starts], j[starts]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(i, minlength=n), out=indptr[1:])
    return indptr, j.astype(np.int32), w


def leiden_edges(src, dst, weight=None, *, n: int | None = None, **kwargs) -> np.ndarray:
    """:func:`leiden` on an undirected edge list: one edge per row, e.g. the columns of a polars
    or pandas DataFrame (``df["src"], df["dst"], df["w"]``), pyarrow arrays or numpy arrays.
    Each edge counts once in both directions. Keyword arguments as :func:`leiden`."""
    return leiden(edges_to_csr(src, dst, weight, n), **kwargs)


def modularity(graph, labels, *, resolution: float = 1.0) -> float:
    """Newman modularity of ``labels`` (any values, one per vertex) at ``resolution``."""
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    return modularity_csr(*_csr(graph), codes.astype(np.int64).ravel(), float(resolution))
