"""Seeded, deterministic, parallel Leiden community detection.

The same ``seed`` gives the same labels at any number of threads. Graphs are undirected and
weighted: a symmetric sparse matrix (scipy, e.g. scanpy's ``adata.obsp["connectivities"]``), a
CSR ``(indptr, indices, data)`` triple, or an edge list (:func:`leiden_edges`, which takes
polars / pyarrow / numpy columns). Labels are ``0..k-1``, community 0 the largest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import tl
from ._core import __version__, assess_csr, champ_csr, ecs_labels, leiden_csr, modularity_csr, neosbm_csr

__all__ = ["Champ", "assess", "champ", "ecs", "leiden", "leiden_edges", "modularity", "neosbm", "tl", "__version__"]


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
           min_gain: float = 0.0, threads: int | None = None, check_symmetry: bool = True,
           return_info: bool = False):
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
    check_symmetry
        Verify that the matrix is symmetric (O(m)). False skips only that check, for graphs
        symmetric by construction (scanpy's connectivities); bounds and weights are always
        validated, and an asymmetric graph then gives meaningless labels, never a crash.

    return_info
        Also return how the run ended (see Returns).

    Returns
    -------
    int32 labels ``0..k-1``, community 0 the largest (ties: by smallest vertex). With
    ``return_info=True``, ``(labels, info)``: ``info["iterations"]`` (Leiden iterations run),
    ``info["converged"]`` (the last iteration changed nothing), ``info["stop"]`` ("converged",
    "min_gain" or "n_iterations": why it stopped) and ``info["modularity"]``.
    """
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1 or None")
    labels, iterations, stop, q = leiden_csr(*_csr(graph), float(resolution), int(seed), int(n_iterations),
                                             float(theta), float(min_gain), threads, bool(check_symmetry))
    if not return_info:
        return labels
    return labels, {"iterations": iterations, "converged": stop == "converged", "stop": stop, "modularity": q}


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


def ecs(a, b) -> np.ndarray:
    """Element-centric similarity (Gates et al. 2019) of each vertex between partitions ``a`` and ``b`` (any
    label values, one per vertex), in [0, 1]: the overlap of the vertex's two communities over the larger one.
    1 where the communities coincide; the mean is the global ECS. Shows *which* cells two seeds or two
    resolutions move, which NVI/AMI cannot: e.g. ``adata.obs["ecs"] = cl.ecs(labels_seed0, labels_seed1)``."""
    code = lambda l: np.unique(np.asarray(l).ravel(), return_inverse=True)[1].astype(np.int64)  # noqa: E731
    return ecs_labels(code(a), code(b))


@dataclass
class Champ:
    """Partitions on the CHAMP envelope, in increasing resolution: ``labels[i]`` has the highest modularity
    in the pool for every resolution in ``[gamma_lo[i], gamma_hi[i]]``. ``resolution[i]`` and ``seed[i]``:
    the grid point that produced it. ``best``: the index with the widest domain in log resolution.
    ``to_prev``, ``to_neighbours``, ``to_seeds``: agreement under ``metric`` with the previous domain's
    partition, the mean with both neighbours' (small NVI / large AMI: a plateau; a jump: a boundary), and the
    mean with the other seeds' partitions at the same grid resolution (algorithmic noise); NaN where there is
    nothing to compare with. ``metric`` "nvi" is a distance (0 = identical), "ami" a similarity (1 = identical).
    """

    labels: np.ndarray  # (n_domains, n_vertices) int32
    gamma_lo: np.ndarray
    gamma_hi: np.ndarray
    resolution: np.ndarray
    seed: np.ndarray
    n_communities: np.ndarray
    metric: str
    to_prev: np.ndarray
    to_neighbours: np.ndarray
    to_seeds: np.ndarray

    @property
    def log_width(self) -> np.ndarray:
        return np.log(self.gamma_hi / self.gamma_lo)

    @property
    def best(self) -> int:
        return int(np.argmax(self.log_width))


def champ(graph, resolutions, seeds=0, *, gamma_range: tuple[float, float] | None = None, n_iterations: int = -1,
          theta: float = 0.01, min_gain: float = 0.0, threads: int | None = None, check_symmetry: bool = True,
          metric: str = "nvi") -> Champ:
    """Experimental: the API and the scores may change. Pick resolutions by CHAMP (Weir et al. 2017): run :func:`leiden` at every pair of ``resolutions`` x
    ``seeds`` (numpy arrays, e.g. ``np.logspace(-1, 1, 20)`` and ``np.arange(10)``), then keep the partitions
    that have the highest modularity among the pool somewhere in ``gamma_range`` (default: the span of
    ``resolutions``), each with its interval. Modularity is linear in the resolution, so the intervals are
    exact; they are optimal relative to the pool, not globally. Each partition is also scored by its agreement
    with its neighbouring domains and with the other seeds (``metric``: "nvi", normalised variation of
    information, the default, a metric so steps along the domains add up, as in Markov stability; or "ami",
    adjusted mutual information, corrected for chance agreement between fine partitions). Deterministic, at
    any number of ``threads``. Other arguments as :func:`leiden`."""
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1 or None")
    res = np.atleast_1d(np.asarray(resolutions, dtype=np.float64)).ravel()
    seeds = np.atleast_1d(np.asarray(seeds)).ravel()
    if metric not in ("nvi", "ami"):
        raise ValueError("metric must be 'nvi' or 'ami'")
    if not len(res) or not len(seeds):
        raise ValueError("resolutions and seeds must not be empty")
    if not np.all(np.isfinite(res)) or res.min() <= 0:
        raise ValueError("resolutions must be finite and > 0")
    if seeds.dtype.kind not in "iu" or seeds.min() < 0:
        raise ValueError("seeds must be integers >= 0")
    lo, hi = (res.min(), res.max()) if gamma_range is None else map(float, gamma_range)
    if not 0 < lo <= hi < np.inf:
        raise ValueError("gamma_range must satisfy 0 < low <= high < inf")
    csr = _csr(graph)
    index, gamma_lo, gamma_hi, labels, to_prev, to_neighbours, to_seeds = champ_csr(
        *csr, res, seeds.astype(np.uint64), lo, hi, int(n_iterations), float(theta), float(min_gain), threads,
        bool(check_symmetry), metric)
    labels = labels.reshape(len(index), len(csr[0]) - 1)
    i_res, i_seed = np.unravel_index(index, (len(res), len(seeds)))
    return Champ(labels=labels, gamma_lo=gamma_lo, gamma_hi=gamma_hi, resolution=res[i_res], seed=seeds[i_seed],
                 n_communities=labels.max(axis=1, initial=-1).astype(np.int64) + 1, metric=metric, to_prev=to_prev,
                 to_neighbours=to_neighbours, to_seeds=to_seeds)


def _codes(labels):
    """Integer codes 0..K-1 of any label vector, with the sorted unique labels."""
    labels = np.asarray(labels)
    uniq, codes = np.unique(labels, return_inverse=True)
    return np.ascontiguousarray(codes, dtype=np.int64), uniq


def assess(graph, labels, *, check_symmetry: bool = True) -> dict:
    """Statistics of a *given* partition on the graph, labels taken as fixed (an annotation, a
    clustering from another modality). Returns a dict:

    ``loglik``: degree-corrected SBM profile log-likelihood (Karrer & Newman 2011), the BESTest statistic
    (Peel, Larremore & Clauset 2017); ``description_length``: Peixoto (2017) microcanonical DC-SBM
    description length with uniform priors, nats (graph-tool's ``entropy(degree_dl_kind="uniform")``),
    which ranks nested annotations: shorter is better supported; ``significance``: Traag et al. (2013),
    nats; ``surprise``: per label, -ln P(X >= inner edges) under the configuration model, the label's
    cohesion beyond chance; plus ``labels`` (sorted unique), ``size``, ``degree_sum`` and ``inner_edges``
    per label. Exact on unweighted graphs; weights count as multiplicities.
    """
    codes, uniq = _codes(labels)
    ll, dl, sig, surprise, size, kappa, inner = assess_csr(*_csr(graph), codes, check_symmetry)
    return {"loglik": ll, "description_length": dl, "significance": sig, "surprise": surprise, "labels": uniq,
            "size": size, "degree_sum": kappa, "inner_edges": inner}


def neosbm(graph, labels, *, thetas=(1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.4, 0.5),
           max_sweeps: int = 20, seed: int | None = 0, check_symmetry: bool = True) -> dict:
    """neoSBM (Peel, Larremore & Clauset 2017): every vertex keeps its label's block or defects at a cost of
    ln((1 - theta) / theta) nats of DC-SBM likelihood. Greedy single-vertex moves, warm-started along
    ``thetas`` (ascending). Returns ``thetas``, ``defect`` (fraction of vertices off their label at each
    theta; at theta = 1/2 the prior is neutral), ``loglik_gain``, ``sweeps`` and ``blocks`` (the block of
    every vertex at the last theta, as codes into ``labels``). ``seed`` fixes the vertex order of the
    sweeps (a permutation); ``seed=None`` sweeps in vertex order, the deterministic choice for cross-checks.
    """
    codes, uniq = _codes(labels)
    n = len(codes)
    order = np.arange(n, dtype=np.int64) if seed is None else np.random.default_rng(seed).permutation(n).astype(np.int64)
    th = np.ascontiguousarray(sorted(thetas), dtype=np.float64)
    defect, gain, sweeps, blocks = neosbm_csr(*_csr(graph), codes, th, max_sweeps, order, check_symmetry)
    return {"thetas": th, "defect": defect, "loglik_gain": gain, "sweeps": sweeps, "blocks": blocks, "labels": uniq}
