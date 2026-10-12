"""scanpy-style tools: ``cluster_libreville.tl.leiden(adata, ...)`` mirrors ``scanpy.tl.leiden``."""

from __future__ import annotations


def _adjacency(adata, adjacency, neighbors_key, obsp):
    """``adjacency``, else ``adata.obsp[obsp]``, else the connectivities of ``sc.pp.neighbors``."""
    if adjacency is not None:
        return adjacency
    if obsp is not None and neighbors_key is not None:
        raise ValueError("pass obsp or neighbors_key, not both")
    if obsp is None:
        key = neighbors_key or "neighbors"
        if key not in adata.uns:
            raise ValueError(f"no neighbors graph in adata.uns[{key!r}]: run sc.pp.neighbors first")
        obsp = adata.uns[key].get("connectivities_key", "connectivities")
    return adata.obsp[obsp]


def leiden(adata, resolution: float = 1.0, *, random_state: int = 0, key_added: str = "leiden", adjacency=None,
           neighbors_key: str | None = None, obsp: str | None = None, n_iterations: int = -1, theta: float = 0.01,
           min_gain: float = 0.0, threads: int | None = None, check_symmetry: bool = True, copy: bool = False):
    """Cluster cells with deterministic Leiden, like ``sc.tl.leiden``.

    The graph is ``adjacency``, else ``adata.obsp[obsp]``, else the connectivities of
    ``sc.pp.neighbors`` (``adata.uns[neighbors_key or "neighbors"]``), weighted. Writes
    ``adata.obs[key_added]`` (categorical, categories "0".."k-1", "0" the largest) and
    ``adata.uns[key_added]["params"]``, plus how the run ended (``iterations``, ``converged``, ``stop``,
    ``modularity``). Returns the AnnData if ``copy``, else None. The same
    ``random_state`` gives the same clusters at any number of ``threads``.
    """
    import pandas as pd

    from . import leiden as _leiden

    adata = adata.copy() if copy else adata
    adjacency = _adjacency(adata, adjacency, neighbors_key, obsp)
    labels, info = _leiden(adjacency, resolution=resolution, seed=random_state, n_iterations=n_iterations,
                           theta=theta, min_gain=min_gain, threads=threads, check_symmetry=check_symmetry,
                           return_info=True)
    k = int(labels.max()) + 1 if len(labels) else 0
    adata.obs[key_added] = pd.Categorical.from_codes(labels, categories=[str(c) for c in range(k)])
    adata.uns[key_added] = {"params": {"resolution": resolution, "random_state": random_state,
                                       "n_iterations": n_iterations, "theta": theta, "min_gain": min_gain,
                                       "flavor": "cluster-libreville"}, **info}
    return adata if copy else None


def champ(adata, resolutions=None, seeds=None, *, gamma_range=None, key_added: str = "champ", adjacency=None,
          neighbors_key: str | None = None, obsp: str | None = None, n_iterations: int = -1, theta: float = 0.01,
          min_gain: float = 0.0, threads: int | None = None, check_symmetry: bool = True, metric: str = "nvi",
          copy: bool = False):
    """Experimental: the API and the scores may change. Pick the resolution by CHAMP: :func:`cluster_libreville.champ` on the cells' graph (as :func:`leiden`).

    ``resolutions`` (default ``np.logspace(-1, 1, 20)``) x ``seeds`` (default ``np.arange(5)``) is the grid
    of Leiden runs. Writes the partition with the widest domain (in log resolution) to
    ``adata.obs[key_added]`` (categorical, "0" the largest), every partition on the envelope to
    ``adata.obsm[key_added]`` (cells x domains, int32, in increasing resolution), and to
    ``adata.uns[key_added]``: ``params``, ``best`` (its column in obsm) and ``domains`` (one entry per
    column: ``gamma_lo``, ``gamma_hi``, ``log_width``, ``n_communities``, ``resolution``, ``seed``, and the
    agreement under ``metric``: ``to_prev``, ``to_neighbours``, ``to_seeds``; e.g.
    ``pd.DataFrame(adata.uns[key_added]["domains"])``). Returns the AnnData if ``copy``, else None.
    """
    import numpy as np
    import pandas as pd

    from . import champ as _champ

    resolutions = np.logspace(-1, 1, 20) if resolutions is None else np.asarray(resolutions, dtype=np.float64)
    seeds = np.arange(5) if seeds is None else np.asarray(seeds)
    adata = adata.copy() if copy else adata
    c = _champ(_adjacency(adata, adjacency, neighbors_key, obsp), resolutions, seeds, gamma_range=gamma_range,
               n_iterations=n_iterations, theta=theta, min_gain=min_gain, threads=threads,
               check_symmetry=check_symmetry, metric=metric)
    best = c.labels[c.best]
    k = int(best.max()) + 1 if len(best) else 0
    adata.obs[key_added] = pd.Categorical.from_codes(best, categories=[str(i) for i in range(k)])
    adata.obsm[key_added] = np.ascontiguousarray(c.labels.T)
    adata.uns[key_added] = {
        "params": {"resolutions": resolutions, "seeds": seeds,
                   "gamma_range": np.array([c.gamma_lo[0], c.gamma_hi[-1]]) if len(c.gamma_lo) else np.array([]),
                   "n_iterations": n_iterations, "theta": theta, "min_gain": min_gain, "metric": metric,
                   "flavor": "cluster-libreville"},
        "best": c.best,
        "domains": {"gamma_lo": c.gamma_lo, "gamma_hi": c.gamma_hi, "log_width": c.log_width,
                    "n_communities": c.n_communities, "resolution": c.resolution, "seed": c.seed,
                    "to_prev": c.to_prev, "to_neighbours": c.to_neighbours, "to_seeds": c.to_seeds},
    }
    return adata if copy else None
