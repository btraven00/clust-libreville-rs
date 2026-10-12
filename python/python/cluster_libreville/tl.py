"""scanpy-style tools: ``cluster_libreville.tl.leiden(adata, ...)`` mirrors ``scanpy.tl.leiden``."""

from __future__ import annotations


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
    if adjacency is None:
        if obsp is not None and neighbors_key is not None:
            raise ValueError("pass obsp or neighbors_key, not both")
        if obsp is None:
            key = neighbors_key or "neighbors"
            if key not in adata.uns:
                raise ValueError(f"no neighbors graph in adata.uns[{key!r}]: run sc.pp.neighbors first")
            obsp = adata.uns[key].get("connectivities_key", "connectivities")
        adjacency = adata.obsp[obsp]
    labels, info = _leiden(adjacency, resolution=resolution, seed=random_state, n_iterations=n_iterations,
                           theta=theta, min_gain=min_gain, threads=threads, check_symmetry=check_symmetry,
                           return_info=True)
    k = int(labels.max()) + 1 if len(labels) else 0
    adata.obs[key_added] = pd.Categorical.from_codes(labels, categories=[str(c) for c in range(k)])
    adata.uns[key_added] = {"params": {"resolution": resolution, "random_state": random_state,
                                       "n_iterations": n_iterations, "theta": theta, "min_gain": min_gain,
                                       "flavor": "cluster-libreville"}, **info}
    return adata if copy else None
