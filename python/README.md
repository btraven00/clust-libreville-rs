# sc-cluster-libreville

Seeded, deterministic, parallel Leiden community detection (modularity), written in Rust. **The same
seed gives the same labels at any number of threads**, so results are reproducible on a laptop
and on a 64-core server. The parallel design follows GVE-Leiden (Sahu 2023), with synchronous,
order-independent moves instead of asynchronous ones.

```
pip install sc-cluster-libreville   # import cluster_libreville
```

```python
import cluster_libreville as cl
import numpy as np

# a symmetric scipy sparse matrix, e.g. scanpy's kNN connectivities
graph = adata.obsp["connectivities"]
labels = cl.leiden(graph, resolution=1.0, seed=0)  # int32, 0 = largest
cl.modularity(graph, labels)  # of any labels on the graph, e.g. to compare partitions

# scanpy-style: writes adata.obs["leiden"] (categorical) like sc.tl.leiden
cl.tl.leiden(adata, resolution=1.0, random_state=0)

# an undirected edge list: polars / pandas / pyarrow / numpy columns
labels = cl.leiden_edges(df["src"], df["dst"], df["weight"], seed=0)

# experimental, the API may change. Pick the resolution (CHAMP): Leiden over resolutions x seeds, keep the partitions that are best somewhere
c = cl.champ(graph, np.logspace(-1, 1, 20), seeds=np.arange(5))  # metric="nvi" (default) or "ami"
cl.ecs(c.labels[0], c.labels[1])  # per-cell element-centric similarity: which cells move between two partitions
c.labels[c.best]  # the widest resolution range; c.to_seeds near 0 = stable across seeds
cl.tl.champ(adata)  # scanpy-style: adata.obs["champ"], adata.uns["champ"]["domains"]
```

Options: `resolution`, `seed`, `n_iterations` (-1: until an iteration changes nothing),
`theta` (refinement randomness), `min_gain` (optional early stop), `threads` (default: all
cores; never changes the result).

License: MIT.
