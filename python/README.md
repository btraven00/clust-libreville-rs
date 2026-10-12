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

# a symmetric scipy sparse matrix, e.g. scanpy's kNN connectivities
labels = cl.leiden(adata.obsp["connectivities"], resolution=1.0, seed=0)  # int32, 0 = largest

# scanpy-style: writes adata.obs["leiden"] (categorical) like sc.tl.leiden
cl.tl.leiden(adata, resolution=1.0, random_state=0)

# an undirected edge list: polars / pandas / pyarrow / numpy columns
labels = cl.leiden_edges(df["src"], df["dst"], df["weight"], seed=0)

cl.modularity(graph, labels)
```

Options: `resolution`, `seed`, `n_iterations` (-1: until an iteration changes nothing),
`theta` (refinement randomness), `min_gain` (optional early stop), `threads` (default: all
cores; never changes the result).

License: MIT.
