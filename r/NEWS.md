# clusterLibreville 0.99.0

* Initial Bioconductor submission: `leiden()`, `modularity()` and
  `cluster_libreville()` (igraph / bluster interface).
* `find_clusters()`: `leiden()` with the interface of `Seurat::FindClusters`;
  `with_libreville()` runs code (e.g. CHOIR) with `FindClusters` replaced by it.
* `neosbm()`: exact on graphs with self-loops (Seurat's SNN graphs have a unit
  diagonal); the block counts had drifted from the partition's.
