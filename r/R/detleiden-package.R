#' detleiden: deterministic parallel Leiden community detection
#'
#' Seeded, deterministic Leiden clustering of large sparse graphs, in Rust:
#' [leiden()] on sparse matrices, Seurat graphs and igraph graphs,
#' [modularity()], and [cluster_detleiden()] for igraph and bluster. The same
#' seed gives the same labels at any number of threads.
#'
#' @importClassesFrom Matrix CsparseMatrix generalMatrix dMatrix
#' @keywords internal
"_PACKAGE"
