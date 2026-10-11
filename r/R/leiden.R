# Any supported graph -> a general, double dgCMatrix (for a symmetric matrix, its CSC is a CSR).
as_graph <- function(graph) {
  if (inherits(graph, "igraph")) {
    w <- if ("weight" %in% igraph::edge_attr_names(graph)) "weight" else NULL
    graph <- igraph::as_adjacency_matrix(graph, attr = w, sparse = TRUE)
  }
  m <- methods::as(methods::as(methods::as(graph, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  if (nrow(m) != ncol(m)) stop("graph must be a square (symmetric) matrix")
  m
}

#' Deterministic Leiden community detection
#'
#' Leiden (Traag, Waltman & van Eck 2019) with modularity (resolution `resolution`), seeded and
#' deterministic: the same `seed` gives the same labels at any number of threads.
#'
#' @param graph Undirected weighted graph: a symmetric sparse matrix (anything
#'   `methods::as(graph, "CsparseMatrix")` accepts: a `dgCMatrix`, a Seurat `Graph` such as
#'   `obj@graphs$RNA_snn`, a symmetric `dsCMatrix`, a pattern matrix) or an igraph graph
#'   (edge attribute `weight` if present).
#' @param resolution Resolution of the modularity (gamma).
#' @param seed Seed, a non-negative whole number below 2^53.
#' @param n_iterations Leiden iterations; -1: until an iteration changes nothing.
#' @param theta Randomness of the refinement.
#' @param min_gain With `n_iterations = -1`, also stop once an iteration gains less modularity
#'   than this (0: strict).
#' @param threads Worker threads; `NULL`: all cores. Does not change the result.
#' @return Integer community labels `1..k`, community 1 the largest (ties: by smallest vertex),
#'   named by the graph's row names if it has them.
#' @examples
#' m <- Matrix::sparseMatrix(i = c(1, 2, 3, 4), j = c(2, 3, 4, 1), x = 1, dims = c(4, 4), symmetric = TRUE)
#' leiden(m, seed = 1)
#' @export
leiden <- function(graph, resolution = 1, seed = 0, n_iterations = -1L, theta = 0.01, min_gain = 0, threads = NULL) {
  m <- as_graph(graph)
  stopifnot(seed >= 0, seed < 2^53, seed == floor(seed), is.null(threads) || threads >= 1)
  r <- leiden_rs(m@p, m@i, m@x, as.double(resolution), as.double(seed), as.integer(n_iterations), as.double(theta),
                 as.double(min_gain), if (is.null(threads)) 0L else as.integer(threads))
  if (!is.null(rownames(m))) names(r) <- rownames(m)
  r
}

#' Modularity of a partition
#'
#' Newman's modularity at resolution `resolution`, summed in a fixed order (the quantity
#' [leiden()] maximises).
#'
#' @inheritParams leiden
#' @param membership Labels, one per vertex (any atomic type).
#' @return A number.
#' @export
modularity <- function(graph, membership, resolution = 1) {
  m <- as_graph(graph)
  stopifnot(length(membership) == nrow(m))
  modularity_rs(m@p, m@i, m@x, match(membership, unique(membership)) - 1L, as.double(resolution))
}
