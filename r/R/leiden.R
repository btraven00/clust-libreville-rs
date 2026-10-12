## Any supported graph -> a general, double dgCMatrix (for a symmetric matrix,
## its CSC arrays are a CSR).
.as_graph <- function(graph) {
    if (inherits(graph, "igraph")) {
        w <- if ("weight" %in% igraph::edge_attr_names(graph)) "weight"
        graph <- igraph::as_adjacency_matrix(graph, attr = w, sparse = TRUE)
    }
    m <- methods::as(graph, "CsparseMatrix")
    m <- methods::as(methods::as(m, "generalMatrix"), "dMatrix")
    if (nrow(m) != ncol(m)) stop("graph must be a square (symmetric) matrix")
    m
}

#' Deterministic Leiden community detection
#'
#' Leiden (Traag, Waltman & van Eck 2019) maximising modularity at resolution
#' `resolution`, seeded and deterministic: the same `seed` gives the same
#' labels at any number of threads.
#'
#' @param x A graph or a `SingleCellExperiment`.
#'   * Graph: undirected and weighted, a symmetric sparse matrix (anything
#'     `methods::as(x, "CsparseMatrix")` accepts: a `dgCMatrix`, a Seurat
#'     `Graph` such as `obj@graphs$RNA_snn`, a symmetric `dsCMatrix`, a pattern
#'     matrix) or an igraph graph (edge attribute `weight` if present, e.g.
#'     from `scran::buildSNNGraph()` or `bluster::makeSNNGraph()`).
#'   * `SingleCellExperiment`: the cells are clustered on a shared nearest
#'     neighbour graph of `reducedDim(x, use.dimred)`, built by
#'     [bluster::makeSNNGraph()].
#' @param resolution Resolution of the modularity (gamma).
#' @param seed Seed, a non-negative whole number below 2^53.
#' @param n_iterations Leiden iterations; -1: until an iteration changes
#'   nothing.
#' @param theta Randomness of the refinement.
#' @param min_gain With `n_iterations = -1`, also stop once an iteration gains
#'   less modularity than this (0: strict).
#' @param threads Worker threads; `NULL`: all cores. Does not change the
#'   result.
#' @param check_symmetry Verify that the matrix is symmetric. `FALSE` skips
#'   only that check, for graphs symmetric by construction (Seurat, scanpy,
#'   bluster); bounds and weights are always validated.
#' @param use.dimred Name of the reduced dimension to build the graph from.
#' @param k Number of nearest neighbours for the SNN graph.
#' @param type Edge weighting of the SNN graph, as in
#'   [bluster::makeSNNGraph()]: `"rank"`, `"number"` or `"jaccard"`.
#' @param ... Arguments for the graph method (`resolution`, `seed`, ...).
#' @return For a graph: integer community labels `1..k`, community 1 the
#'   largest (ties: by smallest vertex), named by the graph's row names if it
#'   has them. For a `SingleCellExperiment`: the same labels as a factor named
#'   by cell, e.g. for `colLabels(sce) <- leiden(sce)`.
#' @seealso [cluster_libreville()] for igraph and bluster.
#' @examples
#' f <- system.file("extdata", "hao3k.mtx", package = "clusterLibreville")
#' g <- Matrix::readMM(f)
#' labels <- leiden(g, resolution = 1, seed = 0)
#' table(labels)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'     assays = list(counts = matrix(0, 2, 200)),
#'     reducedDims = list(PCA = rbind(matrix(rnorm(400), ncol = 4),
#'                                    matrix(rnorm(400, 5), ncol = 4))))
#' SingleCellExperiment::colLabels(sce) <- leiden(sce, seed = 1)
#' table(SingleCellExperiment::colLabels(sce))
#' @export
setGeneric("leiden", function(x, ...) standardGeneric("leiden"))

#' @rdname leiden
#' @export
setMethod("leiden", "ANY", function(x, resolution = 1, seed = 0,
                                    n_iterations = -1L, theta = 0.01,
                                    min_gain = 0, threads = NULL,
                                    check_symmetry = TRUE) {
    m <- .as_graph(x)
    stopifnot(seed >= 0, seed < 2^53, seed == floor(seed),
                is.null(threads) || threads >= 1)
    r <- leiden_rs(m@p, m@i, m@x, as.double(resolution), as.double(seed),
                    as.integer(n_iterations), as.double(theta),
                    as.double(min_gain),
                    if (is.null(threads)) 0L else as.integer(threads),
                    isTRUE(check_symmetry))
    if (!is.null(rownames(m))) names(r) <- rownames(m)
    r
})

#' @rdname leiden
#' @importFrom SingleCellExperiment reducedDim
#' @importFrom bluster makeSNNGraph
#' @importFrom stats setNames
#' @export
setMethod("leiden", "SingleCellExperiment", function(x, use.dimred = "PCA",
                                                        k = 10L, type = "rank",
                                                        ...) {
    g <- makeSNNGraph(reducedDim(x, use.dimred), k = k, type = type)
    labels <- leiden(g, check_symmetry = FALSE, ...) # symmetric by construction
    factor(setNames(labels, colnames(x)), levels = seq_len(max(labels)))
})

#' Modularity of a partition
#'
#' Newman's modularity at resolution `resolution`, summed in a fixed order
#' (the quantity [leiden()] maximises).
#'
#' @param graph Undirected weighted graph: a symmetric sparse matrix or an
#'   igraph graph, as for [leiden()].
#' @param membership Labels, one per vertex (any atomic type).
#' @param resolution Resolution of the modularity (gamma).
#' @return A number.
#' @examples
#' f <- system.file("extdata", "hao3k.mtx", package = "clusterLibreville")
#' g <- Matrix::readMM(f)
#' modularity(g, leiden(g))
#' @export
modularity <- function(graph, membership, resolution = 1) {
    m <- .as_graph(graph)
    stopifnot(length(membership) == nrow(m))
    codes <- match(membership, unique(membership)) - 1L
    modularity_rs(m@p, m@i, m@x, codes, as.double(resolution))
}

#' Deterministic Leiden as an igraph community function
#'
#' [leiden()] with igraph's `cluster_*()` interface: it returns an igraph
#' `communities` object, so it plugs into igraph and into bluster's graph
#' clustering, e.g.
#' `bluster::clusterRows(pcs, SNNGraphParam(cluster.fun = cluster_libreville))`.
#'
#' @param graph An undirected igraph graph (edge attribute `weight` if
#'   present).
#' @param ... Options of [leiden()]: `resolution`, `seed`, `n_iterations`,
#'   `theta`, `min_gain`, `threads`.
#' @return An igraph `communities` object (`igraph::membership()`,
#'   `igraph::modularity()` and `igraph::sizes()` work on it).
#' @examples
#' g <- igraph::make_graph("Zachary")
#' comm <- cluster_libreville(g, seed = 1)
#' igraph::sizes(comm)
#' @export
cluster_libreville <- function(graph, ...) {
    labels <- leiden(graph, ...)
    igraph::make_clusters(graph, membership = unname(labels),
                            algorithm = "clusterLibreville",
                            modularity = modularity(graph, labels))
}
