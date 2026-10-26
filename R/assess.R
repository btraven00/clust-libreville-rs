#' Statistics of a given partition on a graph
#'
#' How well do these labels describe this graph, with the labels taken as
#' fixed (an annotation, a clustering from another modality)? Every quantity
#' is a closed form of the block edge-count matrix, one pass over the edges.
#'
#' * `loglik`: the degree-corrected SBM profile log-likelihood with the labels
#'   as blocks (Karrer & Newman 2011), the statistic of BESTest (Peel,
#'   Larremore & Clauset 2017), whose p-value is a permutation of the labels
#'   the caller runs.
#' * `description_length`: Peixoto's (2017) microcanonical DC-SBM description
#'   length with uniform priors, in nats, as graph-tool's
#'   `BlockState.entropy(degree_dl_kind = "uniform")`. It charges for the
#'   block matrix and the partition, so it ranks nested annotations: the
#'   shorter, the better the graph supports that level.
#' * `significance`: Traag, Krings & Van Dooren (2013), nats; leidenalg's
#'   `SignificanceVertexPartition.quality()`.
#' * `surprise`: per label, -ln P(X >= inner edges) with X Poisson at the
#'   configuration-model expectation: the label's cohesion beyond chance
#'   (Lancichinetti, Radicchi & Ramasco 2010, edge-count form).
#'
#' Exact on unweighted graphs; weights count as multiplicities.
#'
#' @param graph An undirected graph: a symmetric sparse matrix (e.g. an SNN
#'   graph's adjacency) or an igraph graph.
#' @param membership Labels, one per vertex (any atomic type).
#' @param check_symmetry Validate symmetry (`FALSE` for graphs symmetric by
#'   construction).
#' @return A list: `loglik`, `description_length`, `significance` (numbers);
#'   `labels` (the distinct labels, in order of first appearance) and, per
#'   label, `surprise`, `size`, `degree_sum`, `inner_edges`.
#' @references
#' Peel, Larremore & Clauset (2017) Sci. Adv. 3:e1602548; Peixoto (2017) Phys.
#' Rev. E 95:012317; Traag, Krings & Van Dooren (2013) Sci. Rep. 3:2930;
#' Karrer & Newman (2011) Phys. Rev. E 83:016107; Lancichinetti, Radicchi &
#' Ramasco (2010) Phys. Rev. E 81:046110.
#' @examples
#' f <- system.file("extdata", "hao3k.mtx", package = "clusterLibreville")
#' g <- Matrix::readMM(f)
#' a <- assess(g, leiden(g))
#' a$description_length
#' @export
assess <- function(graph, membership, check_symmetry = TRUE) {
    m <- .as_graph(graph)
    stopifnot(length(membership) == m$n)
    labels <- unique(membership)
    codes <- match(membership, labels) - 1L
    r <- assess_rs(m$p, m$i, m$x, codes, isTRUE(check_symmetry))
    r$labels <- labels
    r
}

#' neoSBM: how many vertices leave an annotation when leaving gets cheaper
#'
#' Peel, Larremore & Clauset's (2017) neoSBM: every vertex keeps its label's
#' block or is free to sit in any block, at a prior cost of
#' `log((1 - theta) / theta)` nats of degree-corrected SBM likelihood.
#' Sweeping `theta` towards 1/2 traces the fraction of vertices that defect
#' from the annotation; at `theta = 1/2` the prior is neutral and that
#' fraction is "vertices whose label is not locally optimal for this graph".
#' Greedy single-vertex moves with exact likelihood deltas, warm-started along
#' `thetas`; a maximiser of the published objective, not Peel's reference
#' code.
#'
#' @inheritParams assess
#' @param thetas The grid, in (0, 1), ascending.
#' @param max_sweeps Sweeps per theta before giving up on convergence.
#' @param seed Fixes the vertex order of the sweeps; `NULL` sweeps in vertex
#'   order (the deterministic choice for cross-checks).
#' @return A list: `thetas`; per theta `defect` (fraction off their label),
#'   `loglik_gain` (nats recovered) and `sweeps`; `blocks` (the block of every
#'   vertex at the last theta, as indices into `labels`); `labels`.
#' @examples
#' f <- system.file("extdata", "hao3k.mtx", package = "clusterLibreville")
#' g <- Matrix::readMM(f)
#' r <- neosbm(g, leiden(g))
#' r$defect
#' @export
neosbm <- function(graph, membership, thetas = c(1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.4, 0.5),
                   max_sweeps = 20L, seed = 0L, check_symmetry = TRUE) {
    m <- .as_graph(graph)
    stopifnot(length(membership) == m$n, all(thetas > 0 & thetas < 1))
    labels <- unique(membership)
    codes <- match(membership, labels) - 1L
    order <- if (is.null(seed)) seq_len(m$n) - 1L else { set.seed(seed); sample.int(m$n) - 1L }
    r <- neosbm_rs(m$p, m$i, m$x, codes, as.double(sort(thetas)), as.integer(max_sweeps), as.integer(order), isTRUE(check_symmetry))
    r$blocks <- r$blocks + 1L
    r$thetas <- sort(thetas)
    r$labels <- labels
    r
}
