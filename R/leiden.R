## Any supported graph -> a symmetric CSR as list(p, i, x, n, names) (0-based,
## like a dgCMatrix's slots: for a symmetric matrix its CSC is a CSR). An
## igraph graph goes through its edge list (Rust), not the much slower
## as_adjacency_matrix(): at 100k cells the SNN graph has ~20M edges.
.as_graph <- function(graph) {
    if (inherits(graph, "igraph")) {
        if (igraph::is_directed(graph)) stop("graph must be undirected")
        e <- igraph::as_edgelist(graph, names = FALSE)
        w <- if ("weight" %in% igraph::edge_attr_names(graph)) {
            as.double(igraph::E(graph)$weight)
        } else {
            rep(1, nrow(e))
        }
        n <- igraph::vcount(graph)
        csr <- edges_csr_rs(as.double(e[, 1]), as.double(e[, 2]), w, n)
        nm <- igraph::V(graph)$name
        return(c(csr, list(n = n, names = nm)))
    }
    m <- methods::as(graph, "CsparseMatrix")
    m <- methods::as(methods::as(m, "generalMatrix"), "dMatrix")
    if (nrow(m) != ncol(m)) stop("graph must be a square (symmetric) matrix")
    list(p = m@p, i = m@i, x = m@x, n = nrow(m), names = rownames(m))
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
#' @param details Also return how the run ended (see Value).
#' @param use.dimred Name of the reduced dimension to build the graph from.
#' @param k Number of nearest neighbours for the SNN graph.
#' @param type Edge weighting of the SNN graph, as in
#'   [bluster::makeSNNGraph()]: `"rank"`, `"number"` or `"jaccard"`.
#' @param ... Arguments for the graph method (`resolution`, `seed`, ...).
#' @return For a graph: integer community labels `1..k`, community 1 the
#'   largest (ties: by smallest vertex), named by the graph's row names if it
#'   has them. For a `SingleCellExperiment`: the same labels as a factor named
#'   by cell, e.g. for `colLabels(sce) <- leiden(sce)`. With `details = TRUE`,
#'   a list: `labels` (as above), `iterations` (Leiden iterations run),
#'   `converged` (whether the last iteration changed nothing: a fixed point),
#'   `stop` (why it stopped: `"converged"`, `"min_gain"` or `"n_iterations"`)
#'   and `modularity`.
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
                                    check_symmetry = TRUE, details = FALSE) {
    m <- .as_graph(x)
    stopifnot(seed >= 0, seed < 2^53, seed == floor(seed),
                is.null(threads) || threads >= 1)
    run <- leiden_rs(m$p, m$i, m$x, as.double(resolution), as.double(seed),
                    as.integer(n_iterations), as.double(theta),
                    as.double(min_gain),
                    if (is.null(threads)) 0L else as.integer(threads),
                    isTRUE(check_symmetry))
    r <- run$labels
    if (!is.null(m$names)) names(r) <- m$names
    if (!isTRUE(details)) return(r)
    list(labels = r, iterations = run$iterations,
         converged = run$stop == "converged", stop = run$stop,
         modularity = run$modularity)
})

#' @rdname leiden
#' @importFrom SingleCellExperiment reducedDim
#' @importFrom bluster makeSNNGraph
#' @importFrom stats setNames
#' @export
setMethod("leiden", "SingleCellExperiment", function(x, use.dimred = "PCA",
                                                        k = 10L, type = "rank",
                                                        details = FALSE, ...) {
    g <- makeSNNGraph(reducedDim(x, use.dimred), k = k, type = type)
    run <- leiden(g, check_symmetry = FALSE, details = TRUE, ...) # symmetric
    labels <- factor(setNames(run$labels, colnames(x)),
                     levels = seq_len(max(run$labels)))
    if (!isTRUE(details)) return(labels)
    run$labels <- labels
    run
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
    stopifnot(length(membership) == m$n)
    codes <- match(membership, unique(membership)) - 1L
    modularity_rs(m$p, m$i, m$x, codes, as.double(resolution))
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

#' Pick the resolution by CHAMP (experimental)
#'
#' Experimental: the interface and the scores may change.
#'
#' Runs [leiden()] at every pair of `resolutions` x `seeds`, then keeps the
#' partitions that have the highest modularity among that pool somewhere in
#' `gamma_range`, each with its interval of resolutions (CHAMP: Weir et al.
#' 2017, <doi:10.3390/a10030093>). Modularity is linear in the resolution, so
#' the intervals are exact; they are optimal relative to the pool, not
#' globally. Each partition is also scored by its agreement with its
#' neighbouring domains and with the other seeds, under `metric`.
#' Deterministic, at any number of threads.
#'
#' @param graph Undirected weighted graph: a symmetric sparse matrix or an
#'   igraph graph, as for [leiden()].
#' @param resolutions Grid of resolutions (gamma), all > 0.
#' @param seeds Seeds, non-negative whole numbers below 2^53.
#' @param gamma_range The range the intervals cover; `NULL`: the span of
#'   `resolutions`.
#' @param metric `"nvi"` (default), normalised variation of information: a
#'   distance (0 = identical) and a metric, so steps along the domains add
#'   up, as in Markov stability. `"ami"`, adjusted mutual information: a
#'   similarity (1 = identical), corrected for the chance agreement between
#'   partitions into many small communities.
#' @param n_iterations,theta,min_gain,threads,check_symmetry Options of
#'   [leiden()], for every run.
#' @return A list: `domains`, a data frame with one row per partition on the
#'   envelope in increasing resolution (`gamma_lo`, `gamma_hi`, `log_width`,
#'   `n_communities`, and the grid point that produced it: `resolution`,
#'   `seed`; and the agreement under `metric`: `to_prev` with the previous
#'   row's partition, `to_neighbours` the mean with both neighbours' (close:
#'   a plateau; far: a boundary), `to_seeds` the mean with the other seeds'
#'   partitions at the same grid resolution (algorithmic noise); `NA` where
#'   there is nothing to compare with); `labels`, an integer matrix (vertices x domains, labels `1..k`,
#'   community 1 the largest); and `best`, the row of `domains` with the
#'   widest interval in log resolution, e.g. `labels[, best]`.
#' @examples
#' f <- system.file("extdata", "hao3k.mtx", package = "clusterLibreville")
#' g <- Matrix::readMM(f)
#' ch <- champ(g, 10^seq(-1, 1, length.out = 10), seeds = 0:2)
#' ch$domains
#' table(ch$labels[, ch$best])
#' @export
champ <- function(graph, resolutions = 10^seq(-1, 1, length.out = 20),
                  seeds = 0:4, gamma_range = NULL, n_iterations = -1L,
                  theta = 0.01, min_gain = 0, threads = NULL,
                  check_symmetry = TRUE, metric = c("nvi", "ami")) {
    metric <- match.arg(metric)
    m <- .as_graph(graph)
    resolutions <- as.double(resolutions)
    seeds <- as.double(seeds)
    stopifnot(length(resolutions) > 0, all(is.finite(resolutions)),
                all(resolutions > 0), length(seeds) > 0, all(seeds >= 0),
                all(seeds < 2^53), all(seeds == floor(seeds)),
                is.null(threads) || threads >= 1)
    if (is.null(gamma_range)) gamma_range <- range(resolutions)
    stopifnot(length(gamma_range) == 2, gamma_range[1] > 0,
                gamma_range[1] <= gamma_range[2], is.finite(gamma_range[2]))
    run <- champ_rs(m$p, m$i, m$x, resolutions, seeds,
                    as.double(gamma_range[1]), as.double(gamma_range[2]),
                    as.integer(n_iterations), as.double(theta),
                    as.double(min_gain),
                    if (is.null(threads)) 0L else as.integer(threads),
                    isTRUE(check_symmetry), metric)
    labels <- matrix(run$labels, nrow = m$n)
    rownames(labels) <- m$names
    i_res <- run$index %/% length(seeds) + 1L
    i_seed <- run$index %% length(seeds) + 1L
    domains <- data.frame(gamma_lo = run$gamma_lo, gamma_hi = run$gamma_hi,
                            log_width = log(run$gamma_hi / run$gamma_lo),
                            n_communities = apply(labels, 2, max),
                            resolution = resolutions[i_res],
                            seed = seeds[i_seed],
                            to_prev = nan_na(run$to_prev),
                            to_neighbours = nan_na(run$to_neighbours),
                            to_seeds = nan_na(run$to_seeds))
    list(domains = domains, labels = labels,
         best = which.max(domains$log_width))
}

nan_na <- function(x) replace(x, is.nan(x), NA_real_)
