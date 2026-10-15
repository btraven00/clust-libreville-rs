#' leiden() with the interface of Seurat's FindClusters
#'
#' A drop-in for `Seurat::FindClusters` on a graph, for tools built on it
#' that accept another clustering function (e.g. a patched CHOIR:
#' `options(CHOIR.cluster_fun = clusterLibreville::find_clusters)`); for
#' those that do not, see [with_libreville()]. Takes the graph (a Seurat
#' `Graph`, as `obj@graphs$RNA_snn`, or any symmetric sparse matrix),
#' `resolution` (a vector gives one column each) and `random.seed` (the
#' [leiden()] seed), and returns what `FindClusters` returns for a graph: a
#' data frame with one factor column per resolution (`res.<r>`, labels
#' `1..k`, 1 the largest community), rows named by cell. As in Seurat,
#' `group.singletons` (default `TRUE`) merges each one-cell community into
#' the community it is best connected to (highest mean edge weight); Seurat
#' breaks ties at random, here the larger community wins. Seurat's other
#' options (`algorithm`, `method`, `n.start`, `n.iter`, ...) are accepted
#' and ignored: the clustering is always this Leiden.
#'
#' @param object A graph, as for [leiden()]; not a Seurat object.
#' @param resolution Resolution(s) of the modularity (gamma).
#' @param random.seed Seed, a non-negative whole number below 2^53.
#' @param group.singletons Merge one-cell communities into their best-connected
#'   community.
#' @param ... Ignored (Seurat's options).
#' @param threads Worker threads, as for [leiden()].
#' @return A data frame, one factor column per resolution.
#' @examples
#' g <- Matrix::readMM(system.file("extdata", "hao3k.mtx", package = "clusterLibreville"))
#' cl <- find_clusters(g, resolution = c(0.5, 1), random.seed = 1)
#' sapply(cl, nlevels)
#' @export
find_clusters <- function(object, resolution = 0.8, random.seed = 0, group.singletons = TRUE,
                          ..., threads = NULL) {
    if (inherits(object, "Seurat")) {
        stop("find_clusters: pass the graph (e.g. obj@graphs$RNA_snn), not the Seurat object")
    }
    cl <- lapply(resolution, function(r) {
        lab <- unname(leiden(object, resolution = r, seed = random.seed, threads = threads))
        if (isTRUE(group.singletons)) lab <- .group_singletons(object, lab)
        factor(lab, levels = seq_len(max(lab)))
    })
    as.data.frame(stats::setNames(cl, paste0("res.", resolution)),
                  row.names = colnames(object), optional = TRUE)
}

## Seurat's GroupSingletons: each one-cell community joins the community of
## highest mean edge weight to it (sum of its edges into the community / the
## community's size), among those with more than one cell. Seurat breaks ties
## at random; here the larger community wins (the lower label), so the
## result stays deterministic. Labels are then renumbered 1..k by size, ties
## by smallest member, as leiden() numbers them.
.group_singletons <- function(graph, lab) {
    sizes <- tabulate(lab)
    single <- which(sizes[lab] == 1L)
    keep <- which(sizes > 1L)
    if (length(single) == 0L || length(keep) == 0L) return(lab)
    m <- methods::as(graph, "CsparseMatrix")
    member <- Matrix::sparseMatrix(i = seq_along(lab), j = lab, x = 1, dims = c(length(lab), max(lab)))
    conn <- as.matrix(m[single, , drop = FALSE] %*% member[, keep, drop = FALSE])
    conn <- sweep(conn, 2L, sizes[keep], "/")
    lab[single] <- keep[max.col(conn, ties.method = "first")]
    first <- match(sort(unique(lab)), lab)
    sz <- tabulate(lab)[sort(unique(lab))]
    old <- sort(unique(lab))[order(-sz, first)]
    match(lab, old)
}

#' Run code with Seurat's FindClusters replaced by find_clusters()
#'
#' Evaluates `expr` with `Seurat::FindClusters` swapped for [find_clusters()]
#' and restores it afterwards (also on error). For tools that cluster
#' through `Seurat::FindClusters` and have no option to choose the clustering
#' function, such as CHOIR: `with_libreville(CHOIR::CHOIR(obj))` makes every
#' clustering in its tree deterministic Leiden, seeded with CHOIR's
#' `random_seed`.
#'
#' The function is replaced in Seurat's namespace, for the whole R session,
#' while `expr` runs.
#'
#' @param expr Code to run.
#' @param threads Worker threads for every clustering, as for [leiden()].
#' @return The value of `expr`.
#' @examplesIf requireNamespace("Seurat", quietly = TRUE)
#' g <- Matrix::readMM(system.file("extdata", "hao3k.mtx", package = "clusterLibreville"))
#' cl <- with_libreville(Seurat::FindClusters(g, resolution = 1, random.seed = 1))
#' table(cl$res.1)
#' @export
with_libreville <- function(expr, threads = NULL) {
    if (!requireNamespace("Seurat", quietly = TRUE)) stop("Seurat is not installed")
    orig <- utils::getFromNamespace("FindClusters", "Seurat")
    shim <- function(object, ...) find_clusters(object, ..., threads = threads)
    utils::assignInNamespace("FindClusters", shim, "Seurat")
    on.exit(utils::assignInNamespace("FindClusters", orig, "Seurat"), add = TRUE)
    expr
}
