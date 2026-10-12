test_that("cluster_libreville is an igraph communities function", {
    skip_if_not_installed("igraph")
    m <- planted()
    g <- igraph::graph_from_adjacency_matrix(m, mode = "undirected",
                                                weighted = TRUE)
    comm <- cluster_libreville(g, seed = 4)
    expect_s3_class(comm, "communities")
    expect_identical(as.integer(igraph::membership(comm)), leiden(m, seed = 4))
    expect_equal(igraph::modularity(comm),
                    modularity(m, igraph::membership(comm)))
})

test_that("bluster: SNNGraphParam(cluster.fun = cluster_libreville)", {
    skip_if_not_installed("bluster")
    set.seed(1)
    pcs <- rbind(matrix(rnorm(400, 0), ncol = 4),
                    matrix(rnorm(400, 6), ncol = 4))
    p <- bluster::SNNGraphParam(cluster.fun = cluster_libreville,
                                cluster.args = list(seed = 2))
    cl <- bluster::clusterRows(pcs, p)
    expect_s3_class(cl, "factor")
    expect_length(cl, 200)
    expect_equal(nlevels(cl), 2)
    expect_identical(cl, bluster::clusterRows(pcs, p)) # deterministic
})

two_blobs_sce <- function() {
    set.seed(1)
    pcs <- rbind(matrix(rnorm(400, 0), ncol = 4),
                    matrix(rnorm(400, 6), ncol = 4))
    SingleCellExperiment::SingleCellExperiment(
        assays = list(counts = matrix(
            0, 3, 200, dimnames = list(NULL, paste0("c", 1:200)))),
        reducedDims = list(PCA = pcs))
}

test_that("leiden(<SingleCellExperiment>) clusters its SNN graph", {
    sce <- two_blobs_sce()
    cl <- leiden(sce, seed = 1)
    expect_s3_class(cl, "factor")
    expect_identical(names(cl), colnames(sce))
    expect_equal(nlevels(cl), 2)
    pcs <- SingleCellExperiment::reducedDim(sce, "PCA")
    g <- bluster::makeSNNGraph(pcs, k = 10, type = "rank")
    expect_identical(as.integer(cl), unname(leiden(g, seed = 1)))
    expect_identical(leiden(sce, seed = 1, threads = 1),
                        leiden(sce, seed = 1, threads = 4))
    SingleCellExperiment::colLabels(sce) <- cl
    expect_identical(SingleCellExperiment::colLabels(sce), cl)
    expect_error(leiden(sce, use.dimred = "UMAP"))
    d <- leiden(sce, seed = 1, details = TRUE)
    expect_identical(d$labels, cl)
    expect_true(d$converged)
})

test_that("igraph input via the edge list equals its adjacency matrix", {
    set.seed(3)
    pcs <- matrix(rnorm(3000), ncol = 10)
    g <- bluster::makeSNNGraph(pcs, k = 10, type = "rank")
    m <- igraph::as_adjacency_matrix(g, attr = "weight", sparse = TRUE)
    expect_identical(leiden(g, seed = 2), leiden(m, seed = 2))
    expect_equal(modularity(g, leiden(g)), modularity(m, leiden(g)))
    # repeated edges add up; a self-loop counts once; no weight attribute: 1
    h <- igraph::make_graph(c(1, 2, 1, 2, 2, 3, 3, 3, 3, 4), directed = FALSE)
    hm <- igraph::as_adjacency_matrix(h, sparse = TRUE)
    expect_identical(leiden(h), leiden(hm))
    expect_equal(modularity(h, c(1, 1, 2, 2)), modularity(hm, c(1, 1, 2, 2)))
})
