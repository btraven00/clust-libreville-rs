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

test_that("SingleCellExperiment workflow: colLabels from an SNN graph", {
    skip_if_not_installed("SingleCellExperiment")
    skip_if_not_installed("bluster")
    set.seed(1)
    pcs <- rbind(matrix(rnorm(400, 0), ncol = 4),
                    matrix(rnorm(400, 6), ncol = 4))
    sce <- SingleCellExperiment::SingleCellExperiment(
        assays = list(counts = matrix(0, 3, 200)),
        reducedDims = list(PCA = pcs))
    g <- bluster::makeSNNGraph(SingleCellExperiment::reducedDim(sce, "PCA"))
    SingleCellExperiment::colLabels(sce) <- factor(leiden(g, seed = 1))
    expect_equal(nlevels(SingleCellExperiment::colLabels(sce)), 2)
})
