test_that("find_clusters returns FindClusters' data frame of leiden labels", {
    m <- planted()
    dimnames(m) <- list(paste0("c", seq_len(nrow(m))), paste0("c", seq_len(ncol(m))))
    cl <- find_clusters(m, resolution = c(0.5, 1), random.seed = 3, algorithm = 1,
                        group.singletons = TRUE)
    expect_identical(names(cl), c("res.0.5", "res.1"))
    expect_identical(rownames(cl), colnames(m))
    expect_identical(as.integer(cl$res.1), unname(leiden(m, resolution = 1, seed = 3)))
    expect_identical(as.integer(cl$res.0.5), unname(leiden(m, resolution = 0.5, seed = 3)))
})

test_that("with_libreville routes Seurat::FindClusters to find_clusters and restores it", {
    skip_if_not_installed("Seurat")
    m <- planted()
    dimnames(m) <- list(paste0("c", seq_len(nrow(m))), paste0("c", seq_len(ncol(m))))
    orig <- Seurat::FindClusters
    cl <- with_libreville(Seurat::FindClusters(m, resolution = c(0.5, 1), random.seed = 3), threads = 2)
    expect_identical(Seurat::FindClusters, orig)
    expect_identical(cl, find_clusters(m, resolution = c(0.5, 1), random.seed = 3))
    expect_error(with_libreville(Seurat::FindClusters(m, random.seed = 0.5)))
    expect_identical(Seurat::FindClusters, orig) # restored after an error too
})

test_that("group.singletons merges one-cell communities as Seurat does, deterministically", {
    g <- two_cliques()
    expect_identical(.group_singletons(g, c(1L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 3L)), rep(1:2, each = 5))
    g <- Matrix::bdiag(g, Matrix::sparseMatrix(1, 1, x = 0)) # vertex 11 isolated: its own community
    cl <- find_clusters(g, resolution = 1)
    expect_identical(as.integer(cl$res.1), c(rep(1:2, each = 5), 1L)) # no edges anywhere: the largest
    expect_identical(max(as.integer(find_clusters(g, group.singletons = FALSE)$res.0.8)), 3L)
})
