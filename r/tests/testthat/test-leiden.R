test_that("two cliques are the two communities, labelled 1..k", {
    expect_identical(leiden(two_cliques()), rep(1:2, each = 5))
})

test_that("labels do not depend on the number of threads", {
    m <- planted()
    a <- leiden(m, seed = 3, threads = 1)
    for (t in c(2, 4, 8)) {
        expect_identical(leiden(m, seed = 3, threads = t), a)
    }
    expect_identical(leiden(m, seed = 3), a)
})

test_that("the same seed gives the same labels, ordered by size", {
    m <- planted()
    a <- leiden(m, seed = 11)
    expect_identical(leiden(m, seed = 11), a)
    expect_type(a, "integer")
    expect_true(all(diff(as.vector(table(a))) <= 0)) # 1 is the largest
    expect_equal(sort(unique(a)), seq_len(max(a)))
})

test_that("the planted partition is recovered", {
    a <- leiden(planted(), seed = 1)
    truth <- rep(1:8, each = 60)
    expect_equal(length(unique(a)), 8)
    expect_equal(nrow(unique(cbind(a, truth))), 8) # labels <-> blocks
})

test_that("input formats agree: dsCMatrix, pattern, igraph, row names", {
    m <- two_cliques()
    a <- leiden(m)
    expect_identical(leiden(Matrix::forceSymmetric(m)), a)
    expect_identical(leiden(m != 0), a) # ngCMatrix: weights 1
    skip_if_not_installed("igraph")
    g <- igraph::graph_from_adjacency_matrix(m, mode = "undirected",
                                                weighted = TRUE)
    expect_identical(leiden(g), a)
    dimnames(m) <- list(letters[1:10], letters[1:10])
    expect_identical(names(leiden(m)), letters[1:10])
})

test_that("resolution, n_iterations and min_gain are passed through", {
    m <- planted()
    expect_gt(length(unique(leiden(m, resolution = 5))),
                length(unique(leiden(m, resolution = 0.2))))
    expect_length(leiden(m, n_iterations = 1), nrow(m))
    expect_length(leiden(m, min_gain = 1e-3), nrow(m))
})

test_that("modularity agrees with igraph; leiden is as good as igraph's", {
    skip_if_not_installed("igraph")
    m <- planted()
    g <- igraph::graph_from_adjacency_matrix(m, mode = "undirected",
                                                weighted = TRUE)
    w <- igraph::E(g)$weight
    a <- leiden(m, seed = 2)
    expect_equal(modularity(m, a), igraph::modularity(g, a, weights = w),
                    tolerance = 1e-12)
    expect_equal(modularity(m, a, resolution = 0.5),
                    igraph::modularity(g, a, weights = w, resolution = 0.5),
                    tolerance = 1e-12)
    set.seed(2)
    ig <- igraph::cluster_leiden(g, objective_function = "modularity",
                                    n_iterations = -1)
    expect_gte(modularity(m, a), modularity(m, igraph::membership(ig)) - 1e-3)
})

test_that("invalid input is an error", {
    m <- two_cliques()
    m[1, 2] <- 2 # no longer symmetric
    expect_error(leiden(m), "symmetric")
    expect_error(leiden(Matrix::sparseMatrix(i = 1, j = 2, x = 1,
                                                dims = c(2, 3))), "square")
    expect_error(leiden(two_cliques(), seed = -1))
    expect_error(leiden(two_cliques(), threads = 0))
})

test_that("check_symmetry = FALSE skips only the symmetry check", {
    m <- planted()
    expect_identical(leiden(m, seed = 6, check_symmetry = FALSE),
                     leiden(m, seed = 6))
    a <- two_cliques()
    a[1, 2] <- 2 # asymmetric: meaningless labels, but no error
    expect_length(leiden(a, check_symmetry = FALSE), 10)
    expect_error(leiden(a), "symmetric")
})
