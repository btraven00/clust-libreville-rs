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

test_that("details = TRUE reports how the run ended", {
    m <- planted()
    d <- leiden(m, seed = 1, details = TRUE)
    expect_identical(d$labels, leiden(m, seed = 1))
    expect_true(d$converged)
    expect_identical(d$stop, "converged")
    expect_equal(d$modularity, modularity(m, d$labels), tolerance = 1e-12)
    one <- leiden(m, seed = 1, n_iterations = 1, details = TRUE)
    expect_identical(one$iterations, 1L)
    expect_false(one$converged)
    expect_identical(one$stop, "n_iterations")
})

test_that("champ: the envelope is the pool's best, from its grid points", {
    m <- planted()
    res <- 10^seq(-1, 1, length.out = 8)
    ch <- champ(m, res, seeds = 0:1)
    d <- ch$domains
    expect_gte(nrow(d), 2)
    expect_equal(c(d$gamma_lo[1], d$gamma_hi[nrow(d)]), range(res))
    expect_identical(d$gamma_hi[-nrow(d)], d$gamma_lo[-1])
    for (j in seq_len(nrow(d))) { # provenance
        expect_identical(ch$labels[, j],
                         leiden(m, resolution = d$resolution[j], seed = d$seed[j]))
    }
    pool <- lapply(res, function(r) lapply(0:1, function(s) leiden(m, resolution = r, seed = s)))
    pool <- unlist(pool, recursive = FALSE)
    for (x in 10^seq(-1, 1, length.out = 30)) {
        best <- max(vapply(pool, function(p) modularity(m, p, x), 0))
        j <- which(d$gamma_lo <= x & x <= d$gamma_hi)[1]
        expect_gte(modularity(m, ch$labels[, j], x), best - 1e-12)
    }
    expect_identical(champ(m, res, 0:1, threads = 1), champ(m, res, 0:1, threads = 4))
    expect_error(champ(m, c(0, 1)))
    expect_error(champ(m, 1, seeds = -1))
})

test_that("champ: agreement with neighbours and seeds, NVI or AMI", {
    skip_if_not_installed("aricode")
    m <- planted()
    res <- 10^seq(-1, 1, length.out = 8)
    for (metric in c("nvi", "ami")) {
        ch <- champ(m, res, seeds = 0:2, metric = metric)
        d <- ch$domains
        ref <- if (metric == "nvi") aricode::NVI else aricode::AMI
        expect_true(is.na(d$to_prev[1]))
        for (j in seq_len(nrow(d))[-1]) {
            expect_equal(d$to_prev[j], ref(ch$labels[, j - 1], ch$labels[, j]),
                         tolerance = 1e-9)
        }
        j <- ch$best
        mates <- lapply(setdiff(0:2, d$seed[j]), function(s)
            leiden(m, resolution = d$resolution[j], seed = s))
        expect_equal(d$to_seeds[j],
                     mean(vapply(mates, function(l) ref(ch$labels[, j], l), 0)),
                     tolerance = 1e-9)
    }
    expect_error(champ(m, res, metric = "ari"))
})
