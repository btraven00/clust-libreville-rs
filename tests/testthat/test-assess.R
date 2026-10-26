planted <- function(n = 600, k = 4, p_in = 0.08, p_out = 0.005, seed = 0) {
    set.seed(seed)
    truth <- rep(seq_len(k), each = n / k)
    iu <- which(upper.tri(matrix(0, n, n)), arr.ind = TRUE)
    same <- truth[iu[, 1]] == truth[iu[, 2]]
    keep <- runif(nrow(iu)) < ifelse(same, p_in, p_out)
    e <- iu[keep, , drop = FALSE]
    a <- Matrix::sparseMatrix(i = c(e[, 1], e[, 2]), j = c(e[, 2], e[, 1]), x = 1, dims = c(n, n))
    list(truth = truth, graph = a)
}

test_that("assess ranks the planted labels above shuffled ones", {
    p <- planted()
    t <- assess(p$graph, p$truth)
    s <- assess(p$graph, sample(p$truth))
    expect_gt(t$loglik, s$loglik)
    expect_lt(t$description_length, s$description_length)
    expect_gt(t$significance, s$significance)
    expect_gt(min(t$surprise), max(s$surprise))
    expect_equal(sum(t$size), 600)
})

test_that("neosbm keeps the planted labels and drops shuffled ones", {
    p <- planted()
    a <- neosbm(p$graph, p$truth, seed = NULL)
    b <- neosbm(p$graph, sample(p$truth), seed = NULL)
    expect_lt(a$defect[length(a$defect)], 0.05)
    expect_gt(b$defect[length(b$defect)], 0.5)
    expect_true(all(diff(b$defect) >= -1e-12))
})
