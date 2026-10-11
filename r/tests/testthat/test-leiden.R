# two 5-cliques joined by one edge
two_cliques <- function() {
  e <- rbind(t(combn(1:5, 2)), t(combn(6:10, 2)), c(5, 6))
  Matrix::sparseMatrix(i = c(e[, 1], e[, 2]), j = c(e[, 2], e[, 1]), x = 1, dims = c(10, 10))
}

# a planted partition: k blocks of size s, dense inside, sparse between (deterministic)
planted <- function(k = 8, s = 60, p_in = 0.2, p_out = 0.005) {
  set.seed(7)
  n <- k * s
  block <- rep(seq_len(k), each = s)
  ij <- which(upper.tri(matrix(0, n, n)), arr.ind = TRUE)
  p <- ifelse(block[ij[, 1]] == block[ij[, 2]], p_in, p_out)
  e <- ij[runif(nrow(ij)) < p, ]
  w <- runif(nrow(e), 0.5, 1.5)
  Matrix::sparseMatrix(i = c(e[, 1], e[, 2]), j = c(e[, 2], e[, 1]), x = c(w, w), dims = c(n, n))
}

test_that("two cliques are the two communities, labelled 1..k", {
  expect_identical(leiden(two_cliques()), rep(1:2, each = 5))
})

test_that("labels do not depend on the number of threads", {
  m <- planted()
  a <- leiden(m, seed = 3, threads = 1)
  for (t in c(2, 4, 8)) expect_identical(leiden(m, seed = 3, threads = t), a)
  expect_identical(leiden(m, seed = 3), a)
})

test_that("the same seed gives the same labels; labels are ordered by size", {
  m <- planted()
  a <- leiden(m, seed = 11)
  expect_identical(leiden(m, seed = 11), a)
  expect_type(a, "integer")
  expect_true(all(diff(as.vector(table(a))) <= 0)) # community 1 the largest
  expect_equal(sort(unique(a)), seq_len(max(a)))
})

test_that("the planted partition is recovered", {
  m <- planted()
  a <- leiden(m, seed = 1)
  truth <- rep(1:8, each = 60)
  expect_equal(length(unique(a)), 8)
  expect_equal(nrow(unique(cbind(a, truth))), 8) # a bijection between labels and blocks
})

test_that("input formats agree: dsCMatrix, pattern matrix, igraph, row names", {
  m <- two_cliques()
  a <- leiden(m)
  expect_identical(leiden(Matrix::forceSymmetric(m)), a)
  expect_identical(leiden(m != 0), a) # ngCMatrix: weights 1
  skip_if_not_installed("igraph")
  g <- igraph::graph_from_adjacency_matrix(m, mode = "undirected", weighted = TRUE)
  expect_identical(leiden(g), a)
  dimnames(m) <- list(letters[1:10], letters[1:10])
  expect_identical(names(leiden(m)), letters[1:10])
})

test_that("resolution, n_iterations and min_gain are passed through", {
  m <- planted()
  expect_gt(length(unique(leiden(m, resolution = 5))), length(unique(leiden(m, resolution = 0.2))))
  one <- leiden(m, n_iterations = 1)
  expect_length(one, nrow(m))
  expect_length(leiden(m, min_gain = 1e-3), nrow(m))
})

test_that("modularity agrees with igraph, and leiden is at least as good as igraph's", {
  skip_if_not_installed("igraph")
  m <- planted()
  g <- igraph::graph_from_adjacency_matrix(m, mode = "undirected", weighted = TRUE)
  a <- leiden(m, seed = 2)
  expect_equal(modularity(m, a), igraph::modularity(g, a, weights = igraph::E(g)$weight), tolerance = 1e-12)
  expect_equal(modularity(m, a, resolution = 0.5),
               igraph::modularity(g, a, weights = igraph::E(g)$weight, resolution = 0.5), tolerance = 1e-12)
  set.seed(2)
  ig <- igraph::membership(igraph::cluster_leiden(g, objective_function = "modularity", n_iterations = -1))
  expect_gte(modularity(m, a), modularity(m, ig) - 1e-3)
})

test_that("invalid input is an error", {
  m <- two_cliques()
  m[1, 2] <- 2 # no longer symmetric
  expect_error(leiden(m), "symmetric")
  expect_error(leiden(Matrix::sparseMatrix(i = 1, j = 2, x = 1, dims = c(2, 3))), "square")
  expect_error(leiden(two_cliques(), seed = -1))
  expect_error(leiden(two_cliques(), threads = 0))
})
