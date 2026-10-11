## two 5-cliques joined by one edge
two_cliques <- function() {
    e <- rbind(t(combn(1:5, 2)), t(combn(6:10, 2)), c(5, 6))
    Matrix::sparseMatrix(i = c(e[, 1], e[, 2]), j = c(e[, 2], e[, 1]), x = 1,
                            dims = c(10, 10))
}

## a planted partition: k blocks of size s, dense inside, sparse between
planted <- function(k = 8, s = 60, p_in = 0.2, p_out = 0.005) {
    set.seed(7)
    n <- k * s
    block <- rep(seq_len(k), each = s)
    ij <- which(upper.tri(matrix(0, n, n)), arr.ind = TRUE)
    p <- ifelse(block[ij[, 1]] == block[ij[, 2]], p_in, p_out)
    e <- ij[runif(nrow(ij)) < p, ]
    w <- runif(nrow(e), 0.5, 1.5)
    Matrix::sparseMatrix(i = c(e[, 1], e[, 2]), j = c(e[, 2], e[, 1]),
                            x = c(w, w), dims = c(n, n))
}

## the shared known-answer fixture (inst/extdata, from tests/fixtures)
hao3k <- function() {
    f <- function(x) system.file("extdata", x, package = "clusterLibreville")
    list(graph = Matrix::readMM(f("hao3k.mtx")),
            cases = utils::read.delim(f("hao3k_cases.tsv"),
                                    stringsAsFactors = FALSE),
            labels = utils::read.delim(f("hao3k_labels.tsv"),
                                    check.names = FALSE))
}
