## Known answers shared with the Rust and Python tests: every binding must give
## the labels the Rust core gives (0-based in the fixture, 1-based in R).
test_that("hao3k: the labels of every case, at 1 and 4 threads", {
    fx <- hao3k()
    for (k in seq_len(nrow(fx$cases))) {
        cs <- fx$cases[k, ]
        want <- as.integer(fx$labels[[cs$name]]) + 1L
        for (t in c(1, 4)) {
            got <- leiden(fx$graph, resolution = cs$resolution, seed = cs$seed,
                            n_iterations = cs$n_iterations, theta = cs$theta,
                            min_gain = cs$min_gain, threads = t)
            expect_identical(got, want, label = paste(cs$name, t, "threads"))
        }
    }
})
