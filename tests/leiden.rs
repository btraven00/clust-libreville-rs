use clust_libreville_rs::{leiden, modularity, Graph, Options};

mod common;
use common::{graph, planted};


fn clique(offset: usize, k: usize, e: &mut Vec<(usize, usize, f64)>) {
    for i in 0..k {
        for j in i + 1..k {
            e.push((offset + i, offset + j, 1.0));
        }
    }
}

const KARATE: [(usize, usize); 78] = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),(6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),(15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),(23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),(25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),(30,32),(30,33),(31,32),(31,33),(32,33)];

fn karate() -> Graph {
    let e: Vec<_> = KARATE.iter().map(|&(u, v)| (u, v, 1.0)).collect();
    graph(34, &e)
}


fn connected_within(n: usize, edges: &[(usize, usize, f64)], labels: &[u32]) -> bool {
    let mut adj = vec![vec![]; n];
    for &(u, v, _) in edges {
        if labels[u] == labels[v] {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    let mut seen = vec![false; n];
    let mut roots = std::collections::HashSet::new();
    for s in 0..n {
        if seen[s] {
            continue;
        }
        if !roots.insert(labels[s]) {
            return false; // a second component with the same label
        }
        let mut stack = vec![s];
        seen[s] = true;
        while let Some(x) = stack.pop() {
            for &y in &adj[x] {
                if !seen[y] {
                    seen[y] = true;
                    stack.push(y);
                }
            }
        }
    }
    true
}

#[test]
fn two_cliques_and_a_bridge() {
    let mut e = vec![];
    clique(0, 5, &mut e);
    clique(5, 5, &mut e);
    e.push((0, 5, 1.0));
    let l = leiden(&graph(10, &e), &Options::default());
    assert!(l[..5].iter().all(|&x| x == l[0]) && l[5..].iter().all(|&x| x == l[5]) && l[0] != l[5], "{l:?}");
}

#[test]
fn components_are_never_merged() {
    let mut e = vec![];
    for c in 0..3 {
        clique(4 * c, 4, &mut e);
    }
    let l = leiden(&graph(12, &e), &Options::default());
    assert_eq!(l, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]); // equal sizes: ordered by smallest member
}

#[test]
fn karate_reaches_the_optimum() {
    // igraph's best over 50 converged seeds is 0.41979 (the known optimum, 4 communities)
    let g = karate();
    let best = (0..20)
        .map(|s| modularity(&g, &leiden(&g, &Options { seed: s, ..Default::default() }), 1.0))
        .fold(0.0f64, f64::max);
    assert!(best > 0.4197, "best {best}");
    for s in 0..20 {
        let q = modularity(&g, &leiden(&g, &Options { seed: s, ..Default::default() }), 1.0);
        assert!(q > 0.40, "seed {s}: {q}");
    }
}

#[test]
fn modularity_matches_the_dense_formula() {
    let (n, e) = planted(7, 3, 8, 0.6, 0.1);
    let g = graph(n, &e);
    let l = leiden(&g, &Options::default());
    let mut a = vec![vec![0.0f64; n]; n];
    for &(u, v, w) in &e {
        a[u][v] += w;
        a[v][u] += w;
    }
    let k: Vec<f64> = a.iter().map(|r| r.iter().sum()).collect();
    let m2: f64 = k.iter().sum();
    let mut q = 0.0;
    for i in 0..n {
        for j in 0..n {
            if l[i] == l[j] {
                q += a[i][j] - k[i] * k[j] / m2;
            }
        }
    }
    assert!((q / m2 - modularity(&g, &l, 1.0)).abs() < 1e-12);
}

#[test]
fn same_seed_same_labels() {
    let (n, e) = planted(3, 6, 30, 0.3, 0.02);
    let g = graph(n, &e);
    let o = Options { seed: 42, ..Default::default() };
    assert_eq!(leiden(&g, &o), leiden(&g, &o));
    let greedy = Options { theta: 0.0, ..o };
    assert_eq!(leiden(&g, &greedy), leiden(&g, &greedy));
}

#[test]
fn communities_are_connected_on_random_graphs() {
    // Leiden's guarantee; sparse, noisy graphs where Louvain-style merges go wrong
    for seed in 0..200u64 {
        let (n, e) = planted(seed, 4, 12, 0.25, 0.05);
        let g = graph(n, &e);
        for theta in [0.01, 0.0] {
            let l = leiden(&g, &Options { seed, theta, ..Default::default() });
            assert!(connected_within(n, &e, &l), "seed {seed} theta {theta}: {l:?}");
            for n_iterations in [1, 2] {
                let l = leiden(&g, &Options { seed, theta, n_iterations, ..Default::default() });
                assert!(connected_within(n, &e, &l), "seed {seed} theta {theta} n_iterations {n_iterations}");
            }
        }
    }
}

#[test]
fn converged_is_a_fixed_point() {
    let (n, e) = planted(11, 5, 20, 0.3, 0.03);
    let g = graph(n, &e);
    let o = Options { seed: 5, ..Default::default() };
    let l = leiden(&g, &o);
    let q = modularity(&g, &l, 1.0);
    let one = leiden(&g, &Options { n_iterations: 1, ..o.clone() });
    assert!(q >= modularity(&g, &one, 1.0) - 1e-12);
}

#[test]
fn rejects_bad_input() {
    assert!(Graph::from_csr(&[0, 1, 1], &[1], &[1.0]).is_err()); // asymmetric
    assert!(Graph::from_csr(&[0, 1, 2], &[1, 0], &[1.0, -1.0]).is_err());
    assert!(Graph::from_csr(&[0, 1, 2], &[1, 5], &[1.0, 1.0]).is_err());
    assert!(Graph::from_csr(&[0, 1, 2], &[1, 0], &[1.0, 1.0]).is_ok());
}

#[test]
fn resolution_controls_granularity() {
    let g = karate();
    let k = |r: f64| *leiden(&g, &Options { resolution: r, ..Default::default() }).iter().max().unwrap() + 1;
    assert!(k(0.3) <= k(1.0) && k(1.0) <= k(3.0), "{} {} {}", k(0.3), k(1.0), k(3.0));
}
