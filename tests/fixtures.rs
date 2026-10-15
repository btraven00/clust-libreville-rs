//! Known-answer fixtures shared with the R and Python tests (tests/fixtures, made by
//! bench/py/make_fixtures.py): every binding must give these labels for these options.

use std::fs;

use clust_libreville_rs::{leiden, Apply, Graph, Options};

mod common;
use common::fnv;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

/// A symmetric coordinate MatrixMarket file (one triangle stored) -> Graph.
fn read_mtx(path: &str) -> Graph {
    let text = fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('%'));
    let n: usize = lines.next().unwrap().split_whitespace().next().unwrap().parse().unwrap();
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for l in lines {
        let f: Vec<&str> = l.split_whitespace().collect();
        let (i, j, w): (usize, usize, f64) = (f[0].parse::<usize>().unwrap() - 1, f[1].parse::<usize>().unwrap() - 1, f[2].parse().unwrap());
        rows[i].push((j as u32, w));
        if i != j {
            rows[j].push((i as u32, w));
        }
    }
    let (mut indptr, mut indices, mut data) = (vec![0usize], Vec::new(), Vec::new());
    for mut r in rows {
        r.sort_by_key(|e| e.0);
        indices.extend(r.iter().map(|e| e.0));
        data.extend(r.iter().map(|e| e.1));
        indptr.push(indices.len());
    }
    Graph::from_csr(&indptr, &indices, &data).unwrap()
}

#[test]
fn hao3k_known_answers() {
    let g = read_mtx(&format!("{DIR}hao3k.mtx"));
    let cases = fs::read_to_string(format!("{DIR}hao3k_cases.tsv")).unwrap();
    let labels = fs::read_to_string(format!("{DIR}hao3k_labels.tsv")).unwrap();
    let mut lines = labels.lines();
    let names: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let cols: Vec<Vec<u32>> = {
        let rows: Vec<Vec<u32>> = lines.map(|l| l.split('\t').map(|x| x.parse().unwrap()).collect()).collect();
        (0..names.len()).map(|c| rows.iter().map(|r| r[c]).collect()).collect()
    };
    for case in cases.lines().skip(1) {
        let f: Vec<&str> = case.split('\t').collect();
        let o = Options {
            seed: f[1].parse().unwrap(),
            resolution: f[2].parse().unwrap(),
            n_iterations: f[3].parse().unwrap(),
            theta: f[4].parse().unwrap(),
            min_gain: f[5].parse().unwrap(),
            ..Default::default()
        };
        let want = &cols[names.iter().position(|&n| n == f[0]).unwrap()];
        for threads in [1, 4] {
            assert_eq!(&leiden(&g, &Options { threads: Some(threads), ..o.clone() }), want, "case {} at {threads} threads", f[0]);
        }
    }
}

/// `Apply::OrderFree` (the GPU reference) on the same fixture: a regression lock on the labels of
/// three cases (the hashes were taken from the code, not from an external reference).
#[test]
fn hao3k_orderfree_lock() {
    let g = read_mtx(&format!("{DIR}hao3k.mtx"));
    let lock = [("default", 0u64, -1i64, 0x75ae09d42476b718), ("seed42", 42, -1, 0xbffd47de8e628d0a), ("niter1", 7, 1, 0x9dfe3b522c5f1d63)];
    for (name, seed, n_iterations, want) in lock {
        for threads in [1, 4] {
            let o = Options { seed, n_iterations, apply: Apply::OrderFree, threads: Some(threads), ..Default::default() };
            let got = fnv(&leiden(&g, &o));
            assert_eq!(got, want, "{name} OrderFree at {threads} threads: {got:#018x}");
        }
    }
}
