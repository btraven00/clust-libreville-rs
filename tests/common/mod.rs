//! Graph builders and loaders shared by the integration tests, the examples and the bench
//! (`#[path = "../tests/common/mod.rs"] mod common;` outside tests/).
#![allow(dead_code)] // each user picks what it needs

use std::path::Path;

use clust_libreville_rs::Graph;

/// A graph from undirected edges (each listed once; a self-loop once), rows in insertion order.
pub fn graph(n: usize, edges: &[(usize, usize, f64)]) -> Graph {
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for &(u, v, w) in edges {
        rows[u].push((v as u32, w));
        if u != v {
            rows[v].push((u as u32, w));
        }
    }
    from_rows(rows)
}

fn from_rows(rows: Vec<Vec<(u32, f64)>>) -> Graph {
    let mut indptr = vec![0];
    let (mut idx, mut dat) = (vec![], vec![]);
    for r in rows {
        for (v, w) in r {
            idx.push(v);
            dat.push(w);
        }
        indptr.push(idx.len());
    }
    Graph::from_csr(&indptr, &idx, &dat).unwrap()
}

/// A seeded LCG in [0, 1).
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A planted-partition graph, O(n²): k groups of `size`, weights in (0.1, 1]. Small graphs only.
pub fn planted(seed: u64, k: usize, size: usize, p_in: f64, p_out: f64) -> (usize, Vec<(usize, usize, f64)>) {
    let mut next = lcg(seed);
    let n = k * size;
    let mut e = vec![];
    for u in 0..n {
        for v in u + 1..n {
            let p = if u / size == v / size { p_in } else { p_out };
            if next() < p {
                e.push((u, v, 0.1 + 0.9 * next()));
            }
        }
    }
    (n, e)
}

/// A planted-partition graph with kNN-like rows, O(n·deg): `n` vertices in `k` blocks, each
/// drawing `deg` targets, a fraction `p_in` of them inside its block; plus `hubs` vertices that
/// draw `hub_deg` targets anywhere (rows longer than the GPU kernels' per-thread paths).
/// Weights in (0.1, 1], exactly f32 (scanpy's dtype); rows sorted, repeated edges summed, so
/// `Graph::from_csr` takes its fast symmetry path.
pub fn sparse_planted(seed: u64, n: usize, k: usize, deg: usize, p_in: f64, hubs: usize, hub_deg: usize) -> Graph {
    let mut next = lcg(seed);
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let block = n.div_ceil(k);
    let add = |rows: &mut Vec<Vec<(u32, f64)>>, u: usize, v: usize, w: f64| {
        if u != v {
            rows[u].push((v as u32, w));
            rows[v].push((u as u32, w));
        }
    };
    for u in 0..n {
        for _ in 0..deg {
            let v = if next() < p_in { (u / block) * block + (next() * block as f64) as usize } else { (next() * n as f64) as usize };
            let w = (0.1 + 0.9 * next()) as f32 as f64;
            add(&mut rows, u, v.min(n - 1), w);
        }
    }
    for u in 0..hubs.min(n) {
        for _ in 0..hub_deg {
            let v = (next() * n as f64) as usize;
            add(&mut rows, u, v, (0.1 + 0.9 * next()) as f32 as f64);
        }
    }
    for r in rows.iter_mut() {
        r.sort_by_key(|e| e.0);
        r.dedup_by(|b, a| {
            let same = a.0 == b.0;
            if same {
                a.1 += b.1;
            }
            same
        });
    }
    from_rows(rows)
}

/// FNV-1a of labels: a short lock for known-answer tests.
pub fn fnv(labels: &[u32]) -> u64 {
    labels.iter().flat_map(|x| x.to_le_bytes()).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// A full symmetric CSR from an .npz (indptr, indices, data; as bench/py/export_graph.py writes
/// it), validated. None if the file can't be opened.
pub fn load_npz(path: impl AsRef<Path>) -> Option<Graph> {
    let (indptr, indices, data) = read_npz(path)?;
    Some(Graph::from_csr(&indptr, &indices, &data).expect("valid graph"))
}

/// The raw CSR arrays of an .npz (int32 or int64 indices, float32 or float64 weights).
pub fn read_npz(path: impl AsRef<Path>) -> Option<(Vec<usize>, Vec<u32>, Vec<f64>)> {
    let mut z = npyz::npz::NpzArchive::open(path).ok()?;
    let mut ints = |k: &str| -> Vec<i64> {
        let p = z.by_name(k).unwrap().unwrap_or_else(|| panic!("no '{k}' in the npz"));
        if p.dtype().descr() == "'<i8'" { p.into_vec::<i64>().unwrap() } else { p.into_vec::<i32>().unwrap().into_iter().map(i64::from).collect() }
    };
    let indptr: Vec<usize> = ints("indptr").into_iter().map(|x| x as usize).collect();
    let indices: Vec<u32> = ints("indices").into_iter().map(|x| x as u32).collect();
    let p = z.by_name("data").unwrap().expect("no 'data' in the npz");
    let data: Vec<f64> = if p.dtype().descr() == "'<f4'" { p.into_vec::<f32>().unwrap().into_iter().map(f64::from).collect() } else { p.into_vec::<f64>().unwrap() };
    Some((indptr, indices, data))
}

/// bench/graphs/<name>.npz (git-ignored; `DETLEIDEN_GRAPHS` names another directory), or None if
/// it isn't there.
pub fn bench_graph(name: &str) -> Option<Graph> {
    let dir = std::env::var("DETLEIDEN_GRAPHS").unwrap_or_else(|_| format!("{}/bench/graphs", env!("CARGO_MANIFEST_DIR")));
    load_npz(format!("{dir}/{name}.npz"))
}
