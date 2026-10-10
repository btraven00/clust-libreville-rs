use detleiden::Graph;

#[allow(dead_code)]
pub fn graph(n: usize, edges: &[(usize, usize, f64)]) -> Graph {
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for &(u, v, w) in edges {
        rows[u].push((v as u32, w));
        if u != v {
            rows[v].push((u as u32, w));
        }
    }
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

/// A planted-partition graph from a seeded LCG: k groups of `size`, weights in (0.1, 1].
#[allow(dead_code)]
pub fn planted(seed: u64, k: usize, size: usize, p_in: f64, p_out: f64) -> (usize, Vec<(usize, usize, f64)>) {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
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
