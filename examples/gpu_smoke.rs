//! cargo run --release --features gpu-cuda --example gpu_smoke -- GRAPH.npz
//! Toolchain smoke test: row sums on the GPU vs the CPU; first call (includes kernel compile) and second call.
fn main() {
    let path = std::env::args().nth(1).expect("GRAPH.npz");
    let mut z = npyz::npz::NpzArchive::open(&path).unwrap();
    let mut ints = |k: &str| -> Vec<i64> {
        let p = z.by_name(k).unwrap().unwrap();
        if p.dtype().descr() == "'<i8'" { p.into_vec::<i64>().unwrap() } else { p.into_vec::<i32>().unwrap().into_iter().map(i64::from).collect() }
    };
    let indptr: Vec<usize> = ints("indptr").into_iter().map(|x| x as usize).collect();
    let indices: Vec<u32> = ints("indices").into_iter().map(|x| x as u32).collect();
    let p = z.by_name("data").unwrap().unwrap();
    let data: Vec<f64> = if p.dtype().descr() == "'<f4'" { p.into_vec::<f32>().unwrap().into_iter().map(f64::from).collect() } else { p.into_vec::<f64>().unwrap() };
    let g = detleiden::Graph::from_csr(&indptr, &indices, &data).unwrap();
    for call in 1..=3 {
        let t = std::time::Instant::now();
        let (s, up, k) = detleiden::gpu::row_sums_gpu(&g);
        let total = t.elapsed().as_secs_f64();
        let cpu: Vec<f64> = (0..g.n()).map(|v| data[indptr[v]..indptr[v + 1]].iter().sum()).collect();
        let err = s.iter().zip(&cpu).map(|(&a, &b)| (a as f64 - b).abs() / b.max(1e-12)).fold(0.0, f64::max);
        println!("call {call}: total {total:.3} s (upload {up:.3} s, kernel+read {k:.4} s) n {} max rel err vs f64 CPU {err:.2e}", s.len());
    }
}
