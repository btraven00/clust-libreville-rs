//! cargo run --release --example ensemble -- GRAPH.npz [--workers W] [--runs R]
//! A shared-memory ensemble (as in Markov stability: scales × seeds over one graph): W OS
//! threads each run leiden with `threads: Some(1)` R times. Prints wall time and runs/s.

#[path = "../tests/common/mod.rs"]
mod common;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str, d: usize| args.iter().position(|a| a == name).map_or(d, |i| args[i + 1].parse().unwrap());
    let (workers, runs) = (flag("--workers", 8), flag("--runs", 20));
    let g = common::load_npz(&args[0]).expect("open npz");
    let t = std::time::Instant::now();
    std::thread::scope(|s| {
        for w in 0..workers {
            let g = &g;
            s.spawn(move || {
                for r in 0..runs {
                    let o = clust_libreville_rs::Options { seed: (w * runs + r) as u64, resolution: 0.5 + r as f64 / runs as f64, threads: Some(1), ..Default::default() };
                    std::hint::black_box(clust_libreville_rs::leiden(g, &o));
                }
            });
        }
    });
    let s = t.elapsed().as_secs_f64();
    println!("n {} workers {workers} runs {} wall {s:.3} s ({:.1} runs/s)", g.n(), workers * runs, (workers * runs) as f64 / s);
}
