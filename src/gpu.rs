//! GPU backend (cubecl), feature-gated. Step 2a: toolchain smoke test only (row sums).
//! Contract (a) of docs/road-to-cubecl.md: deterministic per backend, no float atomics; f32 on
//! devices without f64 (Metal, WGSL).

use cubecl::prelude::*;

use crate::graph::Repr;
use crate::Graph;

/// Sum of row v, in CSR order: one thread per row, so the order is fixed.
#[cube(launch_unchecked)]
fn row_sums(offsets: &[u32], weights: &[f32], out: &mut [f32]) {
    let v = ABSOLUTE_POS;
    if v < out.len() {
        let mut s = f32::new(0.0f32);
        for i in offsets[v] as usize..offsets[v + 1] as usize {
            s += weights[i];
        }
        out[v] = s;
    }
}

/// Node strengths computed on the default device, plus (upload, kernel+read) seconds.
#[doc(hidden)]
pub fn row_sums_gpu(g: &Graph) -> (Vec<f32>, f64, f64) {
    let Repr::F32(c) = &g.0 else { panic!("smoke test needs f32-exact weights") };
    let device = cubecl::Device::default();
    let client = device.client();
    let n = c.n();
    let t = std::time::Instant::now();
    let off: Vec<u32> = c.offsets.iter().map(|&o| o as u32).collect();
    let h_off = client.create_from_slice(u32::as_bytes(&off));
    let h_w = client.create_from_slice(f32::as_bytes(&c.weights));
    let h_out = client.empty(n * std::mem::size_of::<f32>());
    let up = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let block = 256u32;
    unsafe {
        row_sums::launch_unchecked(
            &client,
            CubeCount::Static((n as u32).div_ceil(block), 1, 1),
            CubeDim::new_1d(block),
            BufferArg::from_raw_parts(h_off, off.len()),
            BufferArg::from_raw_parts(h_w, c.weights.len()),
            BufferArg::from_raw_parts(h_out.clone(), n),
        )
    };
    let bytes = client.read_one(h_out).unwrap();
    let out = f32::from_bytes(&bytes).to_vec();
    (out, up, t.elapsed().as_secs_f64())
}

/// Up to this many distinct neighbour communities per vertex fit the decide kernel's local
/// arrays; a vertex with more is flagged and decided on the CPU (rare: degree > 64 is ~1%).
const MAXC: usize = 64;
const STAY: u32 = 0xFFFF_FFFD;
const FALLBACK: u32 = 0xFFFF_FFFE;
// NEW = u32::MAX, as in moving.rs

/// The decide step on the GPU, one thread per vertex: an exact mirror of `moving::best_move`
/// (same neighbour order, same f64 sums per community, same candidate order and strict `>`),
/// so on a device with f64 the decisions should equal the CPU's bit for bit.
#[cube(launch_unchecked)]
fn decide(offsets: &[u32], targets: &[u32], weights: &[f32], strength: &[f64], comm: &[u32], tot: &[f64], size: &[u32],
          scale: f64, out_to: &mut [u32], out_wto: &mut [f64], out_wfrom: &mut [f64]) {
    let v = ABSOLUTE_POS;
    if v < out_to.len() {
        let mut cs = Array::<u32>::new(MAXC);
        let mut ws = Array::<f64>::new(MAXC);
        let mut nt = 0usize;
        let mut overflow = false;
        let own = comm[v];
        for i in offsets[v] as usize..offsets[v + 1] as usize {
            let u = targets[i];
            if u != v as u32 {
                let c = comm[u as usize];
                let w = f64::cast_from(weights[i]);
                let mut found = false;
                for j in 0..nt {
                    if !found && cs[j] == c {
                        ws[j] += w;
                        found = true;
                    }
                }
                if !found {
                    if nt < MAXC {
                        cs[nt] = c;
                        ws[nt] = w;
                        nt += 1;
                    } else {
                        overflow = true;
                    }
                }
            }
        }
        let k = strength[v];
        let mut w_from = f64::new(0.0f32);
        for j in 0..nt {
            if cs[j] == own {
                w_from = ws[j];
            }
        }
        let mut best_c = own;
        let mut best_g = w_from - scale * k * (tot[own as usize] - k);
        let mut best_w = w_from;
        if size[own as usize] > 1 && f64::new(0.0f32) > best_g {
            best_c = u32::MAX;
            best_g = f64::new(0.0f32);
            best_w = f64::new(0.0f32);
        }
        for j in 0..nt {
            let c = cs[j];
            if c != own {
                let gain = ws[j] - scale * k * tot[c as usize];
                if gain > best_g {
                    best_c = c;
                    best_g = gain;
                    best_w = ws[j];
                }
            }
        }
        if overflow {
            out_to[v] = FALLBACK;
        } else if best_c == own {
            out_to[v] = STAY;
        } else {
            out_to[v] = best_c;
        }
        out_wto[v] = best_w;
        out_wfrom[v] = w_from;
    }
}

/// Decide for every vertex of `g` under the partition `comm`, on the GPU and on the CPU, and
/// compare. Returns (vertices, CPU moves, mismatches, fallbacks, GPU warm kernel+read seconds,
/// CPU single-thread seconds).
#[doc(hidden)]
pub fn check_decide(g: &Graph, comm: &[u32], gamma: f64) -> (usize, usize, usize, usize, f64, f64) {
    let Repr::F32(c) = &g.0 else { panic!("needs f32-exact weights") };
    let n = c.n();
    let scale = gamma / c.total;
    let mut tot = vec![0.0f64; n];
    let mut size = vec![0u32; n];
    for v in 0..n {
        tot[comm[v] as usize] += c.strength[v];
        size[comm[v] as usize] += 1;
    }
    // CPU reference
    let t = std::time::Instant::now();
    let (mut wt, mut touched) = (vec![0.0f64; n], Vec::new());
    let cpu: Vec<Option<crate::moving::Move>> =
        (0..n as u32).map(|v| crate::moving::best_move(c, v, comm, &tot, &size, scale, &mut wt, &mut touched)).collect();
    let cpu_s = t.elapsed().as_secs_f64();
    // GPU
    let client = cubecl::Device::default().client();
    let off: Vec<u32> = c.offsets.iter().map(|&o| o as u32).collect();
    let h = |b: &[u8]| client.create_from_slice(b);
    let (h_off, h_t, h_w, h_s) = (h(u32::as_bytes(&off)), h(u32::as_bytes(&c.targets)), h(f32::as_bytes(&c.weights)), h(f64::as_bytes(&c.strength)));
    let (h_c, h_tot, h_size) = (h(u32::as_bytes(comm)), h(f64::as_bytes(&tot)), h(u32::as_bytes(&size)));
    let (o_to, o_wto, o_wfrom) = (client.empty(n * 4), client.empty(n * 8), client.empty(n * 8));
    let block = 128u32;
    let launch = || unsafe {
        decide::launch_unchecked(
            &client,
            CubeCount::Static((n as u32).div_ceil(block), 1, 1),
            CubeDim::new_1d(block),
            BufferArg::from_raw_parts(h_off.clone(), off.len()),
            BufferArg::from_raw_parts(h_t.clone(), c.targets.len()),
            BufferArg::from_raw_parts(h_w.clone(), c.weights.len()),
            BufferArg::from_raw_parts(h_s.clone(), n),
            BufferArg::from_raw_parts(h_c.clone(), n),
            BufferArg::from_raw_parts(h_tot.clone(), n),
            BufferArg::from_raw_parts(h_size.clone(), n),
            scale,
            BufferArg::from_raw_parts(o_to.clone(), n),
            BufferArg::from_raw_parts(o_wto.clone(), n),
            BufferArg::from_raw_parts(o_wfrom.clone(), n),
        )
    };
    launch();
    let _ = client.read_one(o_to.clone()).unwrap(); // first launch: includes the JIT compile
    let t = std::time::Instant::now();
    launch();
    let to_b = client.read_one(o_to.clone()).unwrap();
    let gpu_s = t.elapsed().as_secs_f64();
    let wto_b = client.read_one(o_wto).unwrap();
    let wfrom_b = client.read_one(o_wfrom).unwrap();
    let (to, wto, wfrom) = (u32::from_bytes(&to_b), f64::from_bytes(&wto_b), f64::from_bytes(&wfrom_b));
    let (mut mism, mut fb, mut moves) = (0, 0, 0);
    for v in 0..n {
        if to[v] == FALLBACK {
            fb += 1;
            continue;
        }
        let gpu = (to[v] != STAY).then_some((to[v], wto[v].to_bits(), wfrom[v].to_bits()));
        let cpu_v = cpu[v].as_ref().map(|m| (m.to, m.w_to.to_bits(), m.w_from.to_bits()));
        moves += cpu_v.is_some() as usize;
        if gpu != cpu_v {
            if mism < 5 {
                eprintln!("  mismatch v {v}: gpu {gpu:?} cpu {cpu_v:?}");
            }
            mism += 1;
        }
    }
    (n, moves, mism, fb, gpu_s, cpu_s)
}
