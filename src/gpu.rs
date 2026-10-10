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
