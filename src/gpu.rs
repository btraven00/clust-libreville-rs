//! GPU backend (cubecl), feature-gated: level-0 local moving on the device, everything else on
//! the CPU. It runs the OrderFree apply rule (`crate::Apply`), so its labels equal the CPU's
//! with `Apply::OrderFree`, bit for bit (on devices with f64: CUDA; Metal/WGSL lack f64).
//!
//! Per colour group: decide -> resolve -> link -> accept -> apply -> refresh -> mark, all on the
//! device; the host reads one move counter per round. No float atomics: totals and prefixes are
//! fixed-point integers (moving::FX), whose sums don't depend on order.

use std::borrow::Cow;

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::backend::{Backend, Cpu};
use crate::graph::{Csr, Repr, Weight};
use crate::moving::{self, FX};
use crate::{Apply, Graph};

/// Up to this many distinct neighbour communities per vertex fit the decide kernel's local
/// arrays; a vertex with more is a hub, decided by `decide_hubs`.
const MAXC: usize = 64;
const STAY: u32 = 0xFFFF_FFFD;
const HUB: u32 = 0xFFFF_FFFC;
/// Rows up to this long are decided by the hub kernel in shared memory (power of two).
const HUB_P: usize = 1024;
/// Cubes of the hub kernel (it grid-strides over however many hubs a group has).
const HUB_CUBES: u32 = 64;
const NONE: u32 = 0xFFFF_FFFF;
// moving::NEW = u32::MAX: a move to a new community (id n + v)
/// Below this many vertices local moving stays on the CPU (launch overheads win).
const GPU_MIN_N: usize = 50_000;
const BLOCK: u32 = 128;
/// Threads of the grid-stride per-group kernels (they loop over the group's affected list).
const GRID_CAP: usize = 16_384;
/// Jones–Plassmann rounds launched between two host reads of the frontier size.
const JP_BATCH: usize = 8;

/// Sum of row v, in CSR order: one thread per row, so the order is fixed. (Smoke test.)
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
    let client = cubecl::Device::default().client();
    let n = c.n();
    let t = std::time::Instant::now();
    let off: Vec<u32> = c.offsets.iter().map(|&o| o as u32).collect();
    let h_off = client.create_from_slice(u32::as_bytes(&off));
    let h_w = client.create_from_slice(f32::as_bytes(&c.weights));
    let h_out = client.empty(n * std::mem::size_of::<f32>());
    let up = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    unsafe {
        row_sums::launch_unchecked(
            &client,
            CubeCount::Static((n as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(h_off, off.len()),
            BufferArg::from_raw_parts(h_w, c.weights.len()),
            BufferArg::from_raw_parts(h_out.clone(), n),
        )
    };
    let bytes = client.read_one(h_out).unwrap();
    (f32::from_bytes(&bytes).to_vec(), up, t.elapsed().as_secs_f64())
}

/// The decide step, one thread per vertex of `verts` (skipping, and clearing, unaffected ones):
/// an exact mirror of `moving::best_move` (same neighbour order, same f64 sums per community,
/// same candidate order and strict `>`). Vertices with more than MAXC distinct neighbour
/// communities are listed for `decide_hubs`.
#[cube(launch_unchecked)]
fn decide(offsets: &[u32], targets: &[u32], weights: &[f32], strength: &[f64], comm: &[u32], tot: &[f64], size: &[u32],
          scale: f64, verts: &[u32], cur: &[u32], affected: &mut [u32], out_to: &mut [u32], out_wto: &mut [f64],
          out_wfrom: &mut [f64], hubs: &mut [u32], hub_count: &[Atomic<u32>]) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        let v = verts[i] as usize;
        if affected[v] == 0 {
            out_to[i] = STAY;
        } else {
            affected[v] = 0;
            let (start, end) = (offsets[v] as usize, offsets[v + 1] as usize);
            let own = comm[v];
            let k = strength[v];
            let mut cs = Array::<u32>::new(MAXC);
            let mut ws = Array::<f64>::new(MAXC);
            let mut nt = 0usize;
            let mut overflow = false;
            for e in start..end {
                let u = targets[e];
                if u != v as u32 && !overflow {
                    let c = comm[u as usize];
                    let w = f64::cast_from(weights[e]);
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
            if overflow {
                // too many distinct neighbour communities: the hub kernel decides this vertex
                out_to[i] = HUB;
                hubs[hub_count[0].fetch_add(1u32) as usize] = i as u32;
            } else {
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
                if best_c == own {
                    out_to[i] = STAY;
                } else {
                    out_to[i] = best_c;
                }
                out_wto[i] = best_w;
                out_wfrom[i] = w_from;
            }
        }
        i += stride;
    }
}

/// The decide step for hub vertices (more than MAXC distinct neighbour communities), one cube
/// per hub, grid-striding over the hubs decide listed. The (community, edge) keys of the row are
/// bitonic-sorted in shared memory, so each community's edges are contiguous and in CSR order;
/// each community is then summed in that order, as `moving::best_move` sums it. Selection: the
/// strictly best gain in first-occurrence order = the maximum gain, ties to the earliest first
/// occurrence. Rows longer than HUB_P (none in our graphs) fall back to an O(degree^2) rescan.
#[cube(launch_unchecked)]
fn decide_hubs(offsets: &[u32], targets: &[u32], weights: &[f32], strength: &[f64], comm: &[u32], tot: &[f64], size: &[u32],
               scale: f64, verts: &[u32], hubs: &[u32], hub_count: &[Atomic<u32>], out_to: &mut [u32], out_wto: &mut [f64],
               out_wfrom: &mut [f64]) {
    let mut keys = Shared::<[u64]>::new_slice(HUB_P);
    let count = hub_count[0].load() as usize;
    let unit = UNIT_POS as usize;
    let dim = CUBE_DIM as usize;
    let mut h = CUBE_POS as usize;
    while h < count {
        let i = hubs[h] as usize;
        let v = verts[i] as usize;
        let (start, end) = (offsets[v] as usize, offsets[v + 1] as usize);
        let deg = end - start;
        let own = comm[v];
        let k = strength[v];
        if deg <= HUB_P {
            let mut p = unit;
            while p < HUB_P {
                let mut key = !u64::new(0); // u64::MAX, as a runtime value
                if p < deg {
                    let u = targets[start + p];
                    if u != v as u32 {
                        key = (u64::cast_from(comm[u as usize]) << 32u64) | u64::cast_from(p);
                    }
                }
                keys[p] = key;
                p += dim;
            }
            sync_cube();
            let mut width = 2usize;
            while width <= HUB_P {
                let mut stride = width / 2;
                while stride > 0 {
                    let mut t = unit;
                    while t < HUB_P / 2 {
                        let a = 2 * stride * (t / stride) + t % stride;
                        let b = a + stride;
                        let (ka, kb) = (keys[a], keys[b]);
                        if (ka > kb) == ((a & width) == 0) {
                            keys[a] = kb;
                            keys[b] = ka;
                        }
                        t += dim;
                    }
                    sync_cube();
                    stride /= 2;
                }
                width *= 2;
            }
            if unit == 0 {
                // one serial pass over the sorted keys: each community's edges are contiguous and
                // in CSR order, so a running sum per run is the CPU's summation order
                let mut w_from = f64::new(0.0f32);
                for p in 0..HUB_P {
                    let kp = keys[p];
                    if (kp >> 32u64) as u32 == own && (kp >> 32u64) != 0xFFFF_FFFFu64 {
                        w_from += f64::cast_from(weights[start + (kp & 0xFFFF_FFFFu64) as usize]);
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
                let mut best_first = !u32::new(0); // first occurrence of the best candidate (MAX: own / new)
                let mut cur_c = 0xFFFF_FFFFu64;
                let mut cur_sum = f64::new(0.0f32);
                let mut cur_first = 0u32;
                for p in 0..HUB_P + 1 {
                    // position HUB_P is a virtual sentinel that closes the last run
                    let mut kp = !u64::new(0);
                    if p < HUB_P {
                        kp = keys[p];
                    }
                    let c = kp >> 32u64;
                    if c != cur_c {
                        if cur_c != 0xFFFF_FFFFu64 && cur_c as u32 != own {
                            // close the run of community cur_c: the strictly best gain in first-
                            // occurrence order = max gain, ties to the earliest first occurrence
                            let gain = cur_sum - scale * k * tot[cur_c as usize];
                            if gain > best_g || (gain == best_g && best_first != !u32::new(0) && cur_first < best_first) {
                                best_c = cur_c as u32;
                                best_g = gain;
                                best_w = cur_sum;
                                best_first = cur_first;
                            }
                        }
                        cur_c = c;
                        cur_sum = f64::new(0.0f32);
                        cur_first = (kp & 0xFFFF_FFFFu64) as u32;
                    }
                    if c != 0xFFFF_FFFFu64 {
                        cur_sum += f64::cast_from(weights[start + (kp & 0xFFFF_FFFFu64) as usize]);
                    }
                }
                if best_c == own {
                    out_to[i] = STAY;
                } else {
                    out_to[i] = best_c;
                }
                out_wto[i] = best_w;
                out_wfrom[i] = w_from;
            }
        } else if unit == 0 {
            // O(degree^2) rescan, in the same summation and candidate order
            let mut w_from = f64::new(0.0f32);
            for e in start..end {
                let u = targets[e];
                if u != v as u32 && comm[u as usize] == own {
                    w_from += f64::cast_from(weights[e]);
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
            for e in start..end {
                let u = targets[e];
                if u != v as u32 {
                    let c = comm[u as usize];
                    if c != own {
                        let mut first = true;
                        for e2 in start..e {
                            let u2 = targets[e2];
                            if u2 != v as u32 && comm[u2 as usize] == c {
                                first = false;
                            }
                        }
                        if first {
                            let mut w = f64::new(0.0f32);
                            for e3 in e..end {
                                let u3 = targets[e3];
                                if u3 != v as u32 && comm[u3 as usize] == c {
                                    w += f64::cast_from(weights[e3]);
                                }
                            }
                            let gain = w - scale * k * tot[c as usize];
                            if gain > best_g {
                                best_c = c;
                                best_g = gain;
                                best_w = w;
                            }
                        }
                    }
                }
            }
            if best_c == own {
                out_to[i] = STAY;
            } else {
                out_to[i] = best_c;
            }
            out_wto[i] = best_w;
            out_wfrom[i] = w_from;
        }
        sync_cube(); // the next hub reuses the shared memory
        h += CUBE_COUNT as usize;
    }
}

/// SplitMix64 finaliser (rng::mix); the constants come in as arguments (c0, c1, c2).
#[cube]
fn mix64(z0: u64, c0: u64, c1: u64, c2: u64) -> u64 {
    let mut z = z0 + c0;
    z = (z ^ (z >> 30u64)) * c1;
    z = (z ^ (z >> 27u64)) * c2;
    z ^ (z >> 31u64)
}

/// Is u before v in the seeded order? (rng::order sorts by (key, vertex).)
#[cube]
fn earlier(keys: &[u64], u: usize, v: usize) -> bool {
    let (ku, kv) = (keys[u], keys[v]);
    let mut e = ku < kv;
    if ku == kv {
        e = u < v;
    }
    e
}

/// Colouring, step 1: each vertex's seeded key (rng::key = mix(prefix ^ v), prefix from the
/// host), its number of earlier neighbours, and the first frontier (no earlier neighbours).
#[cube(launch_unchecked)]
fn jp_init(prefix: u64, c0: u64, c1: u64, c2: u64, keys: &mut [u64]) {
    let v = ABSOLUTE_POS;
    if v < keys.len() {
        keys[v] = mix64(prefix ^ u64::cast_from(v), c0, c1, c2);
    }
}

#[cube(launch_unchecked)]
fn jp_count(offsets: &[u32], targets: &[u32], keys: &[u64], waiting: &mut [u32], frontier: &mut [u32], fcount: &[Atomic<u32>]) {
    let v = ABSOLUTE_POS;
    if v < waiting.len() {
        let mut w = 0u32;
        for e in offsets[v] as usize..offsets[v + 1] as usize {
            let u = targets[e] as usize;
            if u != v && earlier(keys, u, v) {
                w += 1;
            }
        }
        waiting[v] = w;
        if w == 0 {
            frontier[fcount[0].fetch_add(1u32) as usize] = v as u32;
        }
    }
}

/// Colouring, one Jones–Plassmann round: each frontier vertex (all its earlier neighbours are
/// coloured) takes the smallest colour none of them has (= the sequential greedy pass), then
/// releases its later neighbours; those with no earlier neighbour left form the next frontier.
#[cube(launch_unchecked)]
fn jp_round(offsets: &[u32], targets: &[u32], keys: &[u64], frontier: &[u32], fcount: &[u32], colour: &mut [u32],
            waiting: &[Atomic<u32>], next: &mut [u32], ncount: &[Atomic<u32>], maxcol: &[Atomic<u32>]) {
    let count = fcount[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        let v = frontier[i] as usize;
        let (start, end) = (offsets[v] as usize, offsets[v + 1] as usize);
        let mut lo = u64::new(0);
        let mut hi = u64::new(0);
        for e in start..end {
            let u = targets[e] as usize;
            if u != v && earlier(keys, u, v) {
                let c = colour[u];
                if c < 64 {
                    lo = lo | (u64::new(1) << u64::cast_from(c));
                } else if c < 128 {
                    hi = hi | (u64::new(1) << u64::cast_from(c - 64));
                }
            }
        }
        let mut col = !u32::new(0);
        for c in 0..128u32 {
            let used = if c < 64 {
                ((lo >> u64::cast_from(c)) & u64::new(1)) == u64::new(1)
            } else {
                ((hi >> u64::cast_from(c - 64)) & u64::new(1)) == u64::new(1)
            };
            if !used && col == !u32::new(0) {
                col = c;
            }
        }
        if col == !u32::new(0) {
            // 128+ colours among earlier neighbours (not seen on level-0 kNN graphs): scan upwards
            let mut c = 128u32;
            let mut found = false;
            while !found {
                let mut used = false;
                for e in start..end {
                    let u = targets[e] as usize;
                    if u != v && earlier(keys, u, v) && colour[u] == c {
                        used = true;
                    }
                }
                if used {
                    c += 1;
                } else {
                    found = true;
                }
            }
            col = c;
        }
        colour[v] = col;
        maxcol[0].fetch_max(col);
        for e in start..end {
            let u = targets[e] as usize;
            if u != v && earlier(keys, v, u) {
                if waiting[u].fetch_sub(1u32) == 1 {
                    next[ncount[0].fetch_add(1u32) as usize] = u as u32;
                }
            }
        }
        i += stride;
    }
}

/// Group sizes, then each vertex into its group's region of the affected list (unordered: all
/// ordering keys on the seeded key).
#[cube(launch_unchecked)]
fn jp_hist(colour: &[u32], counts: &[Atomic<u32>]) {
    let v = ABSOLUTE_POS;
    if v < colour.len() {
        counts[colour[v] as usize].fetch_add(1u32);
    }
}

#[cube(launch_unchecked)]
fn jp_scatter(colour: &[u32], goffs: &[u32], fill: &[Atomic<u32>], alist: &mut [u32]) {
    let v = ABSOLUTE_POS;
    if v < colour.len() {
        let c = colour[v] as usize;
        alist[(goffs[c] + fill[c].fetch_add(1u32)) as usize] = v as u32;
    }
}

/// Start a group: snapshot its affected count (what this group's kernels process) and reset it
/// (later groups' marks append for the next round).
#[cube(launch_unchecked)]
fn begin(acount: &[Atomic<u32>], g: u32, cur: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        cur[0] = acount[g as usize].load();
        acount[g as usize].store(0u32);
    }
}

/// Each proposal's source and target community (a new one is n + v); reset their list heads.
#[cube(launch_unchecked)]
fn resolve(verts: &[u32], cur: &[u32], n: u32, comm: &[u32], out_to: &[u32], res_from: &mut [u32], res_to: &mut [u32],
           heads_to: &[Atomic<u32>], heads_from: &[Atomic<u32>]) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        if out_to[i] != STAY {
        let v = verts[i];
        let from = comm[v as usize];
        let mut to = out_to[i];
        if to == u32::MAX {
            to = n + v;
        }
        res_from[i] = from;
        res_to[i] = to;
        heads_to[to as usize].store(NONE);
        heads_from[from as usize].store(NONE);
        }
        i += stride;
    }
}

/// Link each proposal into its target's and its source's list (unordered; only the set matters).
#[cube(launch_unchecked)]
fn link(cur: &[u32], out_to: &[u32], res_from: &[u32], res_to: &[u32], heads_to: &[Atomic<u32>], heads_from: &[Atomic<u32>],
        next_to: &mut [u32], next_from: &mut [u32], hub_count: &[Atomic<u32>]) {
    if ABSOLUTE_POS == 0 {
        hub_count[0].store(0u32); // decide_hubs of this group has run; ready for the next group
    }
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        if out_to[i] != STAY {
        next_to[i] = heads_to[res_to[i] as usize].exchange(i as u32);
        next_from[i] = heads_from[res_from[i] as usize].exchange(i as u32);
        }
        i += stride;
    }
}

/// Rule C: accept against the frozen fixed-point totals made pessimistic by all earlier
/// proposals (group positions < i) into the target and out of the source. Exactly
/// moving::move_nodes' OrderFree branch.
#[cube(launch_unchecked)]
fn accept(verts: &[u32], cur: &[u32], keys: &[u64], strength: &[f64], kint: &[i64], tot_int: &[i64], size: &[u32], scale: f64,
          out_to: &[u32], out_wto: &[f64], out_wfrom: &[f64], res_from: &[u32], res_to: &[u32],
          heads_to: &[Atomic<u32>], heads_from: &[Atomic<u32>], next_to: &[u32], next_from: &[u32], acc: &mut [u32]) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        acc[i] = 0;
        if out_to[i] != STAY {
            let v = verts[i] as usize;
            // "earlier" = earlier in the seeded order = the group's order (lists are unordered)
            let (from, to) = (res_from[i] as usize, res_to[i] as usize);
            let mut inc = i64::new(0);
            let mut j = heads_to[to].load();
            while j != NONE {
                if earlier(keys, verts[j as usize] as usize, v) {
                    inc += kint[verts[j as usize] as usize];
                }
                j = next_to[j as usize];
            }
            let mut outg = i64::new(0);
            let mut outc = 0u32;
            let mut j = heads_from[from].load();
            while j != NONE {
                if earlier(keys, verts[j as usize] as usize, v) {
                    outg += kint[verts[j as usize] as usize];
                    outc += 1;
                }
                j = next_from[j as usize];
            }
            let k = strength[v];
            let kv = kint[v];
            let fxs = f64::new(16_777_216.0f32); // moving::FX
            let stay = out_wfrom[i] - scale * k * (f64::cast_from(tot_int[from] - kv - outg) / fxs);
            let ok = if out_to[i] == u32::MAX {
                size[to] == 0 && size[from] - outc > 1 && f64::new(0.0f32) > stay
            } else {
                out_wto[i] - scale * k * (f64::cast_from(tot_int[to] + inc) / fxs) > stay
            };
            if ok {
                acc[i] = 1;
            }
        }
        i += stride;
    }
}

/// Apply the accepted moves: integer atomics on totals and sizes, new labels, a move count.
#[cube(launch_unchecked)]
fn apply(verts: &[u32], cur: &[u32], kint: &[i64], acc: &[u32], res_from: &[u32], res_to: &[u32], tot_int: &[Atomic<i64>],
         size: &[Atomic<u32>], comm: &mut [u32], moves: &[Atomic<u32>]) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        if acc[i] == 1 {
        let v = verts[i] as usize;
        let kv = kint[v];
        tot_int[res_from[i] as usize].fetch_sub(kv);
        tot_int[res_to[i] as usize].fetch_add(kv);
        size[res_from[i] as usize].fetch_sub(1u32);
        size[res_to[i] as usize].fetch_add(1u32);
        comm[v] = res_to[i];
        moves[0].fetch_add(1u32);
        }
        i += stride;
    }
}

/// The f64 view of the changed totals (what decide reads): tot = tot_int / FX.
#[cube(launch_unchecked)]
fn refresh(cur: &[u32], acc: &[u32], res_from: &[u32], res_to: &[u32], tot_int: &[i64], tot: &mut [f64]) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        if acc[i] == 1 {
        let fxs = f64::new(16_777_216.0f32);
        let (f, t) = (res_from[i] as usize, res_to[i] as usize);
        tot[f] = f64::cast_from(tot_int[f]) / fxs;
        tot[t] = f64::cast_from(tot_int[t]) / fxs;
        }
        i += stride;
    }
}

/// Flag the neighbours of each moved vertex that are now in another community, and list them in
/// their group's affected list (list order is irrelevant: "earlier" uses group positions).
#[cube(launch_unchecked)]
fn mark(offsets: &[u32], targets: &[u32], comm: &[u32], alist: &mut [u32], goff: u32, cur: &[u32], acc: &[u32],
        affected: &[Atomic<u32>], group_of: &[u32], goffs: &[u32], acount: &[Atomic<u32>]) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        if acc[i] == 1 {
            let v = alist[goff as usize + i] as usize;
            let c = comm[v];
            for e in offsets[v] as usize..offsets[v + 1] as usize {
                let u = targets[e] as usize;
                if comm[u] != c {
                    // first to flag u appends it to its group's list (never this group's: a
                    // colour group is independent), so each vertex is listed at most once
                    if affected[u].exchange(1u32) == 0 {
                        let gu = group_of[u] as usize;
                        let slot = acount[gu].fetch_add(1u32);
                        alist[(goffs[gu] + slot) as usize] = u as u32;
                    }
                }
            }
        }
        i += stride;
    }
}

pub(crate) struct DevGraph {
    off: Handle,
    targets: Handle,
    weights: Handle,
    strength: Handle,
    kint: Handle,
    m: usize,
}

/// A level's graph for the GPU backend: the host CSR (refinement and aggregation run there) and,
/// for a large f32-weighted level 0, its device copy (local moving runs there).
pub(crate) struct GpuGraph<'a, W: Clone> {
    host: Cow<'a, Csr<W>>,
    dev: Option<DevGraph>,
}

pub(crate) struct Gpu {
    client: Client,
}

/// Buffer argument helper (all our kernels take whole buffers).
fn arg(h: &Handle, len: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), len) }
}

impl Gpu {
    pub(crate) fn new() -> Gpu {
        Gpu { client: cubecl::Device::default().client() }
    }

    fn up(&self, b: &[u8]) -> Handle {
        self.client.create_from_slice(b)
    }

    fn grid(len: usize) -> (CubeCount, CubeDim) {
        (CubeCount::Static((len.max(1) as u32).div_ceil(BLOCK), 1, 1), CubeDim::new_1d(BLOCK))
    }

    /// The colouring of moving::colouring(g, seed, step), on the device (Jones–Plassmann over the
    /// seeded keys): group sizes, plus the keys, each vertex's colour (= its group) and the
    /// groups' vertex lists (regions at `goffs`, unordered), all resident.
    fn colour_dev(&self, d: &DevGraph, n: usize, seed: u64, step: u64) -> (Vec<u32>, Vec<u32>, Handle, Handle, Handle) {
        let c = &self.client;
        let prefix = crate::rng::mix(crate::rng::mix(crate::rng::mix(seed) ^ crate::rng::COLOUR_ORDER) ^ step);
        let (h_keys, h_wait, h_colour) = (c.empty(n * 8), c.empty(n * 4), c.empty(n * 4));
        let (mut fr, mut nx) = (c.empty(n * 4), c.empty(n * 4));
        let h_fc = self.up(u32::as_bytes(&[0u32]));
        let h_max = self.up(u32::as_bytes(&[0u32]));
        let (cc, cd) = Self::grid(n);
        let t0 = std::time::Instant::now();
        unsafe {
            jp_init::launch_unchecked(c, cc.clone(), cd.clone(), prefix, 0x9E37_79B9_7F4A_7C15u64, 0xBF58_476D_1CE4_E5B9u64,
                0x94D0_49BB_1331_11EBu64, arg(&h_keys, n));
            jp_count::launch_unchecked(c, cc.clone(), cd.clone(), arg(&d.off, n + 1), arg(&d.targets, d.m), arg(&h_keys, n),
                arg(&h_wait, n), arg(&fr, n), arg(&h_fc, 1));
        }
        // rounds in batches: round r of a batch reads its frontier size from counts[r] and writes
        // the next one to counts[r + 1], so the host reads once per batch; rounds past an empty
        // frontier do nothing
        let mut count = u32::from_bytes(&c.read_one(h_fc.clone()).unwrap())[0];
        let t_setup = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let mut jp_rounds = 0;
        let (rc, rd) = Self::grid(n.min(GRID_CAP));
        while count > 0 {
            let mut init = vec![0u32; JP_BATCH + 1];
            init[0] = count;
            let h_counts = self.up(u32::as_bytes(&init));
            for r in 0..JP_BATCH {
                let at = |h: &Handle, k: usize| arg(&h.clone().offset_start(4 * k as u64), 1);
                unsafe {
                    jp_round::launch_unchecked(c, rc.clone(), rd.clone(), arg(&d.off, n + 1), arg(&d.targets, d.m), arg(&h_keys, n),
                        arg(&fr, n), at(&h_counts, r), arg(&h_colour, n), arg(&h_wait, n), arg(&nx, n), at(&h_counts, r + 1), arg(&h_max, 1));
                }
                std::mem::swap(&mut fr, &mut nx);
            }
            jp_rounds += JP_BATCH;
            count = u32::from_bytes(&c.read_one(h_counts.clone().offset_start(4 * JP_BATCH as u64)).unwrap())[0];
        }
        let t_rounds = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let ng = u32::from_bytes(&c.read_one(h_max).unwrap())[0] as usize + 1;
        let h_counts = self.up(u32::as_bytes(&vec![0u32; ng]));
        unsafe { jp_hist::launch_unchecked(c, cc.clone(), cd.clone(), arg(&h_colour, n), arg(&h_counts, ng)) };
        let counts = u32::from_bytes(&c.read_one(h_counts).unwrap()).to_vec();
        let mut goffs = vec![0u32; ng + 1];
        for g in 0..ng {
            goffs[g + 1] = goffs[g] + counts[g];
        }
        let (h_goffs, h_fill, h_alist) = (self.up(u32::as_bytes(&goffs)), self.up(u32::as_bytes(&vec![0u32; ng])), c.empty(n * 4));
        unsafe { jp_scatter::launch_unchecked(c, cc, cd, arg(&h_colour, n), arg(&h_goffs, ng + 1), arg(&h_fill, ng), arg(&h_alist, n)) };
        if std::env::var_os("DETLEIDEN_TRACE").is_some() {
            let _ = c.read_one(h_fill.clone());
            eprintln!("    jp: setup {t_setup:.4} s (incl. the first touch of the uploaded graph), rounds {t_rounds:.4} s (<= {jp_rounds}), grouping {:.4} s", t0.elapsed().as_secs_f64());
        }
        (counts, goffs, h_keys, h_colour, h_alist)
    }

    /// moving::move_nodes with Apply::OrderFree and tol = 0, entirely on the device.
    fn move_nodes_dev<W: Weight>(&self, g: &Csr<W>, d: &DevGraph, comm: &mut [u32], gamma: f64, seed: u64, step: u64) -> (usize, usize) {
        let n = g.n();
        if n == 0 || g.total <= 0.0 {
            return (0, 0);
        }
        let scale = gamma / g.total;
        let ids = 2 * n;
        let mut tot_int = vec![0i64; ids];
        let mut size = vec![0u32; ids];
        for v in 0..n {
            tot_int[comm[v] as usize] += moving::to_fx(g.strength[v]);
            size[comm[v] as usize] += 1;
        }
        let tot: Vec<f64> = tot_int.iter().map(|&t| moving::fx(t)).collect();
        let c = &self.client;
        let tc = std::time::Instant::now();
        let (lens, goffs, h_keys, h_group_of, h_alist) = self.colour_dev(d, n, seed, step);
        if std::env::var_os("DETLEIDEN_TRACE").is_some() {
            eprintln!("    colouring (GPU) n {n}: {:.4} s, {} colours", tc.elapsed().as_secs_f64(), lens.len());
        }
        let (h_comm, h_tot_int, h_tot, h_size) = (self.up(u32::as_bytes(comm)), self.up(i64::as_bytes(&tot_int)), self.up(f64::as_bytes(&tot)), self.up(u32::as_bytes(&size)));
        let h_aff = self.up(u32::as_bytes(&vec![1u32; n]));
        let (h_heads_to, h_heads_from) = (c.empty(ids * 4), c.empty(ids * 4));
        // affected lists: one region per colour group (initially the whole group, from colour_dev)
        let (h_goffs, h_acount) = (self.up(u32::as_bytes(&goffs)), self.up(u32::as_bytes(&lens)));
        let h_cur = c.empty(4);
        let ng = lens.len();
        let mg = lens.iter().copied().max().unwrap_or(1).max(1) as usize;
        let (o_to, o_wto, o_wfrom) = (c.empty(mg * 4), c.empty(mg * 8), c.empty(mg * 8));
        let (r_from, r_to, n_to, n_from, h_acc) = (c.empty(mg * 4), c.empty(mg * 4), c.empty(mg * 4), c.empty(mg * 4), c.empty(mg * 4));
        let (h_hubs, h_hub_count) = (c.empty(mg * 4), self.up(u32::as_bytes(&[0u32])));
        let zero = [0u32];
        let mut rounds = 0;
        for _ in 0..1000 {
            rounds += 1;
            let h_moves = self.up(u32::as_bytes(&zero));
            for (gi, &glen) in lens.iter().enumerate() {
                let len = glen as usize;
                // grid-stride kernels over the group's affected list: a capped grid
                let (cc, cd) = Self::grid(len.min(GRID_CAP));
                let goff = goffs[gi];
                let verts = h_alist.clone().offset_start(goff as u64 * 4);
                unsafe {
                    begin::launch_unchecked(c, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1), arg(&h_acount, ng), gi as u32, arg(&h_cur, 1));
                    decide::launch_unchecked(c, cc.clone(), cd.clone(), arg(&d.off, n + 1), arg(&d.targets, d.m), arg(&d.weights, d.m),
                        arg(&d.strength, n), arg(&h_comm, n), arg(&h_tot, ids), arg(&h_size, ids), scale, arg(&verts, len), arg(&h_cur, 1),
                        arg(&h_aff, n), arg(&o_to, len), arg(&o_wto, len), arg(&o_wfrom, len), arg(&h_hubs, len), arg(&h_hub_count, 1));
                    decide_hubs::launch_unchecked(c, CubeCount::Static(HUB_CUBES, 1, 1), CubeDim::new_1d(BLOCK), arg(&d.off, n + 1),
                        arg(&d.targets, d.m), arg(&d.weights, d.m), arg(&d.strength, n), arg(&h_comm, n), arg(&h_tot, ids),
                        arg(&h_size, ids), scale, arg(&verts, len), arg(&h_hubs, len), arg(&h_hub_count, 1), arg(&o_to, len),
                        arg(&o_wto, len), arg(&o_wfrom, len));
                    resolve::launch_unchecked(c, cc.clone(), cd.clone(), arg(&verts, len), arg(&h_cur, 1), n as u32, arg(&h_comm, n),
                        arg(&o_to, len), arg(&r_from, len), arg(&r_to, len), arg(&h_heads_to, ids), arg(&h_heads_from, ids));
                    link::launch_unchecked(c, cc.clone(), cd.clone(), arg(&h_cur, 1), arg(&o_to, len), arg(&r_from, len), arg(&r_to, len),
                        arg(&h_heads_to, ids), arg(&h_heads_from, ids), arg(&n_to, len), arg(&n_from, len), arg(&h_hub_count, 1));
                    accept::launch_unchecked(c, cc.clone(), cd.clone(), arg(&verts, len), arg(&h_cur, 1), arg(&h_keys, n), arg(&d.strength, n),
                        arg(&d.kint, n), arg(&h_tot_int, ids), arg(&h_size, ids), scale, arg(&o_to, len), arg(&o_wto, len), arg(&o_wfrom, len),
                        arg(&r_from, len), arg(&r_to, len), arg(&h_heads_to, ids), arg(&h_heads_from, ids), arg(&n_to, len),
                        arg(&n_from, len), arg(&h_acc, len));
                    apply::launch_unchecked(c, cc.clone(), cd.clone(), arg(&verts, len), arg(&h_cur, 1), arg(&d.kint, n), arg(&h_acc, len),
                        arg(&r_from, len), arg(&r_to, len), arg(&h_tot_int, ids), arg(&h_size, ids), arg(&h_comm, n), arg(&h_moves, 1));
                    refresh::launch_unchecked(c, cc.clone(), cd.clone(), arg(&h_cur, 1), arg(&h_acc, len), arg(&r_from, len), arg(&r_to, len),
                        arg(&h_tot_int, ids), arg(&h_tot, ids));
                    mark::launch_unchecked(c, cc, cd, arg(&d.off, n + 1), arg(&d.targets, d.m), arg(&h_comm, n), arg(&h_alist, n), goff,
                        arg(&h_cur, 1), arg(&h_acc, len), arg(&h_aff, n), arg(&h_group_of, n), arg(&h_goffs, ng + 1), arg(&h_acount, ng));
                }
            }
            let moved = u32::from_bytes(&c.read_one(h_moves).unwrap())[0];
            if moved == 0 {
                break;
            }
        }
        comm.copy_from_slice(u32::from_bytes(&c.read_one(h_comm).unwrap()));
        (ng, rounds)
    }
}

impl Backend for Gpu {
    type Graph<'a, W: Weight> = GpuGraph<'a, W>;

    fn upload<'a, W: Weight>(&self, g: &'a Csr<W>) -> Self::Graph<'a, W> {
        let dev = W::as_f32(&g.weights).filter(|_| g.n() >= GPU_MIN_N).map(|w| {
            let off: Vec<u32> = g.offsets.iter().map(|&o| o as u32).collect();
            let kint: Vec<i64> = g.strength.iter().map(|&k| moving::to_fx(k)).collect();
            DevGraph {
                off: self.up(u32::as_bytes(&off)),
                targets: self.up(u32::as_bytes(&g.targets)),
                weights: self.up(f32::as_bytes(w)),
                strength: self.up(f64::as_bytes(&g.strength)),
                kint: self.up(i64::as_bytes(&kint)),
                m: g.targets.len(),
            }
        });
        GpuGraph { host: Cow::Borrowed(g), dev }
    }
    fn n<W: Weight>(&self, g: &Self::Graph<'_, W>) -> usize {
        g.host.n()
    }
    fn edges<W: Weight>(&self, g: &Self::Graph<'_, W>) -> usize {
        g.host.targets.len()
    }
    fn hash<W: Weight>(&self, g: &Self::Graph<'_, W>) -> u64 {
        g.host.hash()
    }
    fn move_nodes<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &mut [u32], gamma: f64, tol: f64, seed: u64, step: u64, apply: Apply) -> (usize, usize) {
        match &g.dev {
            Some(d) if apply == Apply::OrderFree && tol == 0.0 => self.move_nodes_dev(&g.host, d, comm, gamma, seed, step),
            _ => Cpu.move_nodes(&g.host, comm, gamma, tol, seed, step, apply),
        }
    }
    fn refine<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &[u32], gamma: f64, theta: f64, seed: u64, step: u64) -> (Vec<u32>, usize) {
        Cpu.refine(&g.host, comm, gamma, theta, seed, step)
    }
    fn aggregate<W: Weight>(&self, g: &Self::Graph<'_, W>, labels: &[u32], n_labels: usize) -> Self::Graph<'static, f64> {
        GpuGraph { host: Cpu.aggregate(&g.host, labels, n_labels), dev: None }
    }
}

const _: () = assert!(FX == 16_777_216.0, "the kernels hard-code moving::FX");

/// Debug: decide + decide_hubs once for all vertices of `g` under `comm`, against the CPU's
/// best_move. Returns (vertices, hubs, mismatches); prints the first mismatches.
#[doc(hidden)]
pub fn check_decide(g: &Graph, comm: &[u32], gamma: f64) -> (usize, usize, usize) {
    let Repr::F32(c) = &g.0 else { panic!("needs f32-exact weights") };
    let gpu = Gpu::new();
    let cl = &gpu.client;
    let n = c.n();
    let ids = 2 * n;
    let scale = gamma / c.total;
    let mut tot = vec![0.0f64; ids];
    let mut size = vec![0u32; ids];
    for v in 0..n {
        tot[comm[v] as usize] += c.strength[v];
        size[comm[v] as usize] += 1;
    }
    let (mut wt, mut touched) = (vec![0.0f64; ids], Vec::new());
    let cpu: Vec<Option<moving::Move>> = (0..n as u32).map(|v| moving::best_move(c, v, comm, &tot, &size, scale, &mut wt, &mut touched)).collect();
    let off: Vec<u32> = c.offsets.iter().map(|&o| o as u32).collect();
    let verts: Vec<u32> = (0..n as u32).collect();
    let h = |b: &[u8]| gpu.up(b);
    let (h_off, h_t, h_w, h_s) = (h(u32::as_bytes(&off)), h(u32::as_bytes(&c.targets)), h(f32::as_bytes(&c.weights)), h(f64::as_bytes(&c.strength)));
    let (h_c, h_tot, h_size, h_v, h_aff) = (h(u32::as_bytes(comm)), h(f64::as_bytes(&tot)), h(u32::as_bytes(&size)), h(u32::as_bytes(&verts)), h(u32::as_bytes(&vec![1u32; n])));
    let (o_to, o_wto, o_wfrom, h_hubs, h_cnt) = (cl.empty(n * 4), cl.empty(n * 8), cl.empty(n * 8), cl.empty(n * 4), h(u32::as_bytes(&[0u32])));
    let h_cur = h(u32::as_bytes(&[n as u32]));
    let m = c.targets.len();
    let (cc, cd) = Gpu::grid(n);
    unsafe {
        decide::launch_unchecked(cl, cc, cd, arg(&h_off, n + 1), arg(&h_t, m), arg(&h_w, m), arg(&h_s, n), arg(&h_c, n), arg(&h_tot, ids),
            arg(&h_size, ids), scale, arg(&h_v, n), arg(&h_cur, 1), arg(&h_aff, n), arg(&o_to, n), arg(&o_wto, n), arg(&o_wfrom, n), arg(&h_hubs, n), arg(&h_cnt, 1));
        decide_hubs::launch_unchecked(cl, CubeCount::Static(HUB_CUBES, 1, 1), CubeDim::new_1d(BLOCK), arg(&h_off, n + 1), arg(&h_t, m),
            arg(&h_w, m), arg(&h_s, n), arg(&h_c, n), arg(&h_tot, ids), arg(&h_size, ids), scale, arg(&h_v, n), arg(&h_hubs, n), arg(&h_cnt, 1),
            arg(&o_to, n), arg(&o_wto, n), arg(&o_wfrom, n));
    }
    let nh = u32::from_bytes(&cl.read_one(h_cnt).unwrap())[0] as usize;
    let hubs: std::collections::HashSet<u32> = u32::from_bytes(&cl.read_one(h_hubs).unwrap())[..nh].iter().copied().collect();
    let to = u32::from_bytes(&cl.read_one(o_to).unwrap()).to_vec();
    let (wto, wfrom) = (f64::from_bytes(&cl.read_one(o_wto).unwrap()).to_vec(), f64::from_bytes(&cl.read_one(o_wfrom).unwrap()).to_vec());
    let mut mism = 0;
    for v in 0..n {
        let gpu_v = (to[v] != STAY).then_some((to[v], wto[v].to_bits(), wfrom[v].to_bits()));
        let cpu_v = cpu[v].as_ref().map(|m| (m.to, m.w_to.to_bits(), m.w_from.to_bits()));
        if gpu_v != cpu_v {
            if mism < 3 {
                eprintln!("  mismatch v {v} hub {} deg {}: gpu {:?} cpu {:?}", hubs.contains(&(v as u32)), off[v + 1] - off[v],
                          gpu_v.map(|x| (x.0, f64::from_bits(x.1), f64::from_bits(x.2))), cpu_v.map(|x| (x.0, f64::from_bits(x.1), f64::from_bits(x.2))));
            }
            mism += 1;
        }
    }
    (n, nh, mism)
}
