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
/// arrays; a vertex with more takes an exact O(degree^2) rescan instead.
const MAXC: usize = 64;
const STAY: u32 = 0xFFFF_FFFD;
const NONE: u32 = 0xFFFF_FFFF;
// moving::NEW = u32::MAX: a move to a new community (id n + v)
/// Below this many vertices local moving stays on the CPU (launch overheads win).
const GPU_MIN_N: usize = 50_000;
const BLOCK: u32 = 128;

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
/// communities rescan in O(degree^2) with the same summation order.
#[cube(launch_unchecked)]
fn decide(offsets: &[u32], targets: &[u32], weights: &[f32], strength: &[f64], comm: &[u32], tot: &[f64], size: &[u32],
          scale: f64, verts: &[u32], affected: &mut [u32], out_to: &mut [u32], out_wto: &mut [f64], out_wfrom: &mut [f64]) {
    let i = ABSOLUTE_POS;
    if i < verts.len() {
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
            let mut w_from = f64::new(0.0f32);
            if overflow {
                for e in start..end {
                    let u = targets[e];
                    if u != v as u32 && comm[u as usize] == own {
                        w_from += f64::cast_from(weights[e]);
                    }
                }
            } else {
                for j in 0..nt {
                    if cs[j] == own {
                        w_from = ws[j];
                    }
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
            if overflow {
                // candidates in order of first occurrence in the row, each summed in row order
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
            } else {
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
}

/// Each proposal's source and target community (a new one is n + v); reset their list heads.
#[cube(launch_unchecked)]
fn resolve(verts: &[u32], len: u32, n: u32, comm: &[u32], out_to: &[u32], res_from: &mut [u32], res_to: &mut [u32],
           heads_to: &[Atomic<u32>], heads_from: &[Atomic<u32>]) {
    let i = ABSOLUTE_POS;
    if i < len as usize && out_to[i] != STAY {
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
}

/// Link each proposal into its target's and its source's list (unordered; only the set matters).
#[cube(launch_unchecked)]
fn link(len: u32, out_to: &[u32], res_from: &[u32], res_to: &[u32], heads_to: &[Atomic<u32>], heads_from: &[Atomic<u32>],
        next_to: &mut [u32], next_from: &mut [u32]) {
    let i = ABSOLUTE_POS;
    if i < len as usize && out_to[i] != STAY {
        next_to[i] = heads_to[res_to[i] as usize].exchange(i as u32);
        next_from[i] = heads_from[res_from[i] as usize].exchange(i as u32);
    }
}

/// Rule C: accept against the frozen fixed-point totals made pessimistic by all earlier
/// proposals (group positions < i) into the target and out of the source. Exactly
/// moving::move_nodes' OrderFree branch.
#[cube(launch_unchecked)]
fn accept(verts: &[u32], len: u32, strength: &[f64], kint: &[i64], tot_int: &[i64], size: &[u32], scale: f64,
          out_to: &[u32], out_wto: &[f64], out_wfrom: &[f64], res_from: &[u32], res_to: &[u32],
          heads_to: &[Atomic<u32>], heads_from: &[Atomic<u32>], next_to: &[u32], next_from: &[u32], acc: &mut [u32]) {
    let i = ABSOLUTE_POS;
    if i < len as usize {
        acc[i] = 0;
        if out_to[i] != STAY {
            let v = verts[i] as usize;
            let (from, to) = (res_from[i] as usize, res_to[i] as usize);
            let mut inc = i64::new(0);
            let mut j = heads_to[to].load();
            while j != NONE {
                if (j as usize) < i {
                    inc += kint[verts[j as usize] as usize];
                }
                j = next_to[j as usize];
            }
            let mut outg = i64::new(0);
            let mut outc = 0u32;
            let mut j = heads_from[from].load();
            while j != NONE {
                if (j as usize) < i {
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
    }
}

/// Apply the accepted moves: integer atomics on totals and sizes, new labels, a move count.
#[cube(launch_unchecked)]
fn apply(verts: &[u32], len: u32, kint: &[i64], acc: &[u32], res_from: &[u32], res_to: &[u32], tot_int: &[Atomic<i64>],
         size: &[Atomic<u32>], comm: &mut [u32], moves: &[Atomic<u32>]) {
    let i = ABSOLUTE_POS;
    if i < len as usize && acc[i] == 1 {
        let v = verts[i] as usize;
        let kv = kint[v];
        tot_int[res_from[i] as usize].fetch_sub(kv);
        tot_int[res_to[i] as usize].fetch_add(kv);
        size[res_from[i] as usize].fetch_sub(1u32);
        size[res_to[i] as usize].fetch_add(1u32);
        comm[v] = res_to[i];
        moves[0].fetch_add(1u32);
    }
}

/// The f64 view of the changed totals (what decide reads): tot = tot_int / FX.
#[cube(launch_unchecked)]
fn refresh(len: u32, acc: &[u32], res_from: &[u32], res_to: &[u32], tot_int: &[i64], tot: &mut [f64]) {
    let i = ABSOLUTE_POS;
    if i < len as usize && acc[i] == 1 {
        let fxs = f64::new(16_777_216.0f32);
        let (f, t) = (res_from[i] as usize, res_to[i] as usize);
        tot[f] = f64::cast_from(tot_int[f]) / fxs;
        tot[t] = f64::cast_from(tot_int[t]) / fxs;
    }
}

/// Flag the neighbours of each moved vertex that are now in another community (all writes are
/// 1, so their order doesn't matter).
#[cube(launch_unchecked)]
fn mark(offsets: &[u32], targets: &[u32], comm: &[u32], verts: &[u32], len: u32, acc: &[u32], affected: &mut [u32]) {
    let i = ABSOLUTE_POS;
    if i < len as usize && acc[i] == 1 {
        let v = verts[i] as usize;
        let c = comm[v];
        for e in offsets[v] as usize..offsets[v + 1] as usize {
            let u = targets[e] as usize;
            if comm[u] != c {
                affected[u] = 1;
            }
        }
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
        let groups = moving::colouring(g, seed, step);
        let c = &self.client;
        let (h_comm, h_tot_int, h_tot, h_size) = (self.up(u32::as_bytes(comm)), self.up(i64::as_bytes(&tot_int)), self.up(f64::as_bytes(&tot)), self.up(u32::as_bytes(&size)));
        let h_aff = self.up(u32::as_bytes(&vec![1u32; n]));
        let (h_heads_to, h_heads_from) = (c.empty(ids * 4), c.empty(ids * 4));
        let h_groups: Vec<Handle> = groups.iter().map(|grp| self.up(u32::as_bytes(grp))).collect();
        let mg = groups.iter().map(|g| g.len()).max().unwrap_or(1).max(1);
        let (o_to, o_wto, o_wfrom) = (c.empty(mg * 4), c.empty(mg * 8), c.empty(mg * 8));
        let (r_from, r_to, n_to, n_from, h_acc) = (c.empty(mg * 4), c.empty(mg * 4), c.empty(mg * 4), c.empty(mg * 4), c.empty(mg * 4));
        let zero = [0u32];
        let mut rounds = 0;
        for _ in 0..1000 {
            rounds += 1;
            let h_moves = self.up(u32::as_bytes(&zero));
            for (grp, h_grp) in groups.iter().zip(&h_groups) {
                let len = grp.len();
                let (cc, cd) = Self::grid(len);
                let l = len as u32;
                unsafe {
                    decide::launch_unchecked(c, cc.clone(), cd.clone(), arg(&d.off, n + 1), arg(&d.targets, d.m), arg(&d.weights, d.m),
                        arg(&d.strength, n), arg(&h_comm, n), arg(&h_tot, ids), arg(&h_size, ids), scale, arg(h_grp, len), arg(&h_aff, n),
                        arg(&o_to, len), arg(&o_wto, len), arg(&o_wfrom, len));
                    resolve::launch_unchecked(c, cc.clone(), cd.clone(), arg(h_grp, len), l, n as u32, arg(&h_comm, n), arg(&o_to, len),
                        arg(&r_from, len), arg(&r_to, len), arg(&h_heads_to, ids), arg(&h_heads_from, ids));
                    link::launch_unchecked(c, cc.clone(), cd.clone(), l, arg(&o_to, len), arg(&r_from, len), arg(&r_to, len),
                        arg(&h_heads_to, ids), arg(&h_heads_from, ids), arg(&n_to, len), arg(&n_from, len));
                    accept::launch_unchecked(c, cc.clone(), cd.clone(), arg(h_grp, len), l, arg(&d.strength, n), arg(&d.kint, n),
                        arg(&h_tot_int, ids), arg(&h_size, ids), scale, arg(&o_to, len), arg(&o_wto, len), arg(&o_wfrom, len),
                        arg(&r_from, len), arg(&r_to, len), arg(&h_heads_to, ids), arg(&h_heads_from, ids), arg(&n_to, len),
                        arg(&n_from, len), arg(&h_acc, len));
                    apply::launch_unchecked(c, cc.clone(), cd.clone(), arg(h_grp, len), l, arg(&d.kint, n), arg(&h_acc, len),
                        arg(&r_from, len), arg(&r_to, len), arg(&h_tot_int, ids), arg(&h_size, ids), arg(&h_comm, n), arg(&h_moves, 1));
                    refresh::launch_unchecked(c, cc.clone(), cd.clone(), l, arg(&h_acc, len), arg(&r_from, len), arg(&r_to, len),
                        arg(&h_tot_int, ids), arg(&h_tot, ids));
                    mark::launch_unchecked(c, cc, cd, arg(&d.off, n + 1), arg(&d.targets, d.m), arg(&h_comm, n), arg(h_grp, len), l,
                        arg(&h_acc, len), arg(&h_aff, n));
                }
            }
            let moved = u32::from_bytes(&c.read_one(h_moves).unwrap())[0];
            if moved == 0 {
                break;
            }
        }
        comm.copy_from_slice(u32::from_bytes(&c.read_one(h_comm).unwrap()));
        (groups.len(), rounds)
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
