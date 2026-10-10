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
          scale: f64, verts: &[u32], affected: &mut [u32], out_to: &mut [u32], out_wto: &mut [f64], out_wfrom: &mut [f64]) {
    let i = ABSOLUTE_POS;
    if i < verts.len() {
        let v = verts[i] as usize;
        if affected[v] == 0 {
            out_to[i] = STAY; // not affected: skipped, as in moving.rs
        } else {
            affected[v] = 0;
            let mut cs = Array::<u32>::new(MAXC);
            let mut ws = Array::<f64>::new(MAXC);
            let mut nt = 0usize;
            let mut overflow = false;
            let own = comm[v];
            for e in offsets[v] as usize..offsets[v + 1] as usize {
                let u = targets[e];
                if u != v as u32 {
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
                out_to[i] = FALLBACK;
            } else if best_c == own {
                out_to[i] = STAY;
            } else {
                out_to[i] = best_c;
            }
            out_wto[i] = best_w;
            out_wfrom[i] = w_from;
        }
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
    let verts: Vec<u32> = (0..n as u32).collect();
    let h_verts = h(u32::as_bytes(&verts));
    let ones = vec![1u32; n];
    let block = 128u32;
    let launch = || unsafe {
        let h_aff = h(u32::as_bytes(&ones)); // every vertex affected, every launch
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
            BufferArg::from_raw_parts(h_verts.clone(), n),
            BufferArg::from_raw_parts(h_aff.clone(), n),
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

/// dst[buf[idx_off + i]] = buf[val_off + i] for i < len (sections of one packed upload).
#[cube(launch_unchecked)]
fn scatter_u32(buf: &[u32], idx_off: u32, val_off: u32, len: u32, dst: &mut [u32]) {
    let i = ABSOLUTE_POS;
    if i < len as usize {
        dst[buf[idx_off as usize + i] as usize] = buf[val_off as usize + i];
    }
}

/// dst[buf[idx_off + i]] = val[i] for i < len.
#[cube(launch_unchecked)]
fn scatter_f64(buf: &[u32], idx_off: u32, val: &[f64], len: u32, dst: &mut [f64]) {
    let i = ABSOLUTE_POS;
    if i < len as usize {
        dst[buf[idx_off as usize + i] as usize] = val[i];
    }
}

/// Flag the neighbours of each moved vertex that are now in another community. Writes race,
/// but all write 1, so the result doesn't depend on their order (as in moving.rs).
#[cube(launch_unchecked)]
fn mark(offsets: &[u32], targets: &[u32], comm: &[u32], moved: &[u32], len: u32, affected: &mut [u32]) {
    let i = ABSOLUTE_POS;
    if i < len as usize {
        let v = moved[i] as usize;
        let c = comm[v];
        for e in offsets[v] as usize..offsets[v + 1] as usize {
            let u = targets[e] as usize;
            if comm[u] != c {
                affected[u] = 1;
            }
        }
    }
}

use std::borrow::Cow;

use cubecl::client::Client;
use cubecl::server::Handle;

use crate::backend::{Backend, Cpu};
use crate::graph::{Csr, Weight};
use crate::moving::{self, Move, NEW};

/// Below this many vertices local moving stays on the CPU (launch and sync overheads win).
const GPU_MIN_N: usize = 50_000;
const BLOCK: u32 = 128;

pub(crate) struct DevGraph {
    off: Handle,
    targets: Handle,
    weights: Handle,
    strength: Handle,
    n: usize,
    m: usize,
}

/// A level's graph for the GPU backend: the host CSR (refinement and aggregation run there) and,
/// for a large f32-weighted level 0, its device copy (local moving runs there).
pub(crate) struct GpuGraph<'a, W: Clone> {
    host: Cow<'a, Csr<W>>,
    dev: Option<DevGraph>,
}

/// Hybrid backend: level-0 local moving on the GPU, everything else on the CPU. The decide
/// kernel mirrors the CPU exactly and the apply step is the CPU's, so the labels are the CPU's.
pub(crate) struct Gpu {
    client: Client,
}

impl Gpu {
    pub(crate) fn new() -> Gpu {
        Gpu { client: cubecl::Device::default().client() }
    }

    fn up(&self, b: &[u8]) -> Handle {
        self.client.create_from_slice(b)
    }

    fn launch_1d(&self, len: usize) -> (CubeCount, CubeDim) {
        (CubeCount::Static((len as u32).div_ceil(BLOCK), 1, 1), CubeDim::new_1d(BLOCK))
    }

    /// moving::move_nodes with the decide and mark steps on the device.
    fn move_nodes_dev<W: Weight>(&self, g: &Csr<W>, d: &DevGraph, comm: &mut [u32], gamma: f64, tol: f64, seed: u64, step: u64) -> (usize, usize) {
        let n = g.n();
        if n == 0 || g.total <= 0.0 {
            return (0, 0);
        }
        let scale = gamma / g.total;
        let mut tot = vec![0.0f64; n];
        let mut size = vec![0u32; n];
        for v in 0..n {
            tot[comm[v] as usize] += g.strength[v];
            size[comm[v] as usize] += 1;
        }
        let mut empty: Vec<u32> = (0..n as u32).rev().filter(|&c| size[c as usize] == 0).collect();
        let groups = moving::colouring(g, seed, step);

        let c = &self.client;
        let h_comm = self.up(u32::as_bytes(comm));
        let h_tot = self.up(f64::as_bytes(&tot));
        let h_size = self.up(u32::as_bytes(&size));
        let h_aff = self.up(u32::as_bytes(&vec![1u32; n]));
        let h_groups: Vec<Handle> = groups.iter().map(|grp| self.up(u32::as_bytes(grp))).collect();
        let max_g = groups.iter().map(|g| g.len()).max().unwrap_or(0).max(1);
        let (o_to, o_wto, o_wfrom) = (c.empty(max_g * 4), c.empty(max_g * 8), c.empty(max_g * 8));
        let (mut wt, mut touched) = (vec![0.0f64; n], Vec::new());
        let mut seen = vec![false; n]; // communities whose totals changed in this group
        let mut moved: Vec<u32> = Vec::new();
        let mut rounds = 0;

        for _ in 0..1000 {
            rounds += 1;
            let mut round_moves = 0usize;
            let mut round_gain = 0.0f64;
            for (grp, h_grp) in groups.iter().zip(&h_groups) {
                let len = grp.len();
                // decide, on the device, against the frozen state
                let (cc, cd) = self.launch_1d(len);
                unsafe {
                    decide::launch_unchecked(
                        c, cc, cd,
                        BufferArg::from_raw_parts(d.off.clone(), n + 1),
                        BufferArg::from_raw_parts(d.targets.clone(), d.m),
                        BufferArg::from_raw_parts(d.weights.clone(), d.m),
                        BufferArg::from_raw_parts(d.strength.clone(), n),
                        BufferArg::from_raw_parts(h_comm.clone(), n),
                        BufferArg::from_raw_parts(h_tot.clone(), n),
                        BufferArg::from_raw_parts(h_size.clone(), n),
                        scale,
                        BufferArg::from_raw_parts(h_grp.clone(), len),
                        BufferArg::from_raw_parts(h_aff.clone(), n),
                        BufferArg::from_raw_parts(o_to.clone(), len),
                        BufferArg::from_raw_parts(o_wto.clone(), len),
                        BufferArg::from_raw_parts(o_wfrom.clone(), len),
                    )
                };
                // only this group's slice of each output, in one read (one sync)
                let trim = |h: &Handle, bytes: usize| h.clone().offset_end(((max_g - len) * bytes) as u64);
                let out = c.read(vec![trim(&o_to, 4), trim(&o_wto, 8), trim(&o_wfrom, 8)]).unwrap();
                let to = u32::from_bytes(&out[0]);
                if to.iter().all(|&t| t == STAY) {
                    continue;
                }
                let (wto, wfrom) = (f64::from_bytes(&out[1]), f64::from_bytes(&out[2]));
                // fallbacks (too many distinct neighbour communities): the CPU decides, also
                // against the frozen state, before any of this group's moves apply
                let moves: Vec<Move> = (0..len)
                    .filter_map(|i| match to[i] {
                        STAY => None,
                        FALLBACK => moving::best_move(g, grp[i], comm, &tot, &size, scale, &mut wt, &mut touched),
                        t => Some(Move { v: grp[i], to: t, w_to: wto[i], w_from: wfrom[i] }),
                    })
                    .collect();
                // apply, in order, re-checking the gain: exactly moving::move_nodes
                moved.clear();
                let mut changed: Vec<u32> = Vec::new();
                for m in &moves {
                    let (v, k) = (m.v as usize, g.strength[m.v as usize]);
                    let from = comm[v] as usize;
                    let stay = m.w_from - scale * k * (tot[from] - k);
                    let go = if m.to == NEW { 0.0 } else { m.w_to - scale * k * tot[m.to as usize] };
                    if go <= stay || (m.to == NEW && size[from] == 1) {
                        continue;
                    }
                    round_gain += go - stay;
                    let to = if m.to == NEW { empty.pop().expect("an empty community exists") } else { m.to } as usize;
                    tot[from] -= k;
                    size[from] -= 1;
                    if size[from] == 0 {
                        empty.push(from as u32);
                    }
                    tot[to] += k;
                    size[to] += 1;
                    comm[v] = to as u32;
                    moved.push(v as u32);
                    for x in [from, to] {
                        if !seen[x] {
                            seen[x] = true;
                            changed.push(x as u32);
                        }
                    }
                }
                round_moves += moved.len();
                if moved.is_empty() {
                    continue;
                }
                for &x in &changed {
                    seen[x as usize] = false;
                }
                // push the changes to the device in two uploads, then mark there. Packed u32
                // buffer: [moved | their new labels | changed communities | their sizes]
                let (nm, nc) = (moved.len(), changed.len());
                let mut pack: Vec<u32> = Vec::with_capacity(2 * nm + 2 * nc);
                pack.extend_from_slice(&moved);
                pack.extend(moved.iter().map(|&v| comm[v as usize]));
                pack.extend_from_slice(&changed);
                pack.extend(changed.iter().map(|&x| size[x as usize]));
                let ch_tot: Vec<f64> = changed.iter().map(|&x| tot[x as usize]).collect();
                let (h_pack, h_tv) = (self.up(u32::as_bytes(&pack)), self.up(f64::as_bytes(&ch_tot)));
                let (cm, dm) = self.launch_1d(nm);
                let (cx, dx) = self.launch_1d(nc);
                let p = |h: &Handle| unsafe { BufferArg::from_raw_parts(h.clone(), pack.len()) };
                unsafe {
                    scatter_u32::launch_unchecked(c, cm.clone(), dm, p(&h_pack), 0, nm as u32, nm as u32, BufferArg::from_raw_parts(h_comm.clone(), n));
                    scatter_f64::launch_unchecked(c, cx.clone(), dx, p(&h_pack), (2 * nm) as u32, BufferArg::from_raw_parts(h_tv, nc), nc as u32,
                        BufferArg::from_raw_parts(h_tot.clone(), n));
                    scatter_u32::launch_unchecked(c, cx, dx, p(&h_pack), (2 * nm) as u32, (2 * nm + nc) as u32, nc as u32,
                        BufferArg::from_raw_parts(h_size.clone(), n));
                    mark::launch_unchecked(c, cm, dm,
                        BufferArg::from_raw_parts(d.off.clone(), n + 1),
                        BufferArg::from_raw_parts(d.targets.clone(), d.m),
                        BufferArg::from_raw_parts(h_comm.clone(), n),
                        p(&h_pack), nm as u32,
                        BufferArg::from_raw_parts(h_aff.clone(), n));
                }
            }
            if round_moves == 0 || 2.0 * round_gain / g.total <= tol {
                break;
            }
        }
        (groups.len(), rounds)
    }
}

impl Backend for Gpu {
    type Graph<'a, W: Weight> = GpuGraph<'a, W>;

    fn upload<'a, W: Weight>(&self, g: &'a Csr<W>) -> Self::Graph<'a, W> {
        let dev = W::as_f32(&g.weights).filter(|_| g.n() >= GPU_MIN_N).map(|w| {
            let off: Vec<u32> = g.offsets.iter().map(|&o| o as u32).collect();
            DevGraph {
                off: self.up(u32::as_bytes(&off)),
                targets: self.up(u32::as_bytes(&g.targets)),
                weights: self.up(f32::as_bytes(w)),
                strength: self.up(f64::as_bytes(&g.strength)),
                n: g.n(),
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
    fn move_nodes<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &mut [u32], gamma: f64, tol: f64, seed: u64, step: u64) -> (usize, usize) {
        match &g.dev {
            Some(d) => {
                debug_assert_eq!(d.n, g.host.n());
                self.move_nodes_dev(&g.host, d, comm, gamma, tol, seed, step)
            }
            None => Cpu.move_nodes(&g.host, comm, gamma, tol, seed, step),
        }
    }
    fn refine<W: Weight>(&self, g: &Self::Graph<'_, W>, comm: &[u32], gamma: f64, theta: f64, seed: u64, step: u64) -> (Vec<u32>, usize) {
        Cpu.refine(&g.host, comm, gamma, theta, seed, step)
    }
    fn aggregate<W: Weight>(&self, g: &Self::Graph<'_, W>, labels: &[u32], n_labels: usize) -> Self::Graph<'static, f64> {
        GpuGraph { host: Cpu.aggregate(&g.host, labels, n_labels), dev: None }
    }
}
