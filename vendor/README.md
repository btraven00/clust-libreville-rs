# vendor/

`cubecl-llvm/`: cubecl-llvm 0.11.0 from crates.io (MIT OR Apache-2.0, licences included), the code
generator of cubecl's CUDA backend, with FMA contraction switched off so that f64 kernels round
exactly as the CPU does. A fused multiply-add rounds `a * b + c` once where the CPU rounds twice, so a
near-tie between two gains could be decided differently on the device and break CUDA = CPU
(`Apply::OrderFree`) bit for bit. Before the change, the PTX of the decide, accept and refinement
kernels had `fma.rn.f64`. Two changes, both marked `clust-libreville-rs`:

- `src/shared/to_llvm/math.rs`, `fma_contraction`: no `contract` flag on float add, sub and mul
  (upstream sets it on GPU targets), so LLVM does not fuse;
- `src/shared/base.rs`: `InstCombinePass` is not run (it fuses a product and the add or subtract that
  consumes it into an explicit `llvm.fma`).

cubecl has no switch for either. Used through `[patch.crates-io]` in the root `Cargo.toml`, which
applies to this repository's own builds only. Drop it once cubecl can switch FMA off.
