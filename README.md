# clust-libreville-rs

A parallel, seeded, **deterministic** Leiden for community detection on large sparse graphs, aimed at single-cell kNN graphs (1M+ cells). The same seed gives the same labels at any thread count. The CPU code avoids everything that varies between machines (platform `libm`, fused multiply-add, scheduling-dependent sums), and gives the same labels on x86_64 Linux and Apple M1; GPU results are repeatable on one device but may differ from the CPU's.

**Status: under revision.** Implemented (CPU; experimental GPU backends via cubecl: CUDA, Metal; no speedup over the CPU yet), with bindings for R (`r/`, package `clusterLibreville`) and Python (`python/`, `pip install sc-cluster-libreville`, import `cluster_libreville`). How it works and why, and how it is tested (CPU, GPU, Metal over ssh): [docs/design.md](docs/design.md).

Lineage: GVE-Leiden (Sahu, [arXiv:2312.13936](https://arxiv.org/abs/2312.13936), MIT) is the performance reference. Its moves are asynchronous, so it is nondeterministic with more than one thread. This crate uses synchronous, deterministic moves instead.

**The name.** The great circle from Leiden through Louvain crosses the Equator about 50 km from Libreville, the "free town". If you prefer more determinism, run it through Louvain-la-Neuve (UCLouvain, where the method was born): it hits São Tomé within a few km, just off the Equator, but Libreville just sounds better.
