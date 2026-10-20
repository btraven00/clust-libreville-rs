# clust-libreville-rs

A parallel, seeded, **deterministic** Leiden for community detection on large sparse graphs, aimed at single-cell kNN graphs (1M+ cells).

Today you pick one: reproducible but serial (igraph, leidenalg), or parallel but with different labels on every run (GVE-Leiden, networkit, cuGraph). This gives you both: **reproducible like igraph, within 1.5× of GVE-Leiden's time** (same stopping rule, 1 and 8 threads: on par up to 100k cells, 1.25–1.45× at 1M).

The same seed gives the same labels at any thread count. The CPU code that decides labels avoids everything that varies between machines (platform `libm`, fused multiply-add, scheduling-dependent sums), and gives the same labels on x86_64 Linux and Apple M1; GPU results are repeatable on one device but may differ from the CPU's.

**Status: pre-release.** Implemented (CPU; experimental GPU backends via cubecl: CUDA, Metal; no speedup over the CPU yet), with bindings for R (`r/`, package `clusterLibreville`) and Python (`python/`, `pip install sc-cluster-libreville`, import `cluster_libreville`). Details and testing (CPU, GPU): [docs/design.md](docs/design.md).

Lineage: GVE-Leiden (Sahu, [arXiv:2312.13936](https://arxiv.org/abs/2312.13936), MIT) is the performance reference. Its moves are asynchronous, so it is nondeterministic with more than one thread. This crate uses synchronous, deterministic moves instead.

**The name.** The great circle from Leiden through Louvain crosses the Equator about 50 km from Libreville, the "free town". If you prefer more determinism, run it through Louvain-la-Neuve (UCLouvain, where the method was born): it hits São Tomé within a few km, just off the Equator, but Libreville just sounds better.
