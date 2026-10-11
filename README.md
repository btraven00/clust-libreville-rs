# clust-libreville-rs

A parallel, seeded, **deterministic** Leiden for community detection on large sparse graphs, aimed at single-cell kNN graphs (1M+ cells). The same seed gives the same labels at any thread count, on any machine.

Status: implemented (CPU; GPU backends via cubecl: CUDA, Metal in testing), with bindings for R (`r/`, package `clusterLibreville`) and Python (`python/`, `pip install cluster-libreville`). See [PLAN.md](PLAN.md); new contributors start at [HANDOFF.md](HANDOFF.md). A code walkthrough of GVE-Leiden and leiden-rs is in `$PHD/worklog/detleiden/code-walkthrough.md`. The work is gated on a Leiden benchmark (phase 0) in sc-brrr / brrr-gve.

Lineage: GVE-Leiden (Sahu, arXiv:2312.13936, MIT) is the performance reference. Its moves are asynchronous, so it is nondeterministic with more than one thread. This crate uses synchronous, deterministic moves instead.

**The name.** The great circle from Leiden through Louvain crosses the Equator about 50 km from Libreville, "free town". If you prefer more determinism, run it through Louvain-la-Neuve (UCLouvain, where the method was born): it hits São Tomé within a few km, just off the Equator.
