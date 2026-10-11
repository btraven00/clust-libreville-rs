# clust-libreville-rs

A parallel, seeded, **deterministic** Leiden for community detection on large sparse graphs, aimed at single-cell kNN graphs (1M+ cells). The same seed gives the same labels at any thread count, on any machine.

Status: implemented (CPU; GPU backends via cubecl: CUDA, Metal in testing), with R (`r/`) and Python (`python/`) bindings, still published as `detleiden`. See [PLAN.md](PLAN.md); new contributors start at [HANDOFF.md](HANDOFF.md). A code walkthrough of GVE-Leiden and leiden-rs is in `$PHD/worklog/detleiden/code-walkthrough.md`. The work is gated on a Leiden benchmark (phase 0) in sc-brrr / brrr-gve.

Lineage: GVE-Leiden (Sahu, arXiv:2312.13936, MIT) is the performance reference. Its moves are asynchronous, so it is nondeterministic with more than one thread. This crate uses synchronous, deterministic moves instead.

**The name.** The great circle through Leiden and Louvain (Leuven), the two cities that name the
algorithms, crosses the Equator at 9.0°E and passes 52 km from Libreville (0.42°N, 9.47°E):
follow the Leiden–Louvain line south and, within a radius of r ≈ 52 km, you arrive in Libreville.
(Through Louvain-la-Neuve instead, the miss is 309 km.)
