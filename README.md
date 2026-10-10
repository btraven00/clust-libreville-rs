# detleiden

A parallel, seeded, **deterministic** Leiden for community detection on large sparse graphs, aimed at single-cell kNN graphs (1M+ cells). The same seed gives the same labels at any thread count, on any machine.

Status: planning. Nothing is implemented. See [PLAN.md](PLAN.md); new contributors start at [HANDOFF.md](HANDOFF.md). The work is gated on a Leiden benchmark (phase 0) in sc-brrr / brrr-gve.

Lineage: GVE-Leiden (Sahu, arXiv:2312.13936, MIT) is the performance reference. Its moves are asynchronous, so it is nondeterministic with more than one thread. detleiden uses synchronous, deterministic moves instead.
