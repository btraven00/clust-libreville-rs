# clust-libreville-rs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23516379.svg)](https://doi.org/10.5281/zenodo.23516379)

A parallel, seeded, **deterministic** Leiden for community detection on large
sparse graphs, aimed at single-cell kNN graphs (1M+ cells).

## Yes, but why

Usually you have to pick one: reproducible and serial (igraph, leidenalg), or
parallel but getting different labels on every run (GVE-Leiden, networkit,
cuGraph).

This package gives you the best of both worlds: **reproducible like igraph,
within 1.5× of GVE-Leiden's time**.[^timing]

[^timing]: Same stopping rule as GVE-Leiden (`--tol 1e-2 --theta 0`), 1 and 8
    threads, medians of 5 alternating runs (`bench/time_fair.sh`): on par up to
    100k cells, 1.25–1.45× at 1M.

## How

The same seed gives the same labels at any thread count. On a CPU, the trick is that
the code for choosing labels avoids everything that varies between machines and gives
the same labels on x86_64 Linux and Apple M1.

GPU results are also repeatable on one device but may differ from the CPU's
(and the performance gain is not great, so little more than an exercise for now).

## Should I use this?

This is still a pre-release, so who knows.

There are bindings for R (package `clusterLibreville`) 
and Python (`python/`, `pip install sc-cluster-libreville`, import
`cluster_libreville`). Details and testing (CPU, GPU): [docs/design.md](docs/design.md).

## Lineage

GVE-Leiden (Sahu, [arXiv:2312.13936](https://arxiv.org/abs/2312.13936), MIT) is
the performance reference. That code uses asynchronous moves, so it is
nondeterministic with more than one thread. This crate uses
synchronous, deterministic moves instead.

## The name

The great circle from Leiden through Louvain crosses the Equator about 50 km
from Libreville, the "free town". If you prefer more determinism, run it
through Louvain-la-Neuve (UCLouvain, where the method was born): it hits São
Tomé within a few km, just off the Equator, but Libreville just sounds better.
