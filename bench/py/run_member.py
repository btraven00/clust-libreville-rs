#!/usr/bin/env python3
"""run_member.py --impl {igraph,igraph2,leidenalg,gve} --graph hao2021_100k --name N [--seed 100 --replicates 20 --seed_stride 1 --order input|seed]
Writes an sc-brrr-shaped member to bench/out/baselines/<graph>/<N>/ (N_clusters.tsv, rep<r>/, parameters.json, times.json).

igraph/igraph2/leidenalg: the scanpy env (sc.tl.leiden exactly as sc-brrr's baselines/reference.py calls it;
n_iterations -1 / 2; leidenalg = scanpy 1.12's default flavor and arguments). gve: `cd $PHD/brrr-gve && pixi run python -I`,
brrr-gve's own method.cluster; threads = OMP_NUM_THREADS. Times are the wall time of the cluster call alone (incl. igraph's graph build).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

BENCH = Path(__file__).resolve().parents[1]


def load(graph):
    z = np.load(BENCH / "graphs" / f"{graph}.npz")
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])), z["cell_ids"]


def write_tsv(path, ids, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("cell_id\tcluster\n")
        f.writelines(f"{i}\t{l}\n" for i, l in zip(ids, labels))


def write_member(d, name, ids, reps, params):
    """reps: list of label arrays, rep r -> d/rep<r>/; the declared output is a copy of rep0 (sc-brrr layout)."""
    for r, lab in enumerate(reps):
        write_tsv(d / f"rep{r}" / f"{name}_clusters.tsv", ids, lab)
    write_tsv(d / f"{name}_clusters.tsv", ids, reps[0])
    (d / "parameters.json").write_text(json.dumps(params))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--impl", required=True, choices=["igraph", "igraph2", "igraph-fix", "leidenalg", "gve"])
    a.add_argument("--graph", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--seed", type=int, default=100)
    a.add_argument("--replicates", type=int, default=20)
    a.add_argument("--seed_stride", type=int, default=1)
    a.add_argument("--order", default="input", choices=["input", "seed"])
    a.add_argument("--gve_lib", help="gve: a scratch build of gve/leiden.cxx to use instead of brrr-gve's (ablations)")
    o = a.parse_args()
    c, ids = load(o.graph)

    if o.impl == "gve":
        sys.path.insert(0, str(Path.home() / "phd" / "brrr-gve"))
        import method  # compiles/loads GVE; never edited here
        if o.gve_lib:
            import ctypes
            so = ctypes.CDLL(o.gve_lib)
            so.gve_leiden.restype, so.gve_leiden.argtypes = method.GVE.gve_leiden.restype, method.GVE.gve_leiden.argtypes
            method.GVE = so  # method.cluster looks GVE up at call time
        c32 = sp.csr_matrix((c.data.astype(np.float32), c.indices, c.indptr), shape=c.shape)  # lossless: the source is f32

        def cluster(seed):
            return method.cluster(SimpleNamespace(connectivities=c32), SimpleNamespace(options={"leiden_order": o.order}, seed=seed, resolution=1.0))
    elif o.impl == "igraph-fix":
        # python-igraph 0.11.9's n_iterations=-1 stops after ONE iteration (leiden-bench.md); this is the fixed point it should reach:
        # one igraph Leiden iteration at a time from the previous membership, until the partition no longer changes.
        import random
        import igraph as ig
        from scanpy._utils import get_igraph_from_adjacency
        sys.path.insert(0, str(Path(__file__).parent))  # -I drops the script dir
        from bench_score import canon_hash
        g = get_igraph_from_adjacency(c, directed=False)  # the graph sc.tl.leiden builds

        def cluster(seed):
            random.seed(seed); ig.set_random_number_generator(random)
            m, h = None, None
            for it in range(1, 1001):
                m = g.community_leiden(objective_function="modularity", weights="weight", resolution=1.0, n_iterations=1, initial_membership=m).membership
                if h == (h := canon_hash(np.array(m))):
                    break
            iters.append(it)
            return m
    else:
        import anndata as ad
        import scanpy as sc

        def cluster(seed):
            x = ad.AnnData(shape=(c.shape[0], 0))
            x.obsp["connectivities"] = c
            x.uns["neighbors"] = {"connectivities_key": "connectivities"}
            if o.impl == "leidenalg":
                sc.tl.leiden(x, resolution=1.0, random_state=seed, flavor="leidenalg")
            else:
                sc.tl.leiden(x, resolution=1.0, random_state=seed, flavor="igraph", directed=False,
                             n_iterations=-1 if o.impl == "igraph" else 2)
            return x.obs["leiden"].tolist()

    reps, times, iters = [], [], []
    for r in range(o.replicates):
        t = time.perf_counter()
        reps.append(cluster(o.seed + r * o.seed_stride))
        times.append(time.perf_counter() - t)
    d = BENCH / "out" / "baselines" / o.graph / o.name
    write_member(d, o.name, ids, reps, {**vars(o), "resolution": 1.0, "omp_num_threads": os.environ.get("OMP_NUM_THREADS")})
    (d / "times.json").write_text(json.dumps({"times": times, "iterations": iters}))
    print(d, f"k={len(set(reps[0]))}", f"iterations {iters}" if iters else "", f"time min {min(times):.3f}s median {np.median(times):.3f}s")


if __name__ == "__main__":
    main()
