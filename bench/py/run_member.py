#!/usr/bin/env python3
"""run_member.py --impl {igraph,igraph2,leidenalg,gve} --graph hao2021_100k --name N [--seed 100 --replicates 20 --seed_stride 1 --order input|seed]
Writes an sc-brrr-shaped member to bench/out/baselines/<graph>/<N>/ (N_clusters.tsv, rep<r>/, parameters.json, times.json).

igraph/igraph2/leidenalg: the scanpy env (sc.tl.leiden exactly as sc-brrr's baselines/reference.py calls it;
n_iterations -1 / 2; leidenalg = scanpy 1.12's default flavor and arguments). gve: `cd $PHD/brrr-gve && pixi run python -I`,
brrr-gve's own method.cluster, with GVE built from bench/gve/leiden_timed.cxx (the same wrapper plus a clock);
threads = OMP_NUM_THREADS. times: wall time of the cluster call (incl. igraph's / GVE's graph build).
algo_times (gve only): leidenStaticOmp alone, excluding GVE's DiGraph build; comparable to clust-libreville-rs's
examples/run.rs time_s, which excludes Graph::from_csr.
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
PHD = Path(os.environ["PHD"])  # the directory holding the brrr-gve, sc-brrr and clust-libreville-rs checkouts

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


def gve_timed():
    """bench/gve/leiden_timed.cxx built like brrr-gve's method._gve (same compiler and flags), cached by source hash."""
    import glob
    import hashlib
    import subprocess
    src, inc = BENCH / "gve" / "leiden_timed.cxx", PHD / "brrr-gve" / "gve"
    key = hashlib.sha256(src.read_bytes() + b"".join(f.read_bytes() for f in sorted((inc / "inc").glob("*.hxx")))).hexdigest()[:12]
    lib = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "clust-libreville-rs" / f"gve-timed-{key}.so"
    if not lib.exists():
        cxx = os.environ.get("CXX") or next(iter(glob.glob(f"{sys.prefix}/bin/*-linux-gnu-c++")), "c++")
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.so")
        subprocess.run([cxx, "-std=c++17", "-O3", "-fopenmp", "-shared", "-fPIC", f"-I{inc}", "-o", str(tmp), str(src)], check=True)
        tmp.rename(lib)
    return str(lib)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--impl", required=True, choices=["igraph", "igraph2", "igraph-fix", "leidenalg", "gve"])
    a.add_argument("--graph", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--seed", type=int, default=100)
    a.add_argument("--replicates", type=int, default=20)
    a.add_argument("--seed_stride", type=int, default=1)
    a.add_argument("--order", default="input", choices=["input", "seed"])
    a.add_argument("--gve_lib", help="gve: a scratch build of gve/leiden.cxx to use instead of bench/gve/leiden_timed.cxx (ablations)")
    o = a.parse_args()
    c, ids = load(o.graph)
    algo_seconds = None

    if o.impl == "gve":
        sys.path.insert(0, str(PHD / "brrr-gve"))
        import method  # compiles/loads GVE; never edited here
        import ctypes
        so = ctypes.CDLL(o.gve_lib or gve_timed())
        so.gve_leiden.restype, so.gve_leiden.argtypes = method.GVE.gve_leiden.restype, method.GVE.gve_leiden.argtypes
        method.GVE = so  # method.cluster looks GVE up at call time
        if hasattr(so, "gve_last_algo_seconds"):
            so.gve_last_algo_seconds.restype = ctypes.c_double
            algo_seconds = so.gve_last_algo_seconds
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

    reps, times, iters, algo_times = [], [], [], []
    for r in range(o.replicates):
        t = time.perf_counter()
        reps.append(cluster(o.seed + r * o.seed_stride))
        times.append(time.perf_counter() - t)
        if algo_seconds:
            algo_times.append(algo_seconds())
    d = BENCH / "out" / "baselines" / o.graph / o.name
    write_member(d, o.name, ids, reps, {**vars(o), "resolution": 1.0, "omp_num_threads": os.environ.get("OMP_NUM_THREADS")})
    (d / "times.json").write_text(json.dumps({"times": times, "iterations": iters, **({"algo_times": algo_times} if algo_times else {})}))
    print(d, f"k={len(set(reps[0]))}", f"iterations {iters}" if iters else "", f"time min {min(times):.3f}s median {np.median(times):.3f}s",
          f"algo min {min(algo_times):.3f}s median {np.median(algo_times):.3f}s" if algo_times else "")


if __name__ == "__main__":
    main()
