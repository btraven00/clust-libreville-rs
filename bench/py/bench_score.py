#!/usr/bin/env python3
"""bench_score.py <graph> <dir>... [--same_seed] [--reference DIR] [--out DIR]   (e.g. graph = hao2021_100k; any python with numpy+scipy, -I)
Each <dir> is an sc-brrr-shaped member (<name>_clusters.tsv, rep<r>/) or a directory of .npy label files (one per replicate, graph
row order, sorted by name), which is first converted in place into that layout (--same_seed: the .npy are same-seed reruns).
Scores with $PHD/sc-brrr/metrics/ari.py (ari_ref vs the sc-brrr reference, 20 igraph seeds 100-119 at n_iterations=-1 = ONE iteration in python-igraph 0.11.9 (leiden-bench.md); cluster_count_error, rerun_ari_*) and adds
the shared float64 modularity (γ=1), cluster counts and distinct partitions. Writes <out>/bench_metrics.jsonl and prints a table.
`bench_score.py --selftest` checks modularity against igraph on hao2021_10k.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BENCH = Path(__file__).resolve().parents[1]
ARI_PY = Path.home() / "phd/sc-brrr/metrics/ari.py"
_s = importlib.util.spec_from_file_location("sc_brrr_ari", ARI_PY)
ari = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ari)  # reuse coded()/ari()/replicates(), never copied


def load(graph):
    z = np.load(BENCH / "graphs" / f"{graph}.npz")
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])), z["cell_ids"]


def modularity(c, lab):
    """Newman modularity, γ = 1, float64, fixed summation order. c: symmetric CSR (both directions stored), lab: int codes."""
    rows = np.repeat(np.arange(c.shape[0]), np.diff(c.indptr))
    two_m = c.data.sum()
    inside = c.data[lab[rows] == lab[c.indices]].sum()
    tot = np.bincount(lab, weights=np.asarray(c.sum(axis=1)).ravel())
    return float(inside / two_m - (tot @ tot) / two_m**2)


def canon_hash(lab):
    """Hash of the partition, independent of label names (relabelled by first occurrence)."""
    _, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    return hashlib.sha1(rank[inv].tobytes()).hexdigest()[:12]


def default_reference(graph):
    size = graph.split("_")[-1]
    hits = sorted(glob.glob(str(Path.home() / f"phd/sc-brrr/runs/*-20261010T1[89]*-{size}-*/out/data/hao2021_{size}/*/pipeline/reference/*/reference_clusters.tsv")))
    return Path(hits[0]).parent if hits and graph.startswith("hao2021_") else None


def npy_to_member(d, ids, same_seed):
    """A directory of .npy labels -> the sc-brrr member layout in the same directory; returns the declared clusters_tsv."""
    name = d.name
    files = sorted(d.glob("*.npy"), key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)])
    for r, f in enumerate([files[0]] + files):
        lab = np.load(f)
        assert lab.shape == (len(ids),), f"{f}: {lab.shape} labels for {len(ids)} cells"
        out = d / (f"rep{r - 1}" if r else ".") / f"{name}_clusters.tsv"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as h:
            h.write("cell_id\tcluster\n")
            h.writelines(f"{i}\t{l}\n" for i, l in zip(ids, lab))
    (d / "parameters.json").write_text(json.dumps({"seed_stride": 0 if same_seed else 1, "source": [f.name for f in files]}))
    return d / f"{name}_clusters.tsv"


def member_tsv(d, ids, same_seed):
    if any(d.glob("*.npy")):
        return npy_to_member(d, ids, same_seed)  # always reconverted: the .npy are the source of truth
    return next(d.glob("*_clusters.tsv"))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("graph")
    a.add_argument("dirs", nargs="+", type=Path)
    a.add_argument("--same_seed", action="store_true")
    a.add_argument("--reference", type=Path)
    a.add_argument("--out", type=Path)
    o = a.parse_args()
    c, ids = load(o.graph)
    pos = {x: i for i, x in enumerate(ids)}
    ref = o.reference or default_reference(o.graph)
    out = o.out or BENCH / "out" / "baselines" / o.graph / "_score"
    out.mkdir(parents=True, exist_ok=True)

    tsvs = {d.resolve(): member_tsv(d.resolve(), ids, o.same_seed) for d in o.dirs}
    if ref:
        ref = ref.resolve()
        tsvs.setdefault(ref, next(ref.glob("*_clusters.tsv")))
    members = [{"dir": str(d), "module": "reference" if d == ref else d.name} for d in tsvs]
    (out / "lineage.json").write_text(json.dumps({"members": members}))
    cmd = [sys.executable, "-I", str(ARI_PY), "--output_dir", str(out), "--name", "bench", "--clusters_tsv", *map(str, tsvs.values())]
    subprocess.run(cmd + (["--reference", str(tsvs[ref])] if ref else []), check=True)

    recs = [json.loads(l) for l in open(out / "bench_metrics.jsonl")]
    with open(out / "bench_metrics.jsonl", "a") as f:
        for d, tsv in tsvs.items():
            reps = [(r, ari.coded(p, pos)) for r, p in ari.replicates(tsv) if r is not None] or [(None, ari.coded(tsv, pos))]
            for r, lab in reps:
                rec = {"metric": "modularity", "value": modularity(c, lab), "subject": str(d), "rep": r, "attrs": {"n_clusters": int(lab.max() + 1)}}
                f.write(json.dumps(rec) + "\n"); recs.append(rec)
            rec = {"metric": "distinct_partitions", "value": len({canon_hash(l) for _, l in reps}), "subject": str(d), "rep": None, "attrs": {"n_reps": len(reps)}}
            f.write(json.dumps(rec) + "\n"); recs.append(rec)

    def get(d, m):
        return [x for x in recs if x["metric"] == m and x["subject"] == str(d)]
    floor = next((x["value"] for x in get(ref, "rerun_ari_seeds")), None) if ref else None
    print(f"{o.graph}: reference {ref}\nfloor (reference rerun_ari_seeds) = {floor}")
    print(f"{'member':28} {'reps':>4} {'k med':>6} {'Q mean':>8} {'Q min':>8} {'Q max':>8} {'ari_ref':>7} {'rerun':>14} {'min':>6} {'distinct':>8} {'nu':>6}")
    for d in tsvs:
        q = get(d, "modularity"); k = [x["attrs"]["n_clusters"] for x in q]; qv = [x["value"] for x in q]
        ar = [x["value"] for x in get(d, "ari_ref")]
        rr = get(d, "rerun_ari_same_seed") + get(d, "rerun_ari_seeds")
        rv = rr[0] if rr else None
        nu = (1 - rv["value"]) / (1 - floor) if rv and floor and rv["metric"].endswith("same_seed") else None
        print(f"{("reference (sc-brrr)" if d == ref else d.name):28} {len(q):4d} {np.median(k):6.0f} {np.mean(qv):8.5f} {min(qv):8.5f} {max(qv):8.5f} "
              f"{(np.mean(ar) if ar else float('nan')):7.4f} {(rv['metric'][10:] + ' ' + format(rv['value'], '.4f')) if rv else '-':>14} "
              f"{(rv['attrs']['min'] if rv else float('nan')):6.4f} {get(d, 'distinct_partitions')[0]['value']:8d} {(nu if nu is not None else float('nan')):6.3f}")


def selftest():
    import igraph as ig
    c, ids = load("hao2021_10k")
    lab = ari.coded(default_reference("hao2021_10k") / "rep0" / "reference_clusters.tsv", {x: i for i, x in enumerate(ids)})
    u = sp.triu(c, 1).tocoo()
    g = ig.Graph(n=c.shape[0], edges=np.c_[u.row, u.col].tolist(), edge_attrs={"w": u.data.tolist()})
    q_ig, q = g.modularity(lab.tolist(), weights="w", resolution=1.0, directed=False), modularity(c, lab)
    assert abs(q - q_ig) < 1e-12, (q, q_ig)
    assert canon_hash(np.array([2, 2, 0, 1])) == canon_hash(np.array([5, 5, 7, 3])) != canon_hash(np.array([0, 0, 0, 1]))
    print(f"selftest ok: modularity {q:.12f}, igraph {q_ig:.12f}, |diff| {abs(q - q_ig):.1e}")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--selftest"] else main()
