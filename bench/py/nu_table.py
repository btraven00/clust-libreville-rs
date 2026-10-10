#!/usr/bin/env python3
"""nu_table.py <graph>...: GVE same-seed spread vs three seed-to-seed floors, from bench/out/baselines/<graph>/_score/bench_metrics.jsonl
(run bench_score.py first). ν = (1 − same-seed ARI)/(1 − floor). Also: ARI of each t2/t8 replicate vs the (deterministic) t1 labels.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_score import ari, load  # sc-brrr's ari.py helpers

B = Path(__file__).resolve().parents[1] / "out" / "baselines"
for g in sys.argv[1:]:
    recs = [json.loads(l) for l in open(B / g / "_score" / "bench_metrics.jsonl")]
    val = lambda name, m: next(x["value"] for x in recs if x["metric"] == m and x["subject"].endswith("/" + name) or
                               (x["metric"] == m and name == "reference" and "pipeline/reference" in x["subject"]))
    floors = {}
    for k, name in [("sc-brrr ref (igraph 1 iter)", "reference"), ("leidenalg converged", "leidenalg-seeds"),
                    ("igraph fixed point", "igraph-fix-seeds"), ("GVE seeds, 1 thread", "gve-seed-t1-seeds")]:
        try:
            floors[k] = val(name, "rerun_ari_seeds")
        except StopIteration:
            pass  # not run on this graph
    _, ids = load(g)
    pos = {x: i for i, x in enumerate(ids)}
    print(f"## {g}\nfloors: " + ", ".join(f"{k} {v:.4f}" for k, v in floors.items()))
    print("| member | n | same-seed ARI mean (min) | distinct | " + " | ".join(f"ν vs {k}" for k in floors) + " | ARI vs t1 mean (min) |")
    for order in ("input", "seed"):
        t1 = ari.coded(B / g / f"gve-{order}-t1-same" / "rep0" / f"gve-{order}-t1-same_clusters.tsv", pos)
        for t in (1, 2, 8):
            name = f"gve-{order}-t{t}-same"
            r = next(x for x in recs if x["metric"] == "rerun_ari_same_seed" and x["subject"].endswith("/" + name))
            dist = val(name, "distinct_partitions")
            reps = [p for _, p in ari.replicates(B / g / name / f"{name}_clusters.tsv") if _ is not None]
            vs = [ari.ari(ari.coded(p, pos), t1) for p in reps]
            nus = " | ".join(f"{(1 - r['value']) / (1 - f):.2f}" for f in floors.values())
            print(f"| {name} | {len(reps)} | {r['value']:.4f} ({r['attrs']['min']:.4f}) | {dist} | {nus} | {np.mean(vs):.4f} ({min(vs):.4f}) |")
