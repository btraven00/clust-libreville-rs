"""Distinct partitions / coarse-graph hash sequences per variant: summarize.py <out/leiden-rs/graph dir>"""
import glob, json, os, re, sys
from collections import Counter

root = sys.argv[1]
for var in sorted(os.listdir(root)):
    rows = []
    for j in sorted(glob.glob(f"{root}/{var}/*t/run*.json")):
        d = json.load(open(j))
        tr = open(j[:-5] + ".trace").read() if os.path.exists(j[:-5] + ".trace") else ""
        d["levels"] = " ".join(re.findall(r"agg_hash=(\w+)", tr))
        d["dir"] = j.split("/")[-2]
        rows.append(d)
    if not rows:
        continue
    print(f"== {var}: {len(rows)} runs, n={rows[0]['n']}, niter={rows[0]['niter']}, seed={rows[0]['seed']}")
    for t in sorted({r["dir"] for r in rows}, key=lambda s: int(s[:-1])):
        rs = [r for r in rows if r["dir"] == t]
        print(f"  {t:>4}: runs={len(rs):2} partitions={len({r['labels_hash'] for r in rs})} "
              f"levelhash_seqs={len({r['levels'] for r in rs})} k={sorted(Counter(r['n_clusters'] for r in rs).items())} "
              f"Q=[{min(r['quality_impl'] for r in rs):.6f},{max(r['quality_impl'] for r in rs):.6f}]")
    print(f"   all: partitions={len({r['labels_hash'] for r in rows})} "
          f"levelhash_seqs={len({r['levels'] for r in rows if r['levels']})} "
          f"disconnected={sum(r['disconnected'] for r in rows)}")

# Seed sweeps: per seed (run<k>), do stock 1t / 8t / 16t and seqagg 8t give the same partition?
def h(p):
    return json.load(open(p))["labels_hash"] if os.path.exists(p) else None
cols = ["stock-seeds/1t", "stock-seeds/8t", "stock-seeds/16t", "seqagg-seeds/8t"]
ks = sorted({int(re.search(r"run(\d+)", p).group(1)) for p in glob.glob(f"{root}/stock-seeds/1t/run*.json")})
if ks:
    flips = {c: 0 for c in cols[1:]}
    for k in ks:
        hs = [h(f"{root}/{c}/run{k}.json") for c in cols]
        for c, x in zip(cols[1:], hs[1:]):
            flips[c] += x is not None and x != hs[0]
    print(f"== seed sweep, {len(ks)} seeds: partitions differing from stock 1t: {flips}")
