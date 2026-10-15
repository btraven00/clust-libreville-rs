#!/usr/bin/env python3
"""build_graph.py {154k|1M}: `cd $PHD/brrr-gve && pixi run python -I build_graph.py 154k` -> bench/graphs/hao2021_154k.npz / syn_1M.npz + READY_<size>.
154k: brrr-gve method.pca (float32 covariance) + exact kNN (faiss flat), --graph python (float64 UMAP connectivities).
1M: $PHD/sc-brrr-local/scale1m/make_emb.py's recipe (resample the 154k embedding + Gaussian jitter, rng 0) on *this* 154k
embedding (make_emb uses stream.pca), then faiss hnsw (brrr-gve defaults: M 16, efC 100, efS 64), --graph cxx (float32 weights).
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(PHD / "brrr-gve"))
import anndata as ad
import faiss
import method
import os
PHD = Path(os.environ["PHD"])  # the directory holding the brrr-gve, sc-brrr and clust-libreville-rs checkouts

GRAPHS = Path(__file__).resolve().parents[1] / "graphs"
size = sys.argv[1]
emb_path = GRAPHS / "emb154k.npy"
p = lambda **o: SimpleNamespace(n_components=50, n_neighbors=15, seed=0, resolution=1.0, options=o)

if emb_path.exists():
    E = np.load(emb_path); ids154 = np.load(GRAPHS / "ids154k.npy")
else:
    a = ad.read_h5ad(PHD / "sc-brrr/data/prep/hao2021_all.h5ad")
    t = time.time(); E = method.pca(a, p()); print("pca", a.shape, time.time() - t, flush=True)
    ids154 = np.asarray(a.obs_names).astype(str); del a
    np.save(emb_path, E); np.save(GRAPHS / "ids154k.npy", ids154)

if size == "154k":
    ids, t = ids154, time.time()
    g = method.knn(E, p(index="flat", graph="python"))
else:
    ix = faiss.IndexFlatL2(50); ix.add(E[:20000]); d, _ = ix.search(E[:20000], 2)
    sd = 0.5 * np.median(np.sqrt(d[:, 1])) / np.sqrt(50)
    n = {"1M": 1_000_000, "2M": 2_000_000}[size]
    src = np.random.default_rng(0).integers(0, len(E), n)
    Y = (E[src] + np.random.default_rng(1).normal(0, sd, (n, 50))).astype(np.float32)
    ids = np.array([f"syn{i}_{ids154[s]}" for i, s in enumerate(src)])
    t = time.time()
    g = method.knn(Y, p(index="hnsw", graph="cxx"))
print("knn+graph", size, time.time() - t, flush=True)
c = g.connectivities.tocsr(); c.sum_duplicates(); c.sort_indices()
out = GRAPHS / (f"hao2021_{size}.npz" if size == "154k" else f"syn_{size}.npz")  # synthetic: not hao2021
tmp = out.with_suffix(".tmp.npz")
np.savez(tmp, indptr=c.indptr.astype(np.int64), indices=c.indices.astype(np.int32), data=c.data.astype(np.float64),
         shape=np.array(c.shape, np.int64), cell_ids=ids)
tmp.rename(out)
(GRAPHS / f"READY_{size}").touch()
print(out, c.shape, c.nnz, "source dtype", g.connectivities.dtype)
