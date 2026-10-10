#!/usr/bin/env python3
"""export_graph.py <reference_neighbors.h5 | .h5ad> <out.npz>: UMAP connectivities -> CSR .npz
(indptr int64, indices int32, data float64, shape, cell_ids). Writes <dir>/READY_<size> when complete.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

src, out = sys.argv[1], Path(sys.argv[2])
with h5py.File(src) as f:
    g = f["connectivities"] if "connectivities" in f else f["obsp/connectivities"]
    indptr, indices, data = g["indptr"][:], g["indices"][:], g["data"][:]
    ids = f["cell_ids"][:] if "cell_ids" in f else f["obs/_index"][:]
n = len(indptr) - 1
c = sp.csr_matrix((data.astype(np.float64), indices, indptr.astype(np.int64)), shape=(n, n))
c.sum_duplicates(); c.sort_indices()
assert abs(c - c.T).max() < 1e-6, "not symmetric"
assert c.diagonal().max() == 0
tmp = out.with_suffix(".tmp.npz")
np.savez(tmp, indptr=c.indptr.astype(np.int64), indices=c.indices.astype(np.int32), data=c.data.astype(np.float64),
         shape=np.array([n, n], np.int64), cell_ids=np.asarray(ids).astype(str))
tmp.rename(out)
(out.parent / f"READY_{out.stem.split('_')[-1]}").touch()
print(out, n, c.nnz, "source dtype", data.dtype)
