#!/usr/bin/env python3
"""make_fixtures.py: known-answer fixtures shared by the Rust, R and Python tests (python/.venv).
tests/fixtures/hao3k.mtx: the subgraph of hao2021_10k induced by its first 3000 cells (symmetric
MatrixMarket, weights with 17 significant digits, so every reader gets the same f64 values).
tests/fixtures/hao3k_cases.tsv: option sets; tests/fixtures/hao3k_labels.tsv: the labels the Rust
core (examples/run, built at the current commit) gives for each, one column per case (0-based).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures"
N = 3000
CASES = [  # name, seed, resolution, n_iterations, theta, min_gain
    ("default", 0, 1.0, -1, 0.01, 0.0),
    ("seed42", 42, 1.0, -1, 0.01, 0.0),
    ("res0.5", 1, 0.5, -1, 0.01, 0.0),
    ("res2", 1, 2.0, -1, 0.01, 0.0),
    ("niter1", 7, 1.0, 1, 0.01, 0.0),
    ("niter2", 7, 1.0, 2, 0.01, 0.0),
    ("theta0", 5, 1.0, -1, 0.0, 0.0),
    ("mingain", 9, 1.0, -1, 0.01, 1e-4),
]

z = np.load(ROOT / "bench" / "graphs" / "hao2021_10k.npz")
g = sp.csr_matrix((z["data"].astype(np.float64), z["indices"], z["indptr"]))[:N, :N].tocsr()
g.eliminate_zeros()
assert (g != g.T).nnz == 0
OUT.mkdir(parents=True, exist_ok=True)
mtx = OUT / "hao3k.mtx"
scipy.io.mmwrite(mtx, sp.tril(g).tocoo(), symmetry="symmetric", precision=17,
                 comment="hao2021 PBMC kNN connectivities (scanpy), first 3000 cells; detleiden test fixture")
back = scipy.io.mmread(mtx).tocsr()
back.sort_indices()
assert np.array_equal(back.toarray(), g.toarray()), "MatrixMarket round trip changed the graph"

run = ROOT / "target" / "release" / "examples" / "run"
cols = {}
with tempfile.TemporaryDirectory() as d:
    npz = Path(d) / "g.npz"
    np.savez(npz, indptr=back.indptr.astype(np.int64), indices=back.indices.astype(np.int32), data=back.data)
    for name, seed, res, niter, theta, gain in CASES:
        out = Path(d) / f"{name}.npy"
        subprocess.run([run, npz, "--seed", str(seed), "--resolution", repr(res), "--niter", str(niter), "--theta",
                        repr(theta), "--min-gain", repr(gain), "--threads", "4", "--out", out], check=True,
                       stdout=subprocess.DEVNULL)
        cols[name] = np.load(out)
with open(OUT / "hao3k_cases.tsv", "w") as f:
    f.write("name\tseed\tresolution\tn_iterations\ttheta\tmin_gain\n")
    for c in CASES:
        f.write("\t".join(map(str, c)) + "\n")
with open(OUT / "hao3k_labels.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for row in zip(*cols.values()):
        f.write("\t".join(map(str, row)) + "\n")
print({k: int(v.max()) + 1 for k, v in cols.items()}, file=sys.stderr)
