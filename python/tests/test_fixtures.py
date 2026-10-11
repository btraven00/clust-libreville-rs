"""Known-answer fixtures shared with the Rust and R tests (../../tests/fixtures)."""
import csv
from pathlib import Path

import numpy as np
import pytest

import detleiden

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
pytestmark = pytest.mark.skipif(not (FIX / "hao3k.mtx").exists(), reason="fixtures live in the source repository")


def cases():
    with open(FIX / "hao3k_cases.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


@pytest.fixture(scope="module")
def hao3k():
    scipy_io = pytest.importorskip("scipy.io")
    g = scipy_io.mmread(FIX / "hao3k.mtx").tocsr()
    g.sort_indices()
    labels = np.loadtxt(FIX / "hao3k_labels.tsv", delimiter="\t", skiprows=1, dtype=np.int32)
    with open(FIX / "hao3k_labels.tsv") as f:
        names = f.readline().strip().split("\t")
    return g, {n: labels[:, k] for k, n in enumerate(names)}


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["name"])
def test_known_answers(hao3k, case):
    g, want = hao3k
    kw = dict(seed=int(case["seed"]), resolution=float(case["resolution"]), n_iterations=int(case["n_iterations"]),
              theta=float(case["theta"]), min_gain=float(case["min_gain"]))
    for threads in (1, 4):
        assert np.array_equal(detleiden.leiden(g, threads=threads, **kw), want[case["name"]])
    # the same graph as an edge list and as float32 (scanpy's dtype; these weights are f32-exact)
    u = g.tocoo()
    keep = u.row <= u.col
    assert np.array_equal(detleiden.leiden_edges(u.row[keep], u.col[keep], u.data[keep], n=g.shape[0], **kw),
                          want[case["name"]])
    assert np.array_equal(detleiden.leiden(g.astype(np.float32), **kw), want[case["name"]])
