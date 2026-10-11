import numpy as np
import pytest
import scipy.sparse as sp

import cluster_libreville as cl


def two_cliques():
    e = [(a, b) for a in range(5) for b in range(a + 1, 5)] + [(a, b) for a in range(5, 10) for b in range(a + 1, 10)]
    e.append((4, 5))
    src, dst = np.array(e).T
    return src, dst


def sym(src, dst, w, n):
    a = sp.coo_matrix((w, (src, dst)), shape=(n, n))
    return (a + a.T).tocsr()


def planted(k=8, s=60, p_in=0.2, p_out=0.005, seed=7):
    rng = np.random.default_rng(seed)
    n = k * s
    block = np.repeat(np.arange(k), s)
    i, j = np.triu_indices(n, 1)
    p = np.where(block[i] == block[j], p_in, p_out)
    keep = rng.random(len(i)) < p
    i, j = i[keep], j[keep]
    w = rng.uniform(0.5, 1.5, len(i))
    return sym(i, j, w, n), block


def test_two_cliques():
    src, dst = two_cliques()
    labels = cl.leiden(sym(src, dst, np.ones(len(src)), 10))
    assert labels.dtype == np.int32
    assert labels.tolist() == [0] * 5 + [1] * 5


def test_threads_do_not_change_labels():
    g, _ = planted()
    a = cl.leiden(g, seed=3, threads=1)
    for t in (2, 4, 8, None):
        assert np.array_equal(cl.leiden(g, seed=3, threads=t), a)


def test_seeded_and_size_ordered():
    g, _ = planted()
    a = cl.leiden(g, seed=11)
    assert np.array_equal(cl.leiden(g, seed=11), a)
    sizes = np.bincount(a)
    assert np.all(np.diff(sizes) <= 0)
    assert set(a.tolist()) == set(range(len(sizes)))


def test_planted_partition_recovered():
    g, block = planted()
    a = cl.leiden(g, seed=1)
    assert len(set(a.tolist())) == 8
    assert len(set(zip(a.tolist(), block.tolist()))) == 8


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "array", "f32", "i64", "tuple"])
def test_input_formats_agree(fmt):
    g, _ = planted()
    ref = cl.leiden(g, seed=5)
    if fmt == "csr":
        x = g
    elif fmt == "csc":
        x = g.tocsc()
    elif fmt == "coo":
        x = g.tocoo()
    elif fmt == "array":
        x = sp.csr_array(g)
    elif fmt == "f32":  # float32 weights (scanpy): same graph only if the weights are exactly f32
        x = g.astype(np.float32)
        ref = cl.leiden(sp.csr_matrix(x, dtype=np.float64), seed=5)
    elif fmt == "i64":
        x = g.copy()
        x.indices = x.indices.astype(np.int64)
        x.indptr = x.indptr.astype(np.int64)
    else:
        x = (g.indptr, g.indices, g.data)
    assert np.array_equal(cl.leiden(x, seed=5), ref)


def test_edge_list_matches_matrix():
    g, _ = planted()
    u = sp.triu(g).tocoo()
    a = cl.leiden_edges(u.row, u.col, u.data, seed=2)
    assert np.array_equal(a, cl.leiden(g, seed=2))
    # unweighted, repeated edges add up, isolated trailing vertices via n
    src, dst = two_cliques()
    b = cl.leiden_edges(np.r_[src, src[:1]], np.r_[dst, dst[:1]], n=12)
    assert len(b) == 12 and b[10] != b[11]


def test_polars_columns():
    pl = pytest.importorskip("polars")
    g, _ = planted()
    u = sp.triu(g).tocoo()
    df = pl.DataFrame({"src": u.row, "dst": u.col, "w": u.data})
    a = cl.leiden_edges(df["src"], df["dst"], df["w"], seed=4)
    assert np.array_equal(a, cl.leiden(g, seed=4))


def test_options_pass_through():
    g, _ = planted()
    assert cl.leiden(g, resolution=5).max() > cl.leiden(g, resolution=0.2).max()
    assert len(cl.leiden(g, n_iterations=1)) == g.shape[0]
    assert len(cl.leiden(g, min_gain=1e-3)) == g.shape[0]


def test_modularity():
    g, block = planted()
    a = cl.leiden(g, seed=2)
    # Newman modularity by hand
    m2 = g.sum()
    k = np.asarray(g.sum(axis=1)).ravel()
    same = a[g.tocoo().row] == a[g.tocoo().col]
    q = (g.tocoo().data[same].sum() - sum(k[a == c].sum() ** 2 for c in np.unique(a)) / m2) / m2
    assert cl.modularity(g, a) == pytest.approx(q, abs=1e-12)
    assert cl.modularity(g, a.astype(str)) == pytest.approx(q, abs=1e-12)
    assert cl.modularity(g, a) >= cl.modularity(g, block) - 1e-3


def test_invalid_input():
    g, _ = planted()
    bad = g.tolil()
    bad[0, 1] = 7.0
    with pytest.raises(ValueError, match="symmetric"):
        cl.leiden(bad.tocsr())
    with pytest.raises(ValueError, match="square"):
        cl.leiden(sp.csr_matrix((3, 4)))
    with pytest.raises(ValueError):
        cl.leiden(g, threads=0)
    with pytest.raises(TypeError):
        cl.leiden(np.eye(3))


def test_scanpy_style():
    ad = pytest.importorskip("anndata")
    g, _ = planted()
    a = ad.AnnData(np.zeros((g.shape[0], 1), dtype=np.float32))
    a.obsp["connectivities"] = g
    a.uns["neighbors"] = {"connectivities_key": "connectivities"}
    cl.tl.leiden(a, resolution=1.0, random_state=3)
    assert a.obs["leiden"].dtype == "category"
    assert list(a.obs["leiden"].cat.categories) == [str(c) for c in range(a.obs["leiden"].nunique())]
    assert np.array_equal(a.obs["leiden"].cat.codes.to_numpy(), cl.leiden(g, seed=3))
    assert a.uns["leiden"]["params"]["flavor"] == "cluster-libreville"
    b = cl.tl.leiden(a, obsp="connectivities", key_added="cl", copy=True)
    assert "cl" in b.obs and "cl" not in a.obs
