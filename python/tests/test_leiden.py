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


def test_check_symmetry_off():
    g, _ = planted()
    assert np.array_equal(cl.leiden(g, seed=6, check_symmetry=False), cl.leiden(g, seed=6))
    bad = g.tolil()
    bad[0, 1] = 7.0
    labels = cl.leiden(bad.tocsr(), check_symmetry=False)  # meaningless, but no error or crash
    assert len(labels) == g.shape[0]
    with pytest.raises(ValueError, match="out of range"):  # bounds are always checked
        cl.leiden((np.array([0, 1]), np.array([5], dtype=np.int32), np.array([1.0])), check_symmetry=False)


def test_return_info():
    g, _ = planted()
    labels, info = cl.leiden(g, seed=1, return_info=True)
    assert np.array_equal(labels, cl.leiden(g, seed=1))
    assert info["converged"] and info["stop"] == "converged" and info["iterations"] >= 1
    assert info["modularity"] == pytest.approx(cl.modularity(g, labels), abs=1e-12)
    _, one = cl.leiden(g, seed=1, n_iterations=1, return_info=True)
    assert one["iterations"] == 1 and one["stop"] == "n_iterations" and not one["converged"]


def test_champ_envelope_is_the_pools_best():
    g, _ = planted()
    res, seeds = np.logspace(-1, 1, 12), np.arange(3)
    c = cl.champ(g, res, seeds)
    assert 2 <= len(c.labels) < res.size * seeds.size
    assert c.gamma_lo[0] == res[0] and c.gamma_hi[-1] == res[-1]
    assert np.array_equal(c.gamma_hi[:-1], c.gamma_lo[1:])
    for i in range(len(c.labels)):  # provenance: the grid point reproduces the partition
        assert np.array_equal(cl.leiden(g, resolution=c.resolution[i], seed=int(c.seed[i])), c.labels[i])
    pool = [cl.leiden(g, resolution=r, seed=int(s)) for r in res for s in seeds]
    for x in np.geomspace(res[0], res[-1], 60):
        best = max(cl.modularity(g, p, resolution=x) for p in pool)
        i = np.searchsorted(c.gamma_hi, x)
        assert cl.modularity(g, c.labels[i], resolution=x) >= best - 1e-12
    assert np.all(np.diff(c.n_communities) >= 0)  # finer as the resolution grows
    assert c.labels[c.best].max() + 1 == 8  # the planted partition holds the widest domain


def test_champ_threads_and_validation():
    g, _ = planted()
    a = cl.champ(g, np.logspace(-1, 1, 6), np.arange(2), threads=1)
    b = cl.champ(g, np.logspace(-1, 1, 6), np.arange(2), threads=4)
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.gamma_lo, b.gamma_lo)
    for bad in ([], [0.0, 1.0], [np.nan]):
        with pytest.raises(ValueError):
            cl.champ(g, bad)
    with pytest.raises(ValueError):
        cl.champ(g, [1.0], seeds=[-1])


def test_scanpy_style_champ(tmp_path):
    ad = pytest.importorskip("anndata")
    g, _ = planted()
    a = ad.AnnData(np.zeros((g.shape[0], 1), dtype=np.float32))
    a.obsp["connectivities"] = g
    a.uns["neighbors"] = {"connectivities_key": "connectivities"}
    res, seeds = np.logspace(-1, 1, 8), np.arange(2)
    cl.tl.champ(a, res, seeds)
    c = cl.champ(g, res, seeds)
    u = a.uns["champ"]
    assert a.obs["champ"].dtype == "category"
    assert np.array_equal(a.obs["champ"].cat.codes.to_numpy(), c.labels[c.best])
    assert np.array_equal(a.obsm["champ"], c.labels.T) and u["best"] == c.best
    assert np.array_equal(u["domains"]["gamma_hi"], c.gamma_hi)
    a.write_h5ad(tmp_path / "a.h5ad")  # uns stays writable
    b = cl.tl.champ(a, res, seeds, key_added="ch", copy=True)
    assert "ch" in b.obs and "ch" not in a.obs


def _nvi(a, b):  # PyGenStability's evaluate_NVI, in numpy
    _, ab = np.unique(np.stack([a, b]), axis=1, return_counts=True)
    h = lambda c: -np.sum(c / c.sum() * np.log(c / c.sum()))  # noqa: E731
    ha, hb, hab = h(np.unique(a, return_counts=True)[1]), h(np.unique(b, return_counts=True)[1]), h(ab)
    return 0.0 if hab < 1e-12 else (2 * hab - ha - hb) / hab


def test_champ_agreement_scores():
    g, _ = planted()
    res, seeds = np.logspace(-1, 1, 12), np.arange(3)
    c = cl.champ(g, res, seeds)
    assert c.metric == "nvi" and np.isnan(c.to_prev[0])
    for i in range(1, len(c.labels)):
        assert c.to_prev[i] == pytest.approx(_nvi(c.labels[i - 1], c.labels[i]), abs=1e-12)
    i = c.best
    mates = [cl.leiden(g, resolution=c.resolution[i], seed=int(s)) for s in seeds if s != c.seed[i]]
    assert c.to_seeds[i] == pytest.approx(np.mean([_nvi(c.labels[i], m) for m in mates]), abs=1e-12)
    a = cl.champ(g, res, seeds, metric="ami")
    assert np.array_equal(a.labels, c.labels) and np.all(a.to_neighbours <= 1 + 1e-12)
    sk = pytest.importorskip("sklearn.metrics")
    for i in range(1, len(a.labels)):
        ref = sk.adjusted_mutual_info_score(a.labels[i - 1], a.labels[i], average_method="max")
        assert a.to_prev[i] == pytest.approx(ref, abs=1e-9)
    with pytest.raises(ValueError):
        cl.champ(g, res, metric="ari")
