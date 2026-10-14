import numpy as np
import scipy.sparse as sp

import cluster_libreville as cl


def planted(n=600, k=4, p_in=0.08, p_out=0.005, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(k), n // k)
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < np.where(truth[iu[0]] == truth[iu[1]], p_in, p_out)
    a = sp.coo_matrix((np.ones(keep.sum()), (iu[0][keep], iu[1][keep])), shape=(n, n))
    return truth, (a + a.T).tocsr()


def test_assess_ranks_true_labels_above_shuffled():
    truth, g = planted()
    t = cl.assess(g, truth)
    s = cl.assess(g, np.random.default_rng(1).permutation(truth))
    assert t["loglik"] > s["loglik"]
    assert t["description_length"] < s["description_length"]
    assert t["significance"] > s["significance"]
    assert t["surprise"].min() > s["surprise"].max()
    assert list(t["labels"]) == [0, 1, 2, 3] and t["size"].sum() == 600


def test_neosbm_keeps_true_labels_and_drops_shuffled():
    truth, g = planted()
    a = cl.neosbm(g, truth, seed=None)
    b = cl.neosbm(g, np.random.default_rng(1).permutation(truth), seed=None)
    assert a["defect"][-1] < 0.05
    assert b["defect"][-1] > 0.5
    assert np.all(np.diff(b["defect"]) >= -1e-12)
    assert b["loglik_gain"][-1] > 0


def test_string_labels_are_coded():
    truth, g = planted()
    names = np.array(["B", "Mono", "NK", "T"])[truth]
    r = cl.assess(g, names)
    assert list(r["labels"]) == ["B", "Mono", "NK", "T"]
