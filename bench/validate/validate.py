# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.2.6", "scipy==1.15.3", "igraph==0.11.8", "leidenalg==0.10.2", "networkit==11.2.2", "huggingface-hub==1.28.0"]
# ///
"""validate.py [--graphs DIR] [--sizes 10k,50k,100k,154k] [--seeds 20] [--threads 8]

Reproduces docs/design.md's Validation section from the graphs alone:
`cargo build --release --example run && uv run bench/validate/validate.py`.

- quality: mean modularity of each detleiden arm (default = strict convergence, min_gain 1e-5,
  OrderFree) and of two converged references: leidenalg (n_iterations=-1, its own fixed point) and
  igraph iterated one iteration at a time from the previous membership until it no longer changes
  (python-igraph's own n_iterations=-1 stopped after one iteration before C core 0.10.17);
- non-inferiority of each arm against each reference (ARI; procedure below);
- reproducibility: same-seed reruns at --threads threads of detleiden and of networkit's
  ParallelLeiden (called as scanpy PR #4170 calls it).

Non-inferiority: arm seeds a_1..a_n, reference seeds r_1..r_m (disjoint ranges). Floor
θ_R = mean ARI(r_j, r_k) over pairs, θ_A = mean ARI(a_i, r_j), D = θ_A − θ_R. SE by delete-one-seed
jackknife of the two-sample U-statistic (Arvesen 1969): SE² = (n−1)/n Σ_i (D₍₋ᵢ₎ − D̄)² +
(m−1)/m Σ_j (D₍₋ⱼ₎ − D̄)². NI if LB = D − t(0.95, min(n, m) − 1)·SE > −δ, inferior if UB < −δ,
else inconclusive. δ = 0.024 (frozen: the pooled SD of reference seed-pair ARIs when the protocol was
fixed) and δ recomputed on this reference.

Graphs: `<size>` reads DIR/hao2021_<size>.npz (CSR: indptr, indices, data, shape; symmetric, no
diagonal); a missing one is downloaded from the Hugging Face dataset btraven/clust-libreville-graphs
(provenance in its card) and checked against its sha256. Labels are cached in bench/out/validate/; the report goes to stdout and
bench/out/validate/report.md.
"""
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # leidenalg's docstrings, at import

import argparse
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "target" / "release" / "examples" / "run"
OUT = ROOT / "bench" / "out" / "validate"
DELTA = 0.024
ARM_SEEDS, REF_SEEDS = range(0, 1000), range(1000, 2000)  # disjoint; the first --seeds of each
ARMS = {"strict": [], "min_gain": ["--min-gain", "1e-5"], "orderfree": ["--apply", "orderfree"]}
REFS = ("leidenalg", "igraph_fix")
DATASET = "btraven/clust-libreville-graphs"
SHA256 = {
    "10k": "2db665667c38b66b20bc6abc0a5d3faf35c55fd61aecee6deda4ce937de9ca10",
    "50k": "20c262b30546380689d978b53d17fe44fe92cfaac9e260aacf2b9f807d7e1ce0",
    "100k": "bfa7c3de7e122999d861ae4efe84284a241bcf05045f5189ca7c33192398d826",
    "154k": "d0e16995aebeb58d21aea6999c737b281a10ccc1a3de53c9ba127d6ac2968a8d",
}


def fetch(graphs, size):
    """DIR/hao2021_<size>.npz, downloaded if missing; its sha256 checked either way."""
    import hashlib
    path = Path(graphs) / f"hao2021_{size}.npz"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(DATASET, path.name, repo_type="dataset", local_dir=graphs)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256[size], f"{path}: not the published graph"
    return str(path)


def load(path):
    z = np.load(path)
    return sp.csr_matrix((z["data"].astype(np.float64), z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def modularity(a, labels):
    """Q = Σ_c (in_c / 2m − (K_c / 2m)²), resolution 1; a symmetric, weights counted in both directions."""
    k = np.asarray(a.sum(1)).ravel()
    m2 = k.sum()
    coo = a.tocoo()
    inside = coo.data[labels[coo.row] == labels[coo.col]].sum()
    tot = np.bincount(labels, weights=k)
    return inside / m2 - ((tot / m2) ** 2).sum()


def ari(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    t = sp.coo_matrix((np.ones(len(a)), (a, b))).tocsr()
    c2 = lambda x: (x * (x - 1) / 2).sum()
    sij, si, sj, n = c2(t.data), c2(np.asarray(t.sum(1)).ravel()), c2(np.asarray(t.sum(0)).ravel()), c2(np.array([len(a)]))
    e = si * sj / n
    return (sij - e) / ((si + sj) / 2 - e)


def detleiden(npz, seed, extra, threads, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([RUN, npz, "--seed", str(seed), "--threads", str(threads), "--trusted", "--out", out, *extra],
                   check=True, stdout=subprocess.DEVNULL)
    labels = np.load(out).astype(np.int64)
    out.unlink()
    return labels


_G = {}


def reference(job):
    """One converged reference run, single-threaded (a worker process per seed)."""
    impl, npz, seed = job
    import igraph as ig
    if npz not in _G:
        a = sp.triu(load(npz), 1).tocoo()
        g = ig.Graph(n=a.shape[0], edges=list(zip(a.row.tolist(), a.col.tolist())), directed=False)
        g.es["weight"] = a.data.tolist()
        _G[npz] = g
    g = _G[npz]
    if impl == "leidenalg":
        import leidenalg as la
        p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0,
                              n_iterations=-1, seed=seed)
        return np.array(p.membership)
    random.seed(seed)
    ig.set_random_number_generator(random)
    m = None
    for _ in range(1000):
        nxt = g.community_leiden(objective_function="modularity", weights="weight", resolution=1.0, n_iterations=1,
                                 initial_membership=m).membership
        if m is not None and canon(nxt) == canon(m):
            break
        m = nxt
    return np.array(m)


def canon(labels):
    _, first = np.unique(labels, return_index=True)
    order = {labels[i]: r for r, i in enumerate(sorted(first))}
    return tuple(order[x] for x in labels)


def cached(path, make):
    if path.exists():
        return np.load(path)
    x = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def ni(arm, ref):
    """D, SE, LB, UB of arm vs ref (lists of label arrays), the floor θ_R, and δ recomputed on ref."""
    n, m = len(arm), len(ref)
    rr = np.full((m, m), np.nan)
    for j, k in combinations(range(m), 2):
        rr[j, k] = rr[k, j] = ari(ref[j], ref[k])
    ar = np.array([[ari(x, r) for r in ref] for x in arm])

    def floor(j=None):
        return np.nanmean(rr if j is None else np.delete(np.delete(rr, j, 0), j, 1))

    def d(i=None, j=None):
        a = ar if i is None else np.delete(ar, i, 0)
        return (a if j is None else np.delete(a, j, 1)).mean() - floor(j)

    di, dj = np.array([d(i=i) for i in range(n)]), np.array([d(j=j) for j in range(m)])
    se = np.sqrt((n - 1) / n * ((di - di.mean()) ** 2).sum() + (m - 1) / m * ((dj - dj.mean()) ** 2).sum())
    tq = student_t.ppf(0.95, min(n, m) - 1)
    return d(), se, d() - tq * se, d() + tq * se, floor(), np.nanstd(rr[np.triu_indices(m, 1)], ddof=1)


def verdict(lb, ub, delta):
    return "NI" if lb > -delta else ("inferior" if ub < -delta else "inconclusive")


def networkit_runs(a, seed, threads, runs):
    import networkit as nk
    nk.setNumberOfThreads(threads)
    u = sp.triu(a, 1).tocoo()
    g = nk.Graph(a.shape[0], weighted=True)
    for x, y, w in zip(u.row.tolist(), u.col.tolist(), u.data.tolist()):
        g.addEdge(x, y, w)
    out = []
    for _ in range(runs):
        nk.setSeed(seed, True)  # useThreadId=True, as scanpy PR #4170
        c = nk.community.ParallelLeiden(g, iterations=3, gamma=1.0)
        c.run()
        out.append(np.array(c.getPartition().getVector()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", default=str(ROOT / "bench" / "graphs"))
    p.add_argument("--sizes", default="10k,50k,100k,154k")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--reruns", type=int, default=10, help="same-seed reruns for reproducibility")
    o = p.parse_args()
    assert RUN.exists(), f"{RUN}: cargo build --release --example run"
    report = [f"# Validation (bench/validate/validate.py, {o.seeds} seeds per arm and reference)\n"]
    say = lambda s="": (print(s, flush=True), report.append(s))
    for size in o.sizes.split(","):
        name = f"hao2021_{size}"
        npz = fetch(o.graphs, size)
        a = load(npz)
        d = OUT / name
        say(f"\n## {name} (n {a.shape[0]})\n")
        labels = {}
        for arm, extra in ARMS.items():
            labels[arm] = [cached(d / arm / f"{s}.npy", lambda s=s: detleiden(npz, s, extra, o.threads, str(d / arm / f"{s}.tmp.npy")))
                           for s in ARM_SEEDS[:o.seeds]]
        with ProcessPoolExecutor() as ex:
            for impl in REFS:
                todo = [s for s in REF_SEEDS[:o.seeds] if not (d / impl / f"{s}.npy").exists()]
                t = time.perf_counter()
                for s, lab in zip(todo, ex.map(reference, [(impl, npz, s) for s in todo])):
                    (d / impl).mkdir(parents=True, exist_ok=True)
                    np.save(d / impl / f"{s}.npy", lab)
                if todo:
                    print(f"  {impl}: {len(todo)} runs in {time.perf_counter() - t:.0f} s", flush=True)
                labels[impl] = [np.load(d / impl / f"{s}.npy") for s in REF_SEEDS[:o.seeds]]
        say("| arm | mean Q | sd Q |\n|---|---|---|")
        for k, v in labels.items():
            q = [modularity(a, x) for x in v]
            say(f"| {k} | {np.mean(q):.5f} | {np.std(q, ddof=1):.5f} |")
        for ref in REFS:
            res = {arm: ni(labels[arm], labels[ref]) for arm in ARMS}
            floor, dh = next(iter(res.values()))[4:]
            say(f"\nvs {ref}: floor θ_R {floor:.4f}, δ here {dh:.4f}\n")
            say("| arm | D | SE | LB | δ = 0.024 | δ here |\n|---|---|---|---|---|---|")
            for arm, (dd, se, lb, ub, _, _) in res.items():
                say(f"| {arm} | {dd:+.4f} | {se:.4f} | {lb:+.4f} | {verdict(lb, ub, DELTA)} | {verdict(lb, ub, dh)} |")
        say("\nSame-seed reruns at %d threads (seed %d): distinct partitions, mean / min pairwise ARI\n" % (o.threads, ARM_SEEDS[0]))
        say("| method | distinct | mean ARI | min ARI |\n|---|---|---|---|")
        reruns = {"detleiden": [detleiden(npz, ARM_SEEDS[0], [], o.threads, str(OUT / "tmp.npy")) for _ in range(o.reruns)],
                  "networkit ParallelLeiden": networkit_runs(a, ARM_SEEDS[0], o.threads, o.reruns)}
        for k, v in reruns.items():
            pairs = [ari(x, y) for x, y in combinations(v, 2)]
            say(f"| {k} | {len({canon(x) for x in v})} | {np.mean(pairs):.4f} | {min(pairs):.4f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
