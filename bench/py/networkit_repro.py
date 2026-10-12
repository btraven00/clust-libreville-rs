#!/usr/bin/env python3
"""networkit_repro.py GRAPH [--runs 20]: is networkit's ParallelLeiden reproducible? (python/.venv)

Called as scanpy PR #4170 calls it (scverse/scanpy, `flavor="networkit"`): networkit.setSeed(seed,
useThreadId=True), ParallelLeiden(g, iterations=3, gamma=1). Same protocol as the GVE determinism
table in $PHD/worklog/leiden-bench.md: same-seed reruns at 1 and 8 threads (distinct partitions,
mean / min pairwise ARI), agreement across thread counts, the seed floor (different seeds, 1 thread),
nu = (1 - same-seed ARI) / (1 - floor); modularity and time next to clust-libreville-rs.
"""
import itertools, statistics as st, sys, time
import numpy as np, scipy.sparse as sp
import networkit as nk
import cluster_libreville as cl

def ari(a, b):
    _, a = np.unique(a, return_inverse=True); _, b = np.unique(b, return_inverse=True)
    t = sp.coo_matrix((np.ones(len(a)), (a, b))).tocsr()
    c2 = lambda x: (x * (x - 1) / 2).sum()
    sij, si, sj, n = c2(t.data), c2(np.asarray(t.sum(1)).ravel()), c2(np.asarray(t.sum(0)).ravel()), c2(np.array([len(a)]))
    e = si * sj / n
    return (sij - e) / ((si + sj) / 2 - e)

def pairs(runs):
    return [ari(x, y) for x, y in itertools.combinations(runs, 2)]

path = sys.argv[1]
runs_n = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 20
z = np.load(path)
g = sp.csr_matrix((z["data"].astype(np.float64), z["indices"], z["indptr"]))
n = g.shape[0]
u = sp.triu(g).tocoo()
G = nk.Graph(n, weighted=True, directed=False)
G.addEdges((u.data, (u.row.astype(np.uint64), u.col.astype(np.uint64))))

def nk_leiden(seed, threads):
    nk.setNumberOfThreads(threads)
    nk.setSeed(seed, useThreadId=True)
    alg = nk.community.ParallelLeiden(G, iterations=3, gamma=1.0)
    t = time.perf_counter(); alg.run(); dt = time.perf_counter() - t
    return np.asarray(alg.getPartition().getVector()), dt

print(f"{path}: n {n}, edges {G.numberOfEdges()}, networkit {nk.__version__}")
same = {}
for threads in (1, 8):
    res = [nk_leiden(100, threads) for _ in range(runs_n)]
    labs = [r[0] for r in res]
    p = pairs(labs)
    distinct = len({tuple(np.unique(l, return_inverse=True)[1]) for l in labs})
    same[threads] = labs
    print(f"networkit {threads} thread(s), seed 100 x{runs_n}: distinct {distinct}/{runs_n}, same-seed ARI mean {st.mean(p):.3f} "
          f"(min {min(p):.3f}), Q {st.mean(cl.modularity(g, l) for l in labs):.5f}, clusters {st.mean(l.max() + 1 for l in labs):.1f}, "
          f"time median {st.median(r[1] for r in res):.3f} s")
cross = [ari(a, b) for a in same[1][:5] for b in same[8][:5]]
print(f"networkit 1 vs 8 threads (same seed): ARI mean {st.mean(cross):.3f} (min {min(cross):.3f})")
floor_runs = [nk_leiden(s, 1)[0] for s in range(200, 200 + runs_n)]
floor = st.mean(pairs(floor_runs))
for threads in (1, 8):
    print(f"nu at {threads} thread(s) vs networkit's seed floor {floor:.3f}: {(1 - st.mean(pairs(same[threads]))) / (1 - floor):.2f}")
t = time.perf_counter(); ours = cl.leiden(g, seed=100, threads=8); dt = time.perf_counter() - t
t1 = cl.leiden(g, seed=100, threads=1)
print(f"clust-libreville-rs, 8 threads: Q {cl.modularity(g, ours):.5f}, clusters {ours.max() + 1}, time {dt:.3f} s, "
      f"identical at 1 thread: {np.array_equal(ours, t1)}, ARI vs networkit (8t) mean {st.mean(ari(ours, l) for l in same[8]):.3f}")
