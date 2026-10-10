#!/usr/bin/env python3
"""ni_test.py <graph> <arm_dir>... --reference DIR [--reference DIR ...]   (sc-brrr python, -I)
Non-inferiority of each arm against a converged reference, with the procedure of
$PHD/worklog/stability-stats.md §1: D = θ_A − θ_R (θ_A = mean ARI(arm seed, reference seed),
θ_R = mean ARI over reference seed pairs), delete-one-seed jackknife SE over both seed sets,
NI if LB = D − t(0.95, min(n,m)−1)·SE > −δ, δ = 0.024 (frozen) and δ recomputed on this reference.
Also the contrast of each arm against the first arm (same reference): Δ = θ_arm − θ_first, jackknife
deleting seed i from both arms at once (arms share seed indices) and reference seed j.
Arms and references are sc-brrr-shaped members (rep<r>/…_clusters.tsv; bench_score.py makes them).
"""
import argparse
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

_s = importlib.util.spec_from_file_location("bench_score", Path(__file__).with_name("bench_score.py"))
bs = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bs)
ari = bs.ari
DELTA = 0.024


def reps(d, pos):
    tsv = next(Path(d).glob("*_clusters.tsv"))
    out = sorted((r, ari.coded(p, pos)) for r, p in ari.replicates(tsv) if r is not None)
    return [lab for _, lab in out]


def jackknife(stat, n, m):
    """SE² = (n−1)/n Σ_i (S₍₋ᵢ₎−S̄)² + (m−1)/m Σ_j (S₍₋ⱼ₎−S̄)², deleting arm seed i or reference seed j."""
    a = np.array([stat(i, None) for i in range(n)])
    b = np.array([stat(None, j) for j in range(m)])
    return np.sqrt((n - 1) / n * ((a - a.mean()) ** 2).sum() + (m - 1) / m * ((b - b.mean()) ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("graph")
    p.add_argument("arms", nargs="+")
    p.add_argument("--reference", action="append", required=True)
    a = p.parse_args()
    _, ids = bs.load(a.graph)
    pos = {x: i for i, x in enumerate(ids.astype(str))}
    arms = {Path(d).name: reps(d, pos) for d in a.arms}
    for ref_dir in a.reference:
        R = reps(ref_dir, pos)
        m = len(R)
        RR = np.full((m, m), np.nan)
        for j, k in combinations(range(m), 2):
            RR[j, k] = RR[k, j] = ari.ari(R[j], R[k])
        floor = lambda drop_j=None: np.nanmean(np.delete(np.delete(RR, drop_j, 0), drop_j, 1)) if drop_j is not None else np.nanmean(RR)
        delta_here = np.nanstd(RR[np.triu_indices(m, 1)], ddof=1)
        AR = {k: np.array([[ari.ari(x, r) for r in R] for x in v]) for k, v in arms.items()}
        print(f"\n{a.graph} vs {Path(ref_dir).name}: floor θ_R = {floor():.4f} (m={m}); δ frozen {DELTA}, δ here {delta_here:.4f}")
        print(f"{'arm':<14} {'n':>3} {'θ_A':>7} {'D':>8} {'SE':>7} {'LB':>8}  verdict(δ=0.024)  verdict(δ here)")
        for k, M in AR.items():
            n = M.shape[0]
            stat = lambda i, j: (np.delete(M, i, 0) if i is not None else M)[:, [c for c in range(m) if c != j]].mean() - floor(j)
            D = M.mean() - floor()
            se = jackknife(stat, n, m)
            tq = student_t.ppf(0.95, min(n, m) - 1)
            lb, ub = D - tq * se, D + tq * se
            v = lambda d: "NI" if lb > -d else ("inferior" if ub < -d else "inconclusive")
            print(f"{k:<14} {n:>3} {M.mean():>7.4f} {D:>+8.4f} {se:>7.4f} {lb:>+8.4f}  {v(DELTA):<16}  {v(delta_here)}")
        first, M0 = next(iter(AR.items()))
        for k, M in list(AR.items())[1:]:
            n = min(M.shape[0], M0.shape[0])
            stat = lambda i, j: ((np.delete(M[:n], i, 0) if i is not None else M[:n]) - (np.delete(M0[:n], i, 0) if i is not None else M0[:n]))[:, [c for c in range(m) if c != j]].mean()
            d = (M[:n] - M0[:n]).mean()
            se = jackknife(stat, n, m)
            tq = student_t.ppf(0.95, min(n, m) - 1)
            print(f"contrast {k} − {first}: Δ = {d:+.4f}, SE {se:.4f}, LB {d - tq * se:+.4f}, UB {d + tq * se:+.4f} → "
                  f"{'NI' if d - tq * se > -DELTA else 'not shown'} at δ=0.024")


if __name__ == "__main__":
    main()
