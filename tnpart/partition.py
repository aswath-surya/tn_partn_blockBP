"""Placers (π: sites → ranks) and the placement objective (brief §2.2–2.4, E10).

Every placer has signature ``placer(g: TNGraph, P: int, **kw) -> np.ndarray``.
``evaluate`` computes the (λ−1)-weighted communication in byte-equivalents plus
memory balance — the objective the χ-aware placers minimise and the ILP solves
exactly.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .costmodel import Machine
from .tngraph import TNGraph

_DEFAULT_MACHINE = Machine()


def _cost_kw(kw):
    return dict(machine=kw.get("machine", _DEFAULT_MACHINE), gauge_iters=kw.get("gauge_iters", 5),
                chi_b=kw.get("chi_b", 32), n_bmps_passes=kw.get("n_bmps_passes", 2),
                include_rows=kw.get("include_rows", True))


# ------------------------------------------------------------------ evaluation

def evaluate(g: TNGraph, pi, **kw):
    ck = _cost_kw(kw)
    pi = np.asarray(pi)
    P = int(kw.get("P", pi.max() + 1))
    pins, cost = g.nets(chi_aware=True, **ck)
    lam = np.array([g.net_lambda(pi, p) for p in pins])
    comm = float(np.sum(cost * (lam - 1)))
    m = g.m
    cut = lam[:m] > 1
    bps = ck["machine"].bytes_per_scalar
    part_bytes = np.zeros(P)
    np.add.at(part_bytes, pi, g.w * bps)
    ideal = part_bytes.sum() / P
    # actual hand-off count along rows/cols (boundaries, not λ−1)
    handoffs = 0
    for line in g.rows() + g.cols():
        handoffs += sum(pi[u] != pi[v] for u, v in zip(line[:-1], line[1:]))
    return dict(
        comm=comm, comm_bonds=float(np.sum(cost[:m] * cut)), comm_lines=float(np.sum(cost[m:] * (lam[m:] - 1))),
        cut_bonds=int(cut.sum()), cut_chi_sum=int(g.chi[cut].sum()), line_handoffs=int(handoffs),
        part_bytes=part_bytes.tolist(), peak_bytes=float(part_bytes.max()),
        imbalance=float(part_bytes.max() / ideal - 1.0) if ideal > 0 else 0.0,
        parts_used=int(len(set(pi.tolist()))),
    )


# ------------------------------------------------------------------ geometric

def _contiguous_split(weights, P):
    """Optimal contiguous min-max partition of a sequence (DP). Returns labels."""
    a = np.asarray(weights, dtype=float)
    n = len(a)
    P = min(P, n)
    pre = np.concatenate([[0.0], np.cumsum(a)])
    INF = float("inf")
    dp = np.full((P + 1, n + 1), INF)
    arg = np.zeros((P + 1, n + 1), dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, P + 1):
        for i in range(1, n + 1):
            best, bj = INF, 0
            for j in range(k - 1, i):
                v = max(dp[k - 1, j], pre[i] - pre[j])
                if v < best:
                    best, bj = v, j
            dp[k, i], arg[k, i] = best, bj
    labels = np.zeros(n, dtype=int)
    i = n
    for k in range(P, 0, -1):
        j = arg[k, i]
        labels[j:i] = k - 1
        i = j
    return labels


def place_rows(g: TNGraph, P, **kw):
    if P > g.Lx:
        return place_tiles(g, P, **kw)
    rows = g.rows()
    lab = _contiguous_split([g.w[r].sum() for r in rows], P)
    pi = np.zeros(g.n, dtype=int)
    for r, l in zip(rows, lab):
        pi[r] = l
    return pi


def place_cols(g: TNGraph, P, **kw):
    if P > g.Ly:
        return place_tiles(g, P, **kw)
    cols = g.cols()
    lab = _contiguous_split([g.w[c].sum() for c in cols], P)
    pi = np.zeros(g.n, dtype=int)
    for c, l in zip(cols, lab):
        pi[c] = l
    return pi


def _factor(P, Lx, Ly):
    best = None
    for px in range(1, P + 1):
        if P % px:
            continue
        py = P // px
        if px > Lx or py > Ly:
            continue
        score = abs(np.log((px / py) / (Lx / Ly)))
        if best is None or score < best[0]:
            best = (score, px, py)
    if best is None:
        raise ValueError(f"cannot tile {Lx}x{Ly} into {P} parts")
    return best[1], best[2]


def place_tiles(g: TNGraph, P, **kw):
    px, py = _factor(P, g.Lx, g.Ly)
    rl = _contiguous_split([g.w[r].sum() for r in g.rows()], px)
    cl = _contiguous_split([g.w[c].sum() for c in g.cols()], py)
    pi = np.zeros(g.n, dtype=int)
    for (i, j), v in g.sidx.items():
        pi[v] = rl[i] * py + cl[j]
    return pi


def place_random(g: TNGraph, P, seed=0, **kw):
    rng = np.random.default_rng(seed)
    pi = np.arange(g.n) % P
    rng.shuffle(pi)
    return pi


# ------------------------------------------------------------------ KaHyPar

def _kahypar_ini(objective):
    import cotengra
    prof = "km1_kKaHyPar_sea20.ini" if objective == "km1" else "cut_kKaHyPar_sea20.ini"
    return os.path.join(os.path.dirname(cotengra.__file__), "pathfinders", "kahypar_profiles", prof)


def _scale_int(x, cap=1_000_000):
    x = np.asarray(x, dtype=float)
    mx = x.max() if x.size else 1.0
    s = cap / mx if mx > cap else 1.0
    return [max(1, int(round(v * s))) for v in x]


def place_kahypar(g: TNGraph, P, objective="km1", chi_aware=True, eps=0.05, seed=0, **kw):
    import kahypar

    ck = _cost_kw(kw)
    pins, cost = g.nets(chi_aware=chi_aware, **ck)
    node_w = g.w if chi_aware else np.ones(g.n)
    index_vec, edge_vec = [0], []
    for p in pins:
        edge_vec += list(p)
        index_vec.append(len(edge_vec))
    hg = kahypar.Hypergraph(g.n, len(pins), index_vec, edge_vec, P, _scale_int(cost), _scale_int(node_w))
    ctx = kahypar.Context()
    ctx.loadINIconfiguration(_kahypar_ini(objective))
    ctx.setK(P)
    ctx.setEpsilon(eps)
    ctx.setSeed(seed)
    ctx.suppressOutput(True)
    kahypar.partition(hg, ctx)
    return np.array([hg.blockID(v) for v in range(g.n)], dtype=int)


# ------------------------------------------------------------------ exact ILP

def place_ilp(g: TNGraph, P, eps=0.05, time_limit=120.0, **kw):
    """Exact minimiser of ``evaluate()['comm']`` under the balance constraint
    (scipy HiGHS MILP). Small instances only (n ≲ 40) — the optimality-gap
    reference demanded in brief Part IX."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    ck = _cost_kw(kw)
    pins, cost = g.nets(chi_aware=True, **ck)
    n, H = g.n, len(pins)
    nv = n * P + H * P
    c = np.zeros(nv)
    for h in range(H):
        c[n * P + h * P: n * P + (h + 1) * P] = cost[h]
    n_rows = n + sum(len(p) for p in pins) * P + P + 1
    A = lil_matrix((n_rows, nv))
    lo, hi = [], []
    r = 0
    for v in range(n):
        for p in range(P):
            A[r, v * P + p] = 1
        lo.append(1); hi.append(1); r += 1
    for h, pin in enumerate(pins):
        for v in pin:
            for p in range(P):
                A[r, v * P + p] = 1
                A[r, n * P + h * P + p] = -1
                lo.append(-np.inf); hi.append(0); r += 1
    W = float(g.w.sum())
    for p in range(P):
        for v in range(n):
            A[r, v * P + p] = g.w[v]
        lo.append(-np.inf); hi.append((1 + eps) * W / P); r += 1
    A[r, 0] = 1; lo.append(1); hi.append(1); r += 1          # symmetry: site 0 → part 0
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        if eps < 1.0:
            return place_ilp(g, P, eps=2 * eps, time_limit=time_limit, **kw)
        raise RuntimeError(f"ILP infeasible: {res.message}")
    x = res.x[: n * P].reshape(n, P)
    return np.argmax(x, axis=1)


# ------------------------------------------------------------------ registry

PLACERS = {
    "rows": place_rows,
    "cols": place_cols,
    "tiles": place_tiles,
    "random": place_random,
    "kahypar_cut": lambda g, P, **kw: place_kahypar(g, P, objective="cut", chi_aware=False, **kw),
    "kahypar_km1": lambda g, P, **kw: place_kahypar(g, P, objective="km1", chi_aware=False, **kw),
    "kahypar_km1_chi": lambda g, P, **kw: place_kahypar(g, P, objective="km1", chi_aware=True, **kw),
    "ilp": place_ilp,
}

# which placers use live χ (and hence benefit from re-partitioning)
CHI_AWARE = {"rows", "cols", "tiles", "kahypar_km1_chi", "ilp"}


def run_placer(name, g, P, **kw):
    """Return (pi, seconds)."""
    t0 = time.perf_counter()
    pi = PLACERS[name](g, P, **kw)
    return np.asarray(pi, dtype=int), time.perf_counter() - t0
