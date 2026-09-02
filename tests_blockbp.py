"""BlockBP tests on small networks, before any circuit-scale run.

  ../.venv/bin/python -u tests_blockbp.py

T1  tree block graph (2x8 random TN, 2x2 blocks → path of 4 blocks): must be exact.
T2  loopy block graph (6x6 random TN): error vs χ_m, block size, schedule, damping,
    and signed vs positive tensors (the stability-relevant contrast).
T3  random NON-brickwork 1+1D circuit amplitude <0|U|0>: blocks = spacetime tiles.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import quimb as qu  # noqa: E402
import quimb.tensor as qtn  # noqa: E402

from tnpart.blockbp import BlockBP  # noqa: E402


def grid_blocks(bx, by):
    """tag 'I{i},{j}' -> block (i//bx, j//by)"""
    def block_of(t):
        for g in t.tags:
            if g.startswith("I") and "," in g:
                i, j = map(int, g[1:].split(","))
                return (i // bx, j // by)
        raise ValueError(t.tags)
    return block_of


def coord_key_2d(tn):
    """boundary-bond ordering key from the coordinates of the tensors sharing the bond"""
    pos = {}
    for tid, t in tn.tensor_map.items():
        for g in t.tags:
            if g.startswith("I") and "," in g:
                pos[tid] = tuple(map(int, g[1:].split(",")))
    def key(ind):
        return tuple(sorted(pos[tid] for tid in tn.ind_map[ind]))
    return key


def rel_err(z, z0):
    return abs(z / z0 - 1.0)


def run(tn, block_of, chi_m, **kw):
    bp = BlockBP(tn, block_of, chi_m=chi_m, optimize="auto", **kw)
    info = bp.run()
    return bp, info


# ------------------------------------------------------------------ T1
def test_tree():
    print("== T1 tree block graph: 2x8 random complex TN (D=3), 2x2 blocks -> path of 4 blocks")
    tn = qtn.TN2D_rand(2, 8, 3, dtype="complex128", seed=1)
    Z0 = complex(tn.contract(optimize="auto-hq"))
    for chi_m in (1, 3, 9):
        bp, info = run(tn, grid_blocks(2, 2), chi_m, order_key=coord_key_2d(tn), init="random", tol=1e-10)
        print(f"   χ_m={chi_m:2d}: tree={info['is_tree']} iters={info['iters']:2d} Δ={info['delta']:.1e} "
              f"rel err |Z_bp/Z-1| = {rel_err(bp.Z(), Z0):.2e}")


# ------------------------------------------------------------------ T2
def test_loopy():
    print("== T2 loopy block graph: 6x6 random TN (D=2); exact vs BlockBP")
    for kind in ("signed", "positive"):
        tn = qtn.TN2D_rand(6, 6, 2, dtype="complex128" if kind == "signed" else "float64", seed=2)
        if kind == "positive":
            for t in tn:
                t.modify(data=np.abs(t.data) + 0.1)
        Z0 = complex(tn.contract(optimize="auto-hq"))
        print(f"  [{kind} tensors]  exact log|Z| = {np.log(abs(Z0)):.4f}")
        for (bx, by) in ((3, 3), (2, 2), (6, 3)):
            for chi_m in (1, 2, 4, 8):
                for sched, damp in (("parallel", 0.0), ("sequential", 0.0), ("parallel", 0.3)):
                    t0 = time.perf_counter()
                    bp, info = run(tn, grid_blocks(bx, by), chi_m, order_key=coord_key_2d(tn), schedule=sched,
                                   damping=damp, init="random", max_iter=60)
                    print(f"   blocks {bx}x{by} ({info['n_blocks']:2d} blocks, max ∂={info['max_boundary']}) χ_m={chi_m:2d} "
                          f"{sched:10s} η={damp:.1f}: iters={info['iters']:2d} Δ={info['delta']:.1e} "
                          f"{'conv' if info['converged'] else 'NOT ':4s} rel err={rel_err(bp.Z(), Z0):.2e}  ({time.perf_counter()-t0:.1f}s)")


# ------------------------------------------------------------------ T3
from tnpart.circuits1d import random_circuit, amplitude_tn, norm_tn, circuit_blocks, circuit_key  # noqa: E402


def test_circuit():
    print("== T3 random non-brickwork 1+1D circuit: amplitude <0|U|0> via BlockBP vs exact")
    n, depth = 8, 8
    circ = random_circuit(n, depth, seed=3)
    A0 = complex(circ.amplitude("0" * n))
    tn = amplitude_tn(circ)
    print(f"   n={n} depth={depth} gates={circ.num_gates}  exact |A| = {abs(A0):.4e}")
    for (bq, bt) in ((4, 3), (4, 5), (2, 3), (8, 3)):
        for chi_m in (2, 4, 8, 16):
            for sched, damp in (("parallel", 0.0), ("sequential", 0.0), ("parallel", 0.3)):
                try:
                    t0 = time.perf_counter()
                    bp, info = run(tn, circuit_blocks(bq, bt, depth), chi_m, order_key=circuit_key(tn),
                                   schedule=sched, damping=damp, init="random", max_iter=60)
                    print(f"   blocks (q{bq},t{bt}) ({info['n_blocks']:2d} blocks, tree={info['is_tree']}, max ∂={info['max_boundary']:2d}) "
                          f"χ_m={chi_m:2d} {sched:10s} η={damp:.1f}: iters={info['iters']:2d} Δ={info['delta']:.1e} "
                          f"{'conv' if info['converged'] else 'NOT ':4s} rel err={rel_err(bp.Z(), A0):.2e} ({time.perf_counter()-t0:.1f}s)")
                except Exception as e:
                    print(f"   blocks (q{bq},t{bt}) χ_m={chi_m} {sched} η={damp}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["tree", "loopy", "circuit"]
    if "tree" in which:
        test_tree()
    if "loopy" in which:
        test_loopy()
    if "circuit" in which:
        test_circuit()


# ------------------------------------------------------------------ T4
def test_norm():
    print("== T4 same circuit, double-layer norm network <ψ|ψ> (=1): BlockBP on PSD-like messages")
    n, depth = 8, 8
    circ = random_circuit(n, depth, seed=3)
    tn = norm_tn(circ)
    def block_of_norm(t):
        qs = [int(g[1:]) for g in t.tags if g.startswith("I") and g[1:].isdigit()]
        rounds = [int(g.split("_")[1]) for g in t.tags if g.startswith("ROUND_")]
        r = -1 if "PSI0" in t.tags else rounds[0]
        return (min(qs) // BQ, (r + 1) // BT)
    for (BQ, BT) in ((4, 3), (2, 3), (4, 5)):
        for chi_m in (2, 4, 8, 16):
            for sched, damp in (("parallel", 0.0), ("sequential", 0.0)):
                try:
                    t0 = time.perf_counter()
                    bp, info = run(tn, block_of_norm, chi_m, order_key=circuit_key(tn), schedule=sched, damping=damp,
                                   init="random", max_iter=60)
                    z = bp.Z()
                    print(f"   blocks (q{BQ},t{BT}) ({info['n_blocks']:2d} blocks, tree={info['is_tree']}, max ∂={info['max_boundary']:2d}) "
                          f"χ_m={chi_m:2d} {sched:10s}: iters={info['iters']:2d} Δ={info['delta']:.1e} "
                          f"{'conv' if info['converged'] else 'NOT ':4s} Z={z:.4f} |Z-1|={abs(z-1):.2e} ({time.perf_counter()-t0:.1f}s)")
                except Exception as e:
                    print(f"   blocks (q{BQ},t{BT}) χ_m={chi_m} {sched}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__" and "norm" in sys.argv[1:]:
    test_norm()
