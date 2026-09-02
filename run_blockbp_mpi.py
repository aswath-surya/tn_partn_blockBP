"""Distributed BlockBP convergence experiment on a random 1+1D circuit (mpi4py).

One rank owns one or more spacetime tiles (bq qubits × bt layers) of the amplitude
network <0|U|0> (or the norm network <ψ|ψ>); MPS messages cross block edges; remote
edges go through MPI. A library of convergence strategies is run one after another,
each from a fresh initialisation, under a time budget; Δ and the Bethe log Z are
recorded per sweep.

  laptop:     mpirun -n 4 ../.venv/bin/python run_blockbp_mpi.py --n 8 --depth 8 --bq 4 --bt 4 --preset basic --exact
  Perlmutter: srun -n 40 python run_blockbp_mpi.py --n 80 --depth 40 --bq 10 --bt 8 --preset full --tag p10

Strategies (dicts): schedule 'parallel' (flooded) | 'colored' (2-colour Gauss–Seidel on the
block grid); damping η fixed, or quenched per directed edge η_e ~ U(lo, hi) (relay-BP style
memory strengths); relay legs = re-draw the quenched η after `leg` sweeps without
convergence, keep the best; init 'random' | 'uniform'; chi_m override.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import quimb.tensor as qtn  # noqa: E402
from mpi4py import MPI  # noqa: E402

from tnpart.circuits1d import amplitude_tn, block_grid, circuit_blocks, circuit_key, norm_tn, random_circuit  # noqa: E402

COMM = MPI.COMM_WORLD
RANK, P = COMM.Get_rank(), COMM.Get_size()


# ------------------------------------------------------------------ strategies
def presets(name, chi_m):
    S = []
    base = dict(chi_m=chi_m, schedule="parallel", damping=0.0, quench=None, legs=1, leg=0, init="random")
    add = lambda **k: S.append({**base, **k})
    if name == "fast":                                   # 3 strategies: the Pareto sweeps
        add(name="parallel"); add(name="colored", schedule="colored"); add(name="parallel_eta0.3", damping=0.3)
        return S
    if name == "agree":                                  # 5 strategies: schedule-agreement test at scale
        add(name="parallel"); add(name="colored", schedule="colored"); add(name="parallel_eta0.3", damping=0.3)
        add(name="quench_0.1-0.7", quench=(0.1, 0.7)); add(name="relay3x10_quench", quench=(0.1, 0.9), legs=3, leg=10)
        return S
    add(name="parallel")
    add(name="colored", schedule="colored")
    add(name="parallel_eta0.3", damping=0.3)
    add(name="parallel_eta0.6", damping=0.6)
    add(name="colored_eta0.3", schedule="colored", damping=0.3)
    add(name="quench_0.1-0.7", quench=(0.1, 0.7))
    add(name="quench_0.3-0.9_colored", schedule="colored", quench=(0.3, 0.9))
    add(name="relay3x20_quench", quench=(0.1, 0.9), legs=3, leg=20)
    add(name="parallel_uniform", init="uniform")
    add(name="quench_uniform", quench=(0.1, 0.7), init="uniform")
    if name == "full":
        add(name="parallel_eta0.15", damping=0.15)
        add(name="parallel_eta0.8", damping=0.8)
        add(name="colored_eta0.6", schedule="colored", damping=0.6)
        add(name="quench_0.5-0.95", quench=(0.5, 0.95))
        add(name="quench_0.0-0.5_colored", schedule="colored", quench=(0.0, 0.5))
        add(name="relay5x15_quench_colored", schedule="colored", quench=(0.2, 0.9), legs=5, leg=15)
        add(name="relay3x30_eta0.5", damping=0.5, legs=3, leg=30)
        add(name="chi_half_parallel", chi_m=max(2, chi_m // 2))
        add(name="chi_double_quench", chi_m=2 * chi_m, quench=(0.1, 0.7))
        add(name="colored_uniform_eta0.3", schedule="colored", damping=0.3, init="uniform")
    return S


# ------------------------------------------------------------------ distributed BlockBP
class DistBlockBP:
    def __init__(self, tn, block_of, order_key, owner, dense_limit=2 ** 22):
        self.tn = tn
        self.tid_block = {tid: block_of(t) for tid, t in tn.tensor_map.items()}
        self.blocks = sorted(set(self.tid_block.values()))
        self.owner = owner                                   # block -> rank
        self.mine = [A for A in self.blocks if owner[A] == RANK]
        self.T = {A: qtn.TensorNetwork([tn.tensor_map[tid].copy() for tid, b in self.tid_block.items() if b == A])
                  for A in self.mine}
        edges = {}
        for ind, tids in tn.ind_map.items():
            a, b = (self.tid_block[t] for t in tids)
            if a != b:
                edges.setdefault(frozenset((a, b)), []).append(ind)
        self.order = {e: sorted(inds, key=order_key) for e, inds in edges.items()}
        self.dims = {e: [tn.ind_size(i) for i in inds] for e, inds in self.order.items()}
        for e, d in self.dims.items():
            if int(np.prod(d)) > dense_limit:
                raise RuntimeError(f"boundary {tuple(e)} has {len(d)} bonds — too large for the dense route")
        self.nbrs = {A: set() for A in self.blocks}
        for e in self.order:
            a, b = tuple(e)
            self.nbrs[a].add(b); self.nbrs[b].add(a)
        self.directed = sorted({(a, b) for e in self.order for a, b in (tuple(e), tuple(e)[::-1])})
        self.tag = {ab: i for i, ab in enumerate(self.directed)}
        self.out_edges = [(A, B) for A in self.mine for B in sorted(self.nbrs[A])]
        self.in_edges = [(B, A) for A in self.mine for B in sorted(self.nbrs[A])]
        self.n_edges = len(self.order)
        self.m = {}

    # ---- message representation: dense complex vector over the ordered boundary bonds
    def _e(self, A, B):
        return frozenset((A, B))

    def _normalize(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _truncate(self, v, e, chi_m):
        """SVD-compress the dense boundary vector to an MPS of bond χ_m, back to dense."""
        dims = self.dims[e]
        if len(dims) < 3 or chi_m >= int(np.prod(dims)) ** 0.5 + 1:
            return self._normalize(v)
        mps = qtn.MatrixProductState.from_dense(v, dims=dims, max_bond=chi_m, cutoff=1e-12)
        return self._normalize(np.asarray(mps.to_dense()).reshape(-1))

    def init_messages(self, kind, rng):
        for (A, B) in self.in_edges + self.out_edges:
            e = self._e(A, B)
            n = int(np.prod(self.dims[e]))
            v = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) if kind == "random" else np.ones(n, dtype=complex)
            self.m[(A, B)] = self._normalize(v)

    def _incoming_tn(self, A, exclude=None):
        tnA = self.T[A].copy()
        for B in self.nbrs[A]:
            if B == exclude:
                continue
            e = self._e(A, B)
            tnA &= qtn.Tensor(self.m[(B, A)].reshape(self.dims[e]), inds=self.order[e])
        return tnA

    def compute(self, A, B, chi_m):
        e = self._e(A, B)
        inds = self.order[e]
        t = self._incoming_tn(A, exclude=B).contract(output_inds=inds, optimize="greedy")
        v = np.asarray(t.transpose(*inds).data).reshape(-1)
        return self._truncate(v, e, chi_m)

    def exchange(self, outgoing, color=None):
        """outgoing: dict (A,B)->vector for my blocks; deliver to owner(B); receive the
        messages sent this phase (all senders, or only blocks of the given colour)."""
        reqs = []
        for (A, B), v in outgoing.items():
            if self.owner[B] == RANK:
                self.m[(A, B)] = v
            else:
                reqs.append(COMM.isend(v, dest=self.owner[B], tag=self.tag[(A, B)]))
        for (B, A) in self.in_edges:
            if color is not None and (B[0] + B[1]) % 2 != color:
                continue
            if self.owner[B] != RANK:
                self.m[(B, A)] = COMM.recv(source=self.owner[B], tag=self.tag[(B, A)])
        MPI.Request.waitall(reqs)
        for (A, B), v in outgoing.items():
            self.m[(A, B)] = v

    def sweep(self, strat, eta_edge):
        """One sweep; returns (Δ², count) local contributions."""
        chi_m = strat["chi_m"]
        d2, cnt = 0.0, 0
        colors = [None] if strat["schedule"] == "parallel" else [0, 1]
        for c in colors:
            outgoing = {}
            for (A, B) in self.out_edges:
                if c is not None and (A[0] + A[1]) % 2 != c:
                    continue
                v = self.compute(A, B, chi_m)
                eta = eta_edge.get((A, B), 0.0)
                if eta > 0:
                    v = self._truncate((1 - eta) * v + eta * self.m[(A, B)], self._e(A, B), chi_m)
                ov = abs(np.vdot(self.m[(A, B)], v)) ** 2
                d2 += 2 * max(0.0, 1 - ov); cnt += 1
                outgoing[(A, B)] = v
            self.exchange(outgoing, color=c)
        return d2, cnt

    def log_Z_local(self):
        lz = 0j
        for A in self.mine:
            lz += np.log(complex(self._incoming_tn(A).contract(optimize="greedy")))
        for (A, B) in self.out_edges:
            if A < B:                                        # each edge once, by its lower block
                lz -= np.log(complex(np.dot(self.m[(A, B)], self.m[(B, A)])))
        return lz


# ------------------------------------------------------------------ driver
def run_strategy(bp, strat, sid, args, t_budget):
    rng = np.random.default_rng(1000 * args.seed + sid)
    bp.init_messages(strat["init"], rng)
    def draw_eta(leg_seed):
        if strat["quench"]:
            r = np.random.default_rng(10_000 * sid + leg_seed)
            return {ab: float(r.uniform(*strat["quench"])) for ab in bp.out_edges}
        return {ab: strat["damping"] for ab in bp.out_edges}
    eta = draw_eta(0)
    hist, lz_hist, times = [], [], []
    t0 = time.perf_counter()
    best = None
    leg_iter, leg_no = 0, 0
    converged = False
    for it in range(1, args.max_sweeps + 1):
        d2, cnt = bp.sweep(strat, eta)
        tot = COMM.allreduce(np.array([d2, cnt]), op=MPI.SUM)
        delta = float(np.sqrt(tot[0] / max(tot[1], 1)))
        lz = COMM.allreduce(bp.log_Z_local(), op=MPI.SUM)
        hist.append(delta); lz_hist.append([lz.real, lz.imag]); times.append(time.perf_counter() - t0)
        if best is None or delta < best[0]:
            best = (delta, lz, it)
        if delta < args.tol and (len(hist) < 2 or hist[-2] < args.tol):
            converged = True
            break
        leg_iter += 1
        if strat["legs"] > 1 and leg_iter >= strat["leg"] and leg_no + 1 < strat["legs"]:
            leg_no += 1; leg_iter = 0
            eta = draw_eta(leg_no)                           # relay: new quenched damping, keep messages
        if time.perf_counter() - t0 > t_budget:
            break
    return dict(strategy=strat, sweeps=len(hist), delta=hist, logZ=lz_hist, t=times, converged=converged,
                trusted=hist[-1] < args.tol_trust, best_delta=best[0], best_logZ=[best[1].real, best[1].imag],
                best_sweep=best[2], legs_used=leg_no + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--bq", type=int, default=4); ap.add_argument("--bt", type=int, default=4)
    ap.add_argument("--p-gate", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chi-m", type=int, default=8); ap.add_argument("--network", default="amplitude", choices=["amplitude", "norm"])
    ap.add_argument("--preset", default="basic", choices=["basic", "full", "fast", "agree"])
    ap.add_argument("--max-sweeps", type=int, default=60); ap.add_argument("--budget", type=float, default=90.0, help="seconds per strategy")
    ap.add_argument("--tol", type=float, default=1e-5); ap.add_argument("--tol-trust", type=float, default=1e-3)
    ap.add_argument("--exact", action="store_true", help="exact amplitude reference (n ≤ 24)")
    ap.add_argument("--tag", default=None); ap.add_argument("--out", default="results")
    args = ap.parse_args()

    circ = random_circuit(args.n, args.depth, seed=args.seed, p_gate=args.p_gate)
    tn = amplitude_tn(circ) if args.network == "amplitude" else norm_tn(circ)
    depth = args.depth
    block_of = circuit_blocks(args.bq, args.bt, depth)
    if args.network == "norm":
        base = block_of
        def block_of(t):                                     # bra tensors sit with their ket partners
            qs = [int(g[1:]) for g in t.tags if g.startswith("I") and g[1:].isdigit()]
            rs = [int(g.split("_")[1]) for g in t.tags if g.startswith("ROUND_")]
            r = -1 if "PSI0" in t.tags else rs[0]
            return (min(qs) // args.bq, (r + 1) // args.bt)
    blocks = sorted({block_of(t) for t in tn})
    nq = max(b[0] for b in blocks) + 1
    # owner map: consecutive q-blocks of one t-row share a node (t-cuts cross nodes, q-cuts stay local)
    flat = sorted(blocks, key=lambda b: (b[1], b[0]))
    per_rank = -(-len(flat) // P)
    owner = {b: min(i // per_rank, P - 1) for i, b in enumerate(flat)}
    bp = DistBlockBP(tn, block_of, circuit_key(tn), owner)
    if RANK == 0:
        print(f"run_blockbp_mpi: {args.network} n={args.n} depth={depth} tiles {args.bq}x{args.bt} → {len(blocks)} blocks "
              f"({nq} × {len(blocks)//nq}) on {P} ranks, {bp.n_edges} block edges, max boundary "
              f"{max(len(v) for v in bp.order.values())} bonds, χ_m={args.chi_m}, preset={args.preset}", flush=True)
    exact = None
    if args.exact and RANK == 0:
        exact = complex(circ.amplitude("0" * args.n)) if args.network == "amplitude" else 1.0 + 0j
        print(f"   exact log Z = {np.log(exact):.6f}", flush=True)
    exact = COMM.bcast(exact, root=0)

    results = []
    for sid, strat in enumerate(presets(args.preset, args.chi_m)):
        COMM.Barrier()
        t0 = time.perf_counter()
        res = run_strategy(bp, strat, sid, args, args.budget)
        res["wall"] = time.perf_counter() - t0
        if RANK == 0:
            lz = complex(*res["best_logZ"])
            err = "" if exact is None else f" |Z/Z_exact−1|={abs(np.exp(lz) / exact - 1):.2e}"
            print(f"   [{sid:2d}] {strat['name']:26s} sweeps={res['sweeps']:3d} Δ_final={res['delta'][-1]:.2e} "
                  f"best Δ={res['best_delta']:.2e}@{res['best_sweep']:2d} {'CONV' if res['converged'] else 'not '} "
                  f"logZ={lz.real:+.4f}{lz.imag:+.4f}i{err} ({res['wall']:.0f}s)", flush=True)
        results.append(res)
    if RANK == 0:
        os.makedirs(args.out, exist_ok=True)
        tag = args.tag or f"{args.network}_n{args.n}_d{depth}_q{args.bq}t{args.bt}_P{P}"
        path = os.path.join(args.out, f"blockbp_{tag}.json")
        json.dump(dict(args=vars(args), P=P, n_blocks=len(blocks), n_edges=bp.n_edges,
                       max_boundary=max(len(v) for v in bp.order.values()),
                       exact_logZ=None if exact is None else [np.log(exact).real, np.log(exact).imag],
                       results=results), open(path, "w"), indent=1)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
