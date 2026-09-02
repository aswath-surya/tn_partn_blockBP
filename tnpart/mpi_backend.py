"""Distributed simple-update PEPS evolution with mpi4py (the executor of the
ledger's message plan; brief §1.4, E1, E2, Part VII).

Ownership: rank p holds Γ_v for every site with π(v) = p (numpy arrays; the
array module ``xp`` is swappable for cupy) and a replicated copy of every bond
gauge Λ_e plus the global χ table. Every circuit round and every gauge-sweep
colour class is a set of pairwise-disjoint bonds, so all bonds of a round are
processed with one rendezvous each, in a fixed global order — deadlock-free.

Gate on bond e = (a, b), executor = owner of a (fixed, so no rank needs the
other side's shape; receive buffers are sized with MPI Probe). No global χ
table is kept: a rank only needs χ_e for bonds incident to its own sites, and
it always holds a current copy of those. Local bond: everything on one rank.
Cut bond:
    other : Q_b R_b = QR(Γ_b·Λ_others)            send R_b  (r_b, χ_e, d)
    exec  : θ = R_a Λ_e R_b G ; SVD ; truncate       send R_b', Λ_e'
    other : Γ_b' = Q_b R_b' / Λ_others
Gauge update on e (simple-update / BP fixed point, brief E1): both sides QR
across e, ``a`` receives R_b, computes SVD(R_a Λ_e R_bᵀ) = U s Vh and sends
(Vh, s) back; Γ_a' = Q_a U, Γ_b' = Q_b Vhᵀ, Λ_e' = s/‖s‖.

Numerics are placement-independent by construction: the same QR/SVD on the
same operands in the same order, whichever rank runs them — ``mpirun -n 1``
and ``-n P`` must agree bit-for-bit.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .circuits import Circuit, grid_bonds, grid_sites, square_rounds

SMUDGE = 1e-12


class Timers:
    KEYS = ("gate_comp", "gate_comm", "gauge_comp", "gauge_comm", "sync", "total")

    def __init__(self):
        self.t = {k: 0.0 for k in self.KEYS}
        self.n_msgs = 0
        self.bytes = 0

    def add(self, key, dt):
        self.t[key] += dt


class DistributedPEPS:
    def __init__(self, circuit: Circuit, D, pi, comm, *, cutoff=1e-10, gauge_iters=5, xp=np,
                 dtype=np.complex128):
        self.circ, self.D, self.cutoff, self.gauge_iters = circuit, int(D), float(cutoff), int(gauge_iters)
        self.comm, self.rank, self.P = comm, comm.Get_rank(), comm.Get_size()
        self.xp, self.dtype = xp, dtype
        self.d = 2
        Lx, Ly = circuit.Lx, circuit.Ly
        self.sites = grid_sites(Lx, Ly)
        self.bonds = grid_bonds(Lx, Ly)
        self.sidx = {s: k for k, s in enumerate(self.sites)}
        self.bidx = {b: k for k, b in enumerate(self.bonds)}
        self.pi = np.asarray(pi, dtype=int)
        n, m = len(self.sites), len(self.bonds)
        # bond axes per site, fixed order
        self.axes = [[] for _ in range(n)]
        self.ends = np.zeros((m, 2), dtype=int)
        for e, (a, b) in enumerate(self.bonds):
            va, vb = self.sidx[a], self.sidx[b]
            self.axes[va].append(e)
            self.axes[vb].append(e)
            self.ends[e] = (va, vb)
        self.chi = np.ones(m, dtype=int)                       # replicated
        self.lam = [np.ones(1, dtype=float) for _ in range(m)]  # replicated
        self.mine = [v for v in range(n) if self.pi[v] == self.rank]
        self.G = {}
        for v in self.mine:
            T = xp.zeros((1,) * len(self.axes[v]) + (self.d,), dtype=dtype)
            T[(0,) * len(self.axes[v]) + (0,)] = 1.0
            self.G[v] = T
        self.timers = Timers()
        self.disc_total = 0.0
        self.log = []

    # ------------------------------------------------------------ helpers
    def rest(self, v, e):
        return int(np.prod([self.chi[f] for f in self.axes[v] if f != e], dtype=np.int64)) if len(self.axes[v]) > 1 else 1

    def _scale(self, T, v, e_skip, invert=False):
        """Multiply (or divide) every bond axis of site v except e_skip by its Λ."""
        xp = self.xp
        for k, f in enumerate(self.axes[v]):
            if f == e_skip:
                continue
            g = xp.asarray(self.lam[f], dtype=self.dtype)
            if invert:
                g = 1.0 / (g + SMUDGE)
            shape = [1] * T.ndim
            shape[k] = -1
            T = T * g.reshape(shape)
        return T

    def _split(self, T, k, with_phys):
        """QR of site tensor across bond axis k (and the physical axis if
        with_phys). Returns Q (rest, r), R (r, χ, d) or (r, χ), rest_shape."""
        xp = self.xp
        nd = T.ndim
        keep = [i for i in range(nd) if i != k and (with_phys is False or i != nd - 1)]
        perm = keep + ([k, nd - 1] if with_phys else [k])
        Tp = xp.transpose(T, perm)
        rest_shape = Tp.shape[: len(keep)]
        tail = Tp.shape[len(keep):]
        M = Tp.reshape(int(np.prod(rest_shape)) if rest_shape else 1, int(np.prod(tail)))
        Q, R = xp.linalg.qr(M)
        return Q, R.reshape((R.shape[0],) + tail), rest_shape, perm

    def _merge(self, Q, Rn, rest_shape, perm):
        xp = self.xp
        Tn = (Q @ Rn.reshape(Rn.shape[0], -1)).reshape(tuple(rest_shape) + Rn.shape[1:])
        inv = np.argsort(perm)
        return xp.transpose(Tn, inv)

    # ------------------------------------------------------------ gates
    def _theta_svd(self, Ra, Rb, lam_e, G4, chi_max):
        """θ = R_a Λ R_b G → SVD → truncated (R_a', R_b', s, disc)."""
        xp = self.xp
        r_a, chi, d = Ra.shape
        r_b = Rb.shape[0]
        lam = xp.asarray(lam_e, dtype=self.dtype)
        th = xp.einsum("axp,x,bxq,PQpq->aPbQ", Ra, lam, Rb, xp.asarray(G4, dtype=self.dtype))
        M = th.reshape(r_a * d, r_b * d)
        U, s, Vh = xp.linalg.svd(M, full_matrices=False)
        s_np = np.asarray(s if xp is np else s.get())
        tot = float(np.sum(s_np ** 2))
        keep = int(np.sum(s_np > self.cutoff * s_np[0])) if tot > 0 else 1
        keep = max(1, min(chi_max, keep))
        disc = float(np.sum(s_np[keep:] ** 2) / tot) if tot > 0 else 0.0
        Ra_n = xp.transpose(U[:, :keep].reshape(r_a, d, keep), (0, 2, 1))          # (r_a, χ', d)
        Rb_n = xp.transpose(Vh[:keep].reshape(keep, r_b, d), (1, 0, 2))            # (r_b, χ', d)
        s_k = s_np[:keep] / np.linalg.norm(s_np[:keep])
        return Ra_n, Rb_n, s_k, disc

    def _recv_sized(self, source, tag):
        """Probe the incoming message, allocate, receive. Returns the array."""
        from mpi4py import MPI
        st = MPI.Status()
        self.comm.Probe(source=source, tag=tag, status=st)
        n = st.Get_count(MPI.COMPLEX16 if self.dtype == np.complex128 else MPI.COMPLEX8)
        buf = self.xp.empty(n, dtype=self.dtype)
        self.comm.Recv(buf, source=source, tag=tag)
        return buf

    def apply_2q_round(self, gates):
        """All gates of one round (disjoint bonds) as a 3-phase exchange:
        (1) b-sides QR and Isend R_b, a-sides QR; (2) a-sides receive, θ-SVD,
        update, Isend (R_b', Λ_e'); local bonds; (3) b-sides receive, finalise."""
        from mpi4py import MPI
        xp, comm, d = self.xp, self.comm, self.d
        T = self.timers
        work_a, work_b, work_loc, reqs, keep_alive = [], [], [], [], []
        for g in gates:
            e = self.bidx[g.bond]
            va, vb = self.ends[e]
            pa, pb = self.pi[va], self.pi[vb]
            if self.rank not in (pa, pb):
                continue
            G4 = np.asarray(g.G).reshape(2, 2, 2, 2)
            if pa == pb:
                work_loc.append((e, va, vb, G4))
            elif self.rank == pa:
                work_a.append((e, va, pb, G4))
            else:
                work_b.append((e, vb, pa))
        # ---- phase 1
        t0 = time.perf_counter()
        pend_b = {}
        for e, vb, pa in work_b:
            kb = self.axes[vb].index(e)
            Qb, Rb, sb, permb = self._split(self._scale(self.G[vb], vb, e), kb, True)
            Rb = xp.ascontiguousarray(Rb)
            pend_b[e] = (Qb, sb, permb, Rb.shape[0])
            T.add("gate_comp", time.perf_counter() - t0); t0 = time.perf_counter()
            reqs.append(comm.Isend(Rb, dest=pa, tag=e)); keep_alive.append(Rb)
            T.n_msgs += 1; T.bytes += Rb.nbytes
            T.add("gate_comm", time.perf_counter() - t0); t0 = time.perf_counter()
        pend_a = {}
        for e, va, pb, G4 in work_a:
            ka = self.axes[va].index(e)
            pend_a[e] = self._split(self._scale(self.G[va], va, e), ka, True)
        T.add("gate_comp", time.perf_counter() - t0)
        # ---- phase 2
        for e, va, pb, G4 in work_a:
            chi = int(self.chi[e])
            t0 = time.perf_counter()
            Rb = self._recv_sized(pb, e).reshape(-1, chi, d)
            T.add("gate_comm", time.perf_counter() - t0); t0 = time.perf_counter()
            T.n_msgs += 1; T.bytes += Rb.nbytes
            Qa, Ra, sa, perma = pend_a.pop(e)
            chi_max = min(self.D, Ra.shape[0] * d, Rb.shape[0] * d)
            Ra_n, Rb_n, s_k, disc = self._theta_svd(Ra, Rb, self.lam[e], G4, chi_max)
            self.G[va] = self._scale(self._merge(Qa, Ra_n, sa, perma), va, e, invert=True)
            self.lam[e], self.chi[e] = s_k, len(s_k)
            self.disc_total += disc
            out = xp.concatenate([Rb_n.reshape(-1), xp.asarray(s_k, dtype=self.dtype)])
            T.add("gate_comp", time.perf_counter() - t0); t0 = time.perf_counter()
            reqs.append(comm.Isend(out, dest=pb, tag=e)); keep_alive.append(out)
            T.n_msgs += 1; T.bytes += out.nbytes
            T.add("gate_comm", time.perf_counter() - t0)
        t0 = time.perf_counter()
        for e, va, vb, G4 in work_loc:
            chi = int(self.chi[e])
            ka, kb = self.axes[va].index(e), self.axes[vb].index(e)
            r_a, r_b = min(self.rest(va, e), d * chi), min(self.rest(vb, e), d * chi)
            chi_max = min(self.D, r_a * d, r_b * d)
            Qa, Ra, sa, perma = self._split(self._scale(self.G[va], va, e), ka, True)
            Qb, Rb, sb, permb = self._split(self._scale(self.G[vb], vb, e), kb, True)
            Ra_n, Rb_n, s_k, disc = self._theta_svd(Ra, Rb, self.lam[e], G4, chi_max)
            self.G[va] = self._scale(self._merge(Qa, Ra_n, sa, perma), va, e, invert=True)
            self.G[vb] = self._scale(self._merge(Qb, Rb_n, sb, permb), vb, e, invert=True)
            self.lam[e], self.chi[e] = s_k, len(s_k)
            self.disc_total += disc
        T.add("gate_comp", time.perf_counter() - t0)
        # ---- phase 3
        for e, vb, pa in work_b:
            t0 = time.perf_counter()
            inbuf = self._recv_sized(pa, e)
            T.add("gate_comm", time.perf_counter() - t0); t0 = time.perf_counter()
            T.n_msgs += 1; T.bytes += inbuf.nbytes
            Qb, sb, permb, r_b = pend_b.pop(e)
            keep = inbuf.size // (r_b * d + 1)
            Rb_n = inbuf[: r_b * keep * d].reshape(r_b, keep, d)
            s_k = np.asarray(inbuf[r_b * keep * d:] if xp is np else inbuf[r_b * keep * d:].get()).real
            self.G[vb] = self._scale(self._merge(Qb, Rb_n, sb, permb), vb, e, invert=True)
            self.lam[e], self.chi[e] = s_k, keep
            T.add("gate_comp", time.perf_counter() - t0)
        t0 = time.perf_counter()
        MPI.Request.Waitall(reqs)
        T.add("gate_comm", time.perf_counter() - t0)

    def apply_1q_round(self, gates):
        xp = self.xp
        t0 = time.perf_counter()
        for g in gates:
            v = self.sidx[g.where[0]]
            if self.pi[v] != self.rank:
                continue
            T = self.G[v]
            self.G[v] = xp.tensordot(T, xp.asarray(g.G, dtype=self.dtype), axes=([T.ndim - 1], [1]))
        self.timers.add("gate_comp", time.perf_counter() - t0)

    def sync_chi_final(self):
        """One Allgather at the end so every rank (and the report) sees all χ_e."""
        t0 = time.perf_counter()
        mine = np.where(self.pi[self.ends[:, 0]] == self.rank, self.chi, -1)
        allc = np.array(self.comm.allgather(mine))
        self.chi = allc.max(axis=0)
        self.timers.add("sync", time.perf_counter() - t0)

    # ------------------------------------------------------------ gauge sweep
    def gauge_sweep(self, n_iter=None):
        """Simple-update gauge fixing (= BP fixed point): colour class by colour
        class; within a class all bonds are disjoint, so each class is one
        3-phase exchange (b-sides Isend R_b; a-sides SVD and Isend (Vh, s);
        b-sides finalise) — numerically identical to the sequential sweep."""
        from mpi4py import MPI
        n_iter = self.gauge_iters if n_iter is None else n_iter
        xp, comm, T = self.xp, self.comm, self.timers
        classes = [[self.bidx[b] for b in cls] for cls in square_rounds(self.circ.Lx, self.circ.Ly)]
        for _ in range(n_iter):
            for cls in classes:
                work_a, work_b, work_loc, reqs, keep_alive = [], [], [], [], []
                for e in cls:
                    va, vb = self.ends[e]
                    pa, pb = self.pi[va], self.pi[vb]
                    if self.rank not in (pa, pb):
                        continue
                    if pa == pb:
                        work_loc.append((e, va, vb))
                    elif self.rank == pa:
                        work_a.append((e, va, pb))
                    else:
                        work_b.append((e, vb, pa))
                t0 = time.perf_counter()
                pend_b = {}
                for e, vb, pa in work_b:
                    kb = self.axes[vb].index(e)
                    Qb, Rb, sb, permb = self._split(self._scale(self.G[vb], vb, e), kb, False)
                    Rb = xp.ascontiguousarray(Rb)
                    pend_b[e] = (Qb, sb, permb, Rb.shape[0])
                    T.add("gauge_comp", time.perf_counter() - t0); t0 = time.perf_counter()
                    reqs.append(comm.Isend(Rb, dest=pa, tag=10000 + e)); keep_alive.append(Rb)
                    T.n_msgs += 1; T.bytes += Rb.nbytes
                    T.add("gauge_comm", time.perf_counter() - t0); t0 = time.perf_counter()
                pend_a = {}
                for e, va, pb in work_a:
                    ka = self.axes[va].index(e)
                    pend_a[e] = self._split(self._scale(self.G[va], va, e), ka, False)
                T.add("gauge_comp", time.perf_counter() - t0)
                for e, va, pb in work_a:
                    t0 = time.perf_counter()
                    Rb = self._recv_sized(pb, 10000 + e).reshape(-1, int(self.chi[e]))
                    T.add("gauge_comm", time.perf_counter() - t0); t0 = time.perf_counter()
                    T.n_msgs += 1; T.bytes += Rb.nbytes
                    Qa, Ra, sa, perma = pend_a.pop(e)
                    U, s_k, Vh = self._bond_svd(Ra, Rb, self.lam[e])
                    self.G[va] = self._scale(self._merge(Qa, U, sa, perma), va, e, invert=True)
                    self.lam[e], self.chi[e] = s_k, len(s_k)
                    out = xp.concatenate([xp.ascontiguousarray(Vh).reshape(-1), xp.asarray(s_k, dtype=self.dtype)])
                    T.add("gauge_comp", time.perf_counter() - t0); t0 = time.perf_counter()
                    reqs.append(comm.Isend(out, dest=pb, tag=10000 + e)); keep_alive.append(out)
                    T.n_msgs += 1; T.bytes += out.nbytes
                    T.add("gauge_comm", time.perf_counter() - t0)
                t0 = time.perf_counter()
                for e, va, vb in work_loc:
                    ka, kb = self.axes[va].index(e), self.axes[vb].index(e)
                    Qa, Ra, sa, perma = self._split(self._scale(self.G[va], va, e), ka, False)
                    Qb, Rb, sb, permb = self._split(self._scale(self.G[vb], vb, e), kb, False)
                    U, s_k, Vh = self._bond_svd(Ra, Rb, self.lam[e])
                    self.G[va] = self._scale(self._merge(Qa, U, sa, perma), va, e, invert=True)
                    self.G[vb] = self._scale(self._merge(Qb, xp.ascontiguousarray(Vh.T), sb, permb), vb, e, invert=True)
                    self.lam[e], self.chi[e] = s_k, len(s_k)
                T.add("gauge_comp", time.perf_counter() - t0)
                for e, vb, pa in work_b:
                    t0 = time.perf_counter()
                    inbuf = self._recv_sized(pa, 10000 + e)
                    T.add("gauge_comm", time.perf_counter() - t0); t0 = time.perf_counter()
                    T.n_msgs += 1; T.bytes += inbuf.nbytes
                    Qb, sb, permb, r_b = pend_b.pop(e)
                    keep = inbuf.size // (r_b + 1)
                    Vh = inbuf[: keep * r_b].reshape(keep, r_b)
                    s_k = np.asarray(inbuf[keep * r_b:] if xp is np else inbuf[keep * r_b:].get()).real
                    self.G[vb] = self._scale(self._merge(Qb, xp.ascontiguousarray(Vh.T), sb, permb), vb, e, invert=True)
                    self.lam[e], self.chi[e] = s_k, keep
                    T.add("gauge_comp", time.perf_counter() - t0)
                t0 = time.perf_counter()
                MPI.Request.Waitall(reqs)
                T.add("gauge_comm", time.perf_counter() - t0)

    def _bond_svd(self, Ra, Rb, lam_e):
        xp = self.xp
        M = (Ra * xp.asarray(lam_e, dtype=self.dtype)[None, :]) @ Rb.T
        U, s, Vh = xp.linalg.svd(M, full_matrices=False)
        s_np = np.asarray(s if xp is np else s.get())
        keep = max(1, int(np.sum(s_np > 0)))
        s_k = s_np[:keep] / np.linalg.norm(s_np[:keep])
        return U[:, :keep], s_k, Vh[:keep]

    # ------------------------------------------------------------ driver
    def run(self, progress=False):
        t_start = time.perf_counter()
        for li, layer in enumerate(self.circ.layers):
            for ri, rnd in enumerate(layer):
                if rnd.kind == "1q":
                    self.apply_1q_round(rnd.gates)
                else:
                    self.apply_2q_round(rnd.gates)
            if self.gauge_iters > 0:
                self.gauge_sweep()
            if progress and self.rank == 0:
                print(f"  layer {li}: local max χ={self.chi.max()} disc={self.disc_total:.3e} "
                      f"t={time.perf_counter() - t_start:.2f}s", flush=True)
        self.timers.add("total", time.perf_counter() - t_start)
        self.sync_chi_final()

    # ------------------------------------------------------------ validation
    def gather_state(self):
        """Rank 0 receives every Γ and the current Λ_e from an owner of each
        bond (a rank's Λ copy is only kept current for bonds it touches) —
        validation only. Returns (G_all, lam) on rank 0."""
        local = {v: (np.asarray(self.G[v]) if self.xp is np else self.G[v].get()) for v in self.mine}
        my_lam = {e: self.lam[e] for e in range(len(self.bonds)) if self.pi[self.ends[e][0]] == self.rank}
        parts = self.comm.gather((local, my_lam), root=0)
        if self.rank != 0:
            return None, None
        G_all = {}
        for loc, lam in parts:
            G_all.update(loc)
            for e, l in lam.items():
                self.lam[e] = l
        return G_all, self.lam

    def dense_state(self, G_all):
        """Exact dense vector (n ≤ 20) via quimb: Λ absorbed symmetrically."""
        import quimb.tensor as qtn
        tensors = []
        for v, T in G_all.items():
            data = T.copy()
            for k, e in enumerate(self.axes[v]):
                shape = [1] * data.ndim
                shape[k] = -1
                data = data * np.sqrt(self.lam[e]).reshape(shape)
            inds = [f"b{e}" for e in self.axes[v]] + [f"k{v}"]
            tensors.append(qtn.Tensor(data, inds=inds, tags={f"s{v}"}))
        tn = qtn.TensorNetwork(tensors)
        psi = tn.contract(output_inds=[f"k{v}" for v in range(len(self.sites))]).data.ravel()
        return psi / np.linalg.norm(psi)

    def report(self):
        """Timers reduced to rank 0: per-key max/min over ranks."""
        allt = self.comm.gather(dict(self.timers.t, n_msgs=self.timers.n_msgs, bytes=self.timers.bytes), root=0)
        disc = self.comm.reduce(self.disc_total, root=0)
        if self.rank != 0:
            return None
        out = {"P": self.P, "D": self.D, "cutoff": self.cutoff, "gauge_iters": self.gauge_iters,
               "disc_total": disc, "max_chi": int(self.chi.max()), "chi_hist": np.bincount(self.chi).tolist(),
               "per_rank": allt}
        for k in Timers.KEYS + ("n_msgs", "bytes"):
            vals = [t[k] for t in allt]
            out[f"{k}_max"], out[f"{k}_min"], out[f"{k}_mean"] = max(vals), min(vals), float(np.mean(vals))
        return out
