"""PEPS simple-update circuit evolution with a per-round trace (brief E1, E2, E6).

The physics is independent of placement, so we evolve once and record, for
every round, the bond dimensions χ_e and site sizes w_v *before* the round, the
bonds acted on, and the exact discarded weight of every truncation. The trace
is then replayed against any placement (``vsim.replay``).

Truncation: ``gate_simple_`` is called *untruncated* (one SVD, full spectrum
returned via ``info``), then the bond is sliced to D by hand. In the Vidal/SU
gauge this is exactly the truncated SVD, and it yields the discarded weight
ε = Σ_{k>D} s_k² / Σ s_k² that quimb does not otherwise expose.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .circuits import Circuit, grid_bonds, grid_sites
from .tngraph import TNGraph


@dataclass
class RoundSnapshot:
    layer: int
    ridx: int
    kind: str
    bonds_acted: list          # bond ids (2q rounds)
    chi: np.ndarray            # (m,) before the round
    w: np.ndarray              # (n,) before the round
    disc: float                # Σ discarded weight over the round's gates
    n_trunc: int
    t_wall: float              # measured local seconds for the round


@dataclass
class Trace:
    Lx: int
    Ly: int
    d: int
    family: str
    meta: dict
    D: int
    cutoff: float
    gauge_iters: int
    sites: list
    bonds: list
    rounds: list
    chi_final: np.ndarray
    w_final: np.ndarray
    n_layers: int
    disc_total: float
    logfid_proxy: float
    chi_b: int | None = None
    z_bmps: np.ndarray | None = None
    z_exact: np.ndarray | None = None
    mae_z: float | None = None
    maxerr_z: float | None = None
    fidelity: float | None = None      # |<ψ_exact|ψ_PEPS>|², n ≤ 20 only
    obs: str = "Z"                     # which single-site Pauli z_bmps / z_exact hold
    amp0_bmps: complex | None = None
    amp0_exact: complex | None = None
    timings: dict = field(default_factory=dict)

    # ------------------------------------------------------------ derived

    @property
    def n(self):
        return len(self.sites)

    @property
    def m(self):
        return len(self.bonds)

    def traffic_suffix(self):
        """(R+1, m) remaining 2q-gate counts per bond from round r onwards."""
        R = len(self.rounds)
        suf = np.zeros((R + 1, self.m), dtype=np.int64)
        for r in range(R - 1, -1, -1):
            suf[r] = suf[r + 1]
            for e in self.rounds[r].bonds_acted:
                suf[r, e] += 1
        return suf

    def graph_at(self, r, suffix=None):
        suffix = self.traffic_suffix() if suffix is None else suffix
        s = self.rounds[r]
        return TNGraph(self.Lx, self.Ly, self.d, self.sites, self.bonds, s.w, s.chi, suffix[r],
                       layers_remaining=self.n_layers - s.layer)

    def graph_final(self, full_traffic=True, suffix=None):
        suffix = self.traffic_suffix() if suffix is None else suffix
        traffic = suffix[0] if full_traffic else np.zeros(self.m, dtype=np.int64)
        return TNGraph(self.Lx, self.Ly, self.d, self.sites, self.bonds, self.w_final, self.chi_final,
                       traffic, layers_remaining=self.n_layers if full_traffic else 0)

    def chi_stats_by_layer(self):
        """Rows (layer, min, median, mean, max, frac_at_D) — the §2.4 heterogeneity data."""
        out = []
        seen = set()
        for s in self.rounds:
            if s.layer in seen:
                continue
            seen.add(s.layer)
            c = s.chi
            out.append((s.layer, int(c.min()), float(np.median(c)), float(c.mean()), int(c.max()),
                        float(np.mean(c >= self.D))))
        c = self.chi_final
        out.append((self.n_layers, int(c.min()), float(np.median(c)), float(c.mean()), int(c.max()),
                    float(np.mean(c >= self.D))))
        return out

    @property
    def t_evolve(self):
        return sum(s.t_wall for s in self.rounds)

    @property
    def max_bond(self):
        return int(self.chi_final.max())


# ------------------------------------------------------------------ physics

def _product_peps(Lx, Ly):
    import quimb.tensor as qtn

    def fill(shape):
        a = np.zeros(shape, dtype=complex)
        a[(0,) * len(shape)] = 1.0
        return a

    return qtn.PEPS.from_fill_fn(fill, Lx=Lx, Ly=Ly, bond_dim=1, phys_dim=2)


def apply_2q_traced(peps, gauges, G, where, D, cutoff):
    """SU gate in the tracked gauge, then slice the bond to D. Returns the
    discarded weight ε of this gate (0 if no truncation)."""
    info = {}
    peps.gate_simple_(G, where, gauges, max_bond=None, cutoff=0.0, info=info, renorm=False)
    ((_, ix), s), = info.items()
    s = np.abs(np.asarray(s))
    if s.size == 0:
        return 0.0
    s2 = s * s
    tot = float(s2.sum())
    keep = int(np.sum(s > cutoff * s[0]))
    keep = max(1, min(D, keep))
    eps = float(s2[keep:].sum() / tot) if tot > 0 else 0.0
    if keep < s.size:
        for site in where:
            t = peps[site]
            ax = t.inds.index(ix)
            t.modify(data=np.take(t.data, np.arange(keep), axis=ax))
        s = s[:keep]
    gauges[ix] = s / np.linalg.norm(s)
    return eps


def bmps_local_z(peps, gauges, sites, chi_b, contract_optimize="auto", obs="Z"):
    """All-site <P_i> (P = obs Pauli) by boundary-MPS (plaquette environments).
    Returns (values, norm). Greedy path search by default: the plaquette
    contractions are small and cotengra's ``auto-hq`` search dominates otherwise."""
    import quimb as qu
    Z = qu.pauli(obs)
    with peps.gauge_simple_temp(gauges):
        ez = peps.compute_local_expectation({s: Z for s in sites}, max_bond=chi_b,
                                            normalized=True, return_all=True,
                                            contract_optimize=contract_optimize)
    z = np.array([float((ez[s][0] / ez[s][1]).real) for s in sites])
    norm = float(abs(complex(ez[sites[0]][1])))
    return z, norm


def bmps_norm(peps, gauges, chi_b):
    """<ψ|ψ> by boundary-MPS. ``layer_tags`` keeps bra and ket separate so no
    (D²)⁴ doubled site tensor is ever formed (memory ~D⁴ not D⁸)."""
    with peps.gauge_simple_temp(gauges):
        nrm = peps.make_norm().contract_boundary(max_bond=chi_b, layer_tags=("KET", "BRA"))
    return float(abs(complex(nrm)))


def bmps_amplitude0(peps, gauges, sites, chi_b, norm=None):
    """<0…0|ψ>/‖ψ‖ — single-layer boundary contraction (cheap)."""
    if norm is None:
        norm = bmps_norm(peps, gauges, chi_b)
    with peps.gauge_simple_temp(gauges):
        amp = peps.isel({peps.site_ind(*s): 0 for s in sites}).contract_boundary(max_bond=chi_b)
    return complex(amp) / np.sqrt(norm)


def exact_statevector(circuit: Circuit):
    """Dense reference (n ≤ ~22). Qubit q = i·Ly + j; quimb gate conventions."""
    n, Ly = circuit.n_qubits, circuit.Ly
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    q = lambda c: c[0] * Ly + c[1]
    for _, _, rnd in circuit.rounds():
        for g in rnd.gates:
            if g.arity == 1:
                qa = q(g.where[0])
                psi = np.moveaxis(np.tensordot(g.G, psi, axes=([1], [qa])), 0, qa)
            else:
                qa, qb = q(g.where[0]), q(g.where[1])
                G4 = g.G.reshape(2, 2, 2, 2)
                psi = np.moveaxis(np.tensordot(G4, psi, axes=([2, 3], [qa, qb])), [0, 1], [qa, qb])
    return psi


_PAULI = {"X": np.array([[0, 1], [1, 0]], dtype=complex), "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def exact_local_z(psi, obs="Z"):
    """<P_q> for every qubit from a dense state of shape (2,)*n."""
    n = psi.ndim
    if obs == "Z":
        p = np.abs(psi) ** 2
        return np.array([p.sum(axis=tuple(k for k in range(n) if k != q))[0] * 2 - 1 for q in range(n)])
    P = _PAULI[obs]
    out = np.empty(n)
    for q in range(n):
        phi = np.moveaxis(np.tensordot(P, psi, axes=([1], [q])), 0, q)
        out[q] = np.vdot(psi, phi).real
    return out


def evolve_and_trace(circuit: Circuit, D, cutoff=1e-10, gauge_iters=5, chi_b=None, exact="auto",
                     observables=True, amplitude=True, progress=False, bmps_max_D=8, exact_max_D=8,
                     obs="Z"):
    """Evolve and trace. Boundary-MPS observables are skipped when D > bmps_max_D
    (doubled-layer legs D² make the site tensors ~D⁸); the exact dense overlap is
    computed whenever n ≤ 20 and is the preferred fidelity metric."""
    Lx, Ly = circuit.Lx, circuit.Ly
    sites, bonds = grid_sites(Lx, Ly), grid_bonds(Lx, Ly)
    bidx = {b: k for k, b in enumerate(bonds)}
    peps = _product_peps(Lx, Ly)
    gauges = {}
    snap_chi = lambda: np.array([peps.bond_size(a, b) for a, b in bonds], dtype=np.int64)
    snap_w = lambda: np.array([peps[s].size for s in sites], dtype=np.int64)

    rounds, disc_total, logfid = [], 0.0, 0.0
    t_ev = time.perf_counter()
    for li, ri, rnd in circuit.rounds():
        chi0, w0 = snap_chi(), snap_w()
        t0 = time.perf_counter()
        disc, ntr, acted = 0.0, 0, []
        if rnd.kind == "1q":
            for g in rnd.gates:
                peps.gate_(g.G, g.where[0], contract=True)
        else:
            for g in rnd.gates:
                eps = apply_2q_traced(peps, gauges, g.G, g.where, D, cutoff)
                acted.append(bidx[g.bond])
                disc += eps
                ntr += eps > 0
                logfid += np.log1p(-eps) if eps < 1 else -np.inf
        if ri == len(circuit.layers[li]) - 1 and gauge_iters > 0 and gauges:
            peps.gauge_all_simple_(max_iterations=gauge_iters, tol=1e-9, gauges=gauges)
        rounds.append(RoundSnapshot(li, ri, rnd.kind, acted, chi0, w0, disc, ntr, time.perf_counter() - t0))
        disc_total += disc
        if progress:
            print(f"  layer {li} round {ri} ({rnd.kind}) max χ={peps.max_bond()} disc={disc:.2e}", flush=True)
    t_ev = time.perf_counter() - t_ev

    tr = Trace(Lx, Ly, 2, circuit.family, dict(circuit.meta), D, cutoff, gauge_iters, sites, bonds, rounds,
               snap_chi(), snap_w(), circuit.depth, disc_total, logfid, timings=dict(evolve=t_ev))

    if chi_b is None:
        chi_b = min(2 * D * D + 10, 64)
    tr.chi_b = chi_b
    tr.obs = obs
    do_bmps = D <= bmps_max_D
    norm = None
    if observables and do_bmps:
        t0 = time.perf_counter()
        tr.z_bmps, norm = bmps_local_z(peps, gauges, sites, chi_b, obs=obs)
        tr.timings["bmps_z"] = time.perf_counter() - t0
    if amplitude and do_bmps:
        t0 = time.perf_counter()
        tr.amp0_bmps = bmps_amplitude0(peps, gauges, sites, chi_b, norm=norm)
        tr.timings["bmps_amp"] = time.perf_counter() - t0
    do_exact = (exact is True) or (exact == "auto" and circuit.n_qubits <= 20 and D <= exact_max_D)
    if do_exact:
        t0 = time.perf_counter()
        psi = exact_statevector(circuit).ravel()
        tr.z_exact = exact_local_z(psi.reshape((2,) * circuit.n_qubits), obs=obs)
        tr.amp0_exact = complex(psi[0])
        with peps.gauge_simple_temp(gauges):
            v = np.asarray(peps.to_dense()).ravel()
        v = v / np.linalg.norm(v)
        tr.fidelity = float(abs(np.vdot(psi, v)) ** 2)
        tr.timings["exact"] = time.perf_counter() - t0
        if tr.z_bmps is not None:
            err = np.abs(tr.z_bmps - tr.z_exact)
            tr.mae_z, tr.maxerr_z = float(err.mean()), float(err.max())
    return tr
