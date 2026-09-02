"""Perlmutter cost model (brief Part I) and the per-round communication ledger.

Every number in ``Machine`` is the brief's *estimate*; Phase-0 microbenchmarks
(§7.1) replace them. Conventions:

* FLOP counts returned by the primitives are textbook ``2·m·n·k`` counts on the
  complex tensors (one complex multiply-add = one "op"); the ledger converts
  them to *real* FLOPs with ``Machine.cplx`` (= 4) and to time with
  ``p_gemm`` (real FLOP/s of an achievable ``zgemm``). SVD-type work is timed
  with the extra inefficiency factor ``c_svd`` (§1.2) but its FLOPs are
  recorded un-inflated, so ``flops`` in the records is true arithmetic.
* Messages are charged to both endpoints (each NIC moves the bytes).
* Round time is reported three ways (no overlap assumed anywhere):
  ``t_round = max_p (comp_p + comm_p)`` (primary), ``t_ub = max_p comp_p +
  max_p comm_p`` (the compute/communication split used for stacked bars) and
  ``t_lb = max_p max(comp_p, comm_p)`` (perfect overlap).
* Resident memory per rank is tracked per round and checked against
  ``hbm_bytes / mem_headroom`` (SVD workspace inflates peaks 2–3×, §7.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

TIERS = ("intra", "inter")


@dataclass(frozen=True)
class Machine:
    name: str = "perlmutter-a100-est"
    bw_inter: float = 25e9        # B/s per GPU NIC, Slingshot-11                        (§1.1)
    nic_share: float = 1.0        # fraction of that usable by one rank (Cray-MPICH: 1 NIC/dir)
    lat_inter: float = 5e-6       # s, point-to-point GPU-aware MPI (collectives ≈ 40 µs)  (§1.1)
    bw_intra: float = 100e9       # B/s NVLink per pair — placeholder, measure            (§1.1)
    lat_intra: float = 1.5e-6     # s, must be < lat_inter
    p_gemm: float = 1.5e13        # real FLOP/s, achievable zgemm                         (§1.3)
    cplx: float = 4.0             # real FLOPs per complex multiply-add op
    c_svd: float = 20.0           # SVD/QR time inefficiency vs GEMM, c ≈ 10–50           (§1.2)
    bytes_per_scalar: int = 16    # complex128
    hbm_bytes: float = 40e9
    mem_headroom: float = 3.0     # workspace inflation applied to resident bytes         (§7.6)

    def bw(self, tier):
        return self.bw_inter * self.nic_share if tier == "inter" else self.bw_intra

    def lat(self, tier):
        return self.lat_inter if tier == "inter" else self.lat_intra

    def byte_equiv(self, nbytes, nmsg=1, tier="inter"):
        """Bytes plus latency expressed in bytes at the tier's bandwidth (used
        by the partitioner's net costs, tngraph.bond_costs)."""
        return nbytes + nmsg * self.lat(tier) * self.bw(tier)

    def t_compute(self, ops, kind="gemm"):
        """Seconds for ``ops`` complex ops of the given kind."""
        t = ops * self.cplx / self.p_gemm
        return t * self.c_svd if kind == "svd" else t

    @property
    def mem_limit(self):
        return self.hbm_bytes / self.mem_headroom

    @classmethod
    def from_json(cls, path):
        """Build from a Phase-0 calibration (perlmutter/plot_phase0.py output);
        unknown keys are ignored, missing ones keep the estimates above."""
        import dataclasses
        import json
        doc = json.load(open(path))
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in doc.items() if k in names})


@dataclass(frozen=True)
class Topology:
    """Rank → node map. Assumes SLURM's default block distribution
    (ranks 0..k-1 on node 0, ...); use ``--distribution=block`` on Perlmutter."""
    P: int
    ranks_per_node: int = 4

    def node(self, p):
        return p // self.ranks_per_node

    def tier(self, p, q):
        if p == q:
            return "local"
        return "intra" if self.node(p) == self.node(q) else "inter"

    @property
    def n_nodes(self):
        return -(-self.P // self.ranks_per_node)


# ------------------------------------------------------------------ primitives

class GateCost(NamedTuple):
    gemm: float        # QR + θ contraction + reabsorption (complex ops)
    svd: float         # full SVD of θ (complex ops; timed with c_svd)
    r_a: int
    r_b: int
    chi_full: int      # rank of θ before truncation
    chi_new: int       # bond dimension actually kept


def gate_cost(d, chi_e, rest_a, rest_b, machine: Machine, chi_new=None):
    """Reduce-split 2-site gate on bond e (brief E2). ``rest_x`` = product of the
    other bond dims of site x; ``chi_new`` = bond dimension after truncation
    (from the trace; defaults to the untruncated rank)."""
    n = d * chi_e
    r_a, r_b = min(rest_a, n), min(rest_b, n)
    qr = 2 * rest_a * n * min(rest_a, n) + 2 * rest_b * n * min(rest_b, n)
    theta = 2 * r_a * r_b * d * d * chi_e + 2 * r_a * r_b * d ** 4
    M, N = r_a * d, d * r_b
    chi_full = min(M, N)
    chi_new = chi_full if chi_new is None else min(int(chi_new), chi_full)
    svd = 4 * M * N * chi_full
    reab = 2 * rest_a * r_a * d * chi_new + 2 * rest_b * r_b * d * chi_new
    return GateCost(qr + theta + reab, svd, r_a, r_b, chi_full, chi_new)


class GaugeCost(NamedTuple):
    qr_a: float
    qr_b: float
    svd: float


def gauge_bond_cost(w_a, w_b, chi_e, machine: Machine):
    """One simple-gauge / BP message update on bond e (brief E1): reduced QR of
    each site reshaped to (w/χ_e)×χ_e — cost 2·(w/χ_e)·χ_e² = 2wχ_e — then a
    χ_e×χ_e SVD."""
    return GaugeCost(2 * w_a * chi_e, 2 * w_b * chi_e, 4 * chi_e ** 3)


class BmpsCost(NamedTuple):
    contract: float
    svd: float


def bmps_site_cost(chi_b, D, machine: Machine, d=2, lazy=True):
    """Absorb one site of the norm network into a boundary MPS of bond χ_b
    (brief E6). ``lazy``: bra and ket contracted separately (E1's 𝒯_v = T⊗T*
    kept lazy), 2dχ_b²D⁷; eager doubled-layer tensor, 2χ_b²D⁸. Compression:
    sequential SVD of the (χ_b D²)×(χ_b D²) merged tensor."""
    contract = 2 * d * chi_b ** 2 * D ** 7 if lazy else 2 * chi_b ** 2 * D ** 8
    return BmpsCost(contract, 4 * (chi_b * D * D) ** 3)


def bmps_handoff_bytes(chi_b, D, machine: Machine):
    """One boundary-MPS *site* tensor (χ_b, D², χ_b) crossing a rank boundary.
    A sweep across a boundary of L sites is charged L times by the caller."""
    return machine.bytes_per_scalar * chi_b ** 2 * D ** 2


def blockbp_message_bytes(chi_m, D, L_boundary, machine: Machine):
    """BlockBP message = an MPS along the whole block boundary (brief §2.6, E3):
    16·χ_m²·D²·L_∂ bytes per block-adjacency per iteration."""
    return machine.bytes_per_scalar * chi_m ** 2 * D ** 2 * L_boundary


# ------------------------------------------------------------------ ledger

@dataclass
class RoundRecord:
    layer: int
    ridx: int
    phase: str
    t_wall: float          # max_p (comp_p + comm_p)
    t_comp: float          # max_p comp_p
    t_comm: float          # max_p comm_p
    t_lb: float            # max_p max(comp_p, comm_p)
    flops: float           # true real FLOPs, all ranks
    flops_svd: float       # of which SVD/QR-type
    flops_max: float       # true real FLOPs on the busiest rank
    bytes_intra: float
    bytes_inter: float
    msgs_intra: int
    msgs_inter: int
    resident_max: float    # peak resident bytes on any rank this round
    oom: bool


class Ledger:
    """Bulk-synchronous per-round accounting of work, messages and memory per rank."""

    def __init__(self, P, topology: Topology, machine: Machine):
        self.P, self.topo, self.m = P, topology, machine
        self.records: list[RoundRecord] = []
        self._open = False

    def new_round(self, layer, ridx, phase):
        if self._open:
            raise RuntimeError("previous round not closed")
        self._layer, self._ridx, self._phase = layer, ridx, phase
        self._ops = {"gemm": np.zeros(self.P), "svd": np.zeros(self.P)}
        self._bytes = {t: np.zeros(self.P) for t in TIERS}
        self._msgs = {t: np.zeros(self.P, dtype=np.int64) for t in TIERS}
        self._resident = np.zeros(self.P)
        self._open = True

    def add_flops(self, p, ops, kind="gemm"):
        self._ops[kind][p] += ops

    def add_msg(self, src, dst, nbytes):
        tier = self.topo.tier(src, dst)
        if tier == "local":
            return
        for p in (src, dst):
            self._bytes[tier][p] += nbytes
            self._msgs[tier][p] += 1

    def add_net(self, src, dst_parts, nbytes):
        """A tensor on ``src`` needed by the parts in ``dst_parts``: charged
        once per distinct destination part ≠ src, i.e. (λ−1) transfers (§2.2)."""
        for q in {int(q) for q in dst_parts} - {int(src)}:
            self.add_msg(src, q, nbytes)

    def set_resident(self, p, nbytes):
        """Resident bytes on rank p during this round (max over calls)."""
        self._resident[p] = max(self._resident[p], nbytes)

    def add_resident(self, p, nbytes):
        self._resident[p] += nbytes

    def close_round(self, serial=False):
        """``serial=True`` models a sequential chain (a boundary-MPS line: each
        site absorption depends on the previous one): work and hand-offs are
        summed over ranks, not maxed."""
        if not self._open:
            raise RuntimeError("no open round")
        comp = self.m.t_compute(self._ops["gemm"]) + self.m.t_compute(self._ops["svd"], "svd")
        comm = np.zeros(self.P)
        for t in TIERS:
            comm += self._bytes[t] / self.m.bw(t) + self._msgs[t] * self.m.lat(t)
        if serial:
            t_comp, t_comm = float(comp.sum()), float(comm.sum() / 2)   # msgs counted at both ends
            t_wall, t_lb = t_comp + t_comm, max(t_comp, t_comm)
        else:
            t_comp, t_comm = float(comp.max()), float(comm.max())
            t_wall, t_lb = float((comp + comm).max()), float(np.maximum(comp, comm).max())
        real = (self._ops["gemm"] + self._ops["svd"]) * self.m.cplx
        rec = RoundRecord(
            self._layer, self._ridx, self._phase, t_wall, t_comp, t_comm, t_lb,
            float(real.sum()), float(self._ops["svd"].sum() * self.m.cplx), float(real.max()),
            float(self._bytes["intra"].sum() / 2), float(self._bytes["inter"].sum() / 2),
            int(self._msgs["intra"].sum() // 2), int(self._msgs["inter"].sum() // 2),
            float(self._resident.max()), bool(self._resident.max() > self.m.mem_limit),
        )
        self.records.append(rec)
        self._open = False
        return rec

    def summary(self):
        R = self.records
        out = dict(
            t_wall=sum(r.t_wall for r in R),
            t_comp=sum(r.t_comp for r in R),
            t_comm=sum(r.t_comm for r in R),
            t_wall_lb=sum(r.t_lb for r in R),
            flops=sum(r.flops for r in R),
            flops_svd=sum(r.flops_svd for r in R),
            bytes_intra=sum(r.bytes_intra for r in R),
            bytes_inter=sum(r.bytes_inter for r in R),
            msgs_intra=sum(r.msgs_intra for r in R),
            msgs_inter=sum(r.msgs_inter for r in R),
            peak_resident=max((r.resident_max for r in R), default=0.0),
            oom_rounds=sum(r.oom for r in R),
        )
        out["t_wall_ub"] = out["t_comp"] + out["t_comm"]
        out["comm_frac"] = out["t_comm"] / out["t_wall_ub"] if out["t_wall_ub"] > 0 else 0.0
        tot, mx = sum(r.flops for r in R), sum(r.flops_max for r in R)
        out["comp_efficiency"] = (tot / self.P) / mx if mx > 0 else 1.0
        out["gflops_eff"] = out["flops"] / out["t_comp"] / 1e9 / self.P if out["t_comp"] > 0 else 0.0
        for ph in sorted({r.phase for r in R}):
            sub = [r for r in R if r.phase == ph]
            out[f"t_wall[{ph}]"] = sum(r.t_wall for r in sub)
            out[f"t_comm[{ph}]"] = sum(r.t_comm for r in sub)
            out[f"t_comp[{ph}]"] = sum(r.t_comp for r in sub)
            out[f"bytes_inter[{ph}]"] = sum(r.bytes_inter for r in sub)
        return out
