"""Weighted TN placement graph: a snapshot of (χ_e, w_v, remaining traffic).

This is the object every placer consumes (brief §2.2, E10). Vertices are site
tensors with weight w_v = d·∏χ (elements); 2-pin *bond nets* carry the gate and
BP-message traffic; Ly-pin *row nets* / Lx-pin *column nets* carry the
boundary-MPS sweep hand-offs — the genuine hyperedges of the problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .costmodel import Machine, bmps_handoff_bytes


@dataclass
class TNGraph:
    Lx: int
    Ly: int
    d: int
    sites: list                # list[(i,j)]
    bonds: list                # list[((i,j),(k,l))]
    w: np.ndarray              # (n,) elements per site tensor
    chi: np.ndarray            # (m,) bond dimensions
    traffic: np.ndarray        # (m,) remaining 2-qubit gates per bond
    layers_remaining: int = 1
    sidx: dict = field(init=False, repr=False)
    bidx: dict = field(init=False, repr=False)
    site_bonds: list = field(init=False, repr=False)

    def __post_init__(self):
        self.sidx = {s: k for k, s in enumerate(self.sites)}
        self.bidx = {b: k for k, b in enumerate(self.bonds)}
        self.site_bonds = [[] for _ in self.sites]
        for e, (a, b) in enumerate(self.bonds):
            self.site_bonds[self.sidx[a]].append(e)
            self.site_bonds[self.sidx[b]].append(e)
        self.w = np.asarray(self.w, dtype=np.int64)
        self.chi = np.asarray(self.chi, dtype=np.int64)
        self.traffic = np.asarray(self.traffic, dtype=np.int64)

    @property
    def n(self):
        return len(self.sites)

    @property
    def m(self):
        return len(self.bonds)

    def bond_ends(self, e):
        a, b = self.bonds[e]
        return self.sidx[a], self.sidx[b]

    def rest(self, v, e):
        """Product of the bond dims of site v other than bond e."""
        return max(1, int(self.w[v] // (self.d * self.chi[e])))

    def rows(self):
        return [[self.sidx[(i, j)] for j in range(self.Ly)] for i in range(self.Lx)]

    def cols(self):
        return [[self.sidx[(i, j)] for i in range(self.Lx)] for j in range(self.Ly)]

    def line_D(self, pins):
        """Max χ touching a set of sites (the D entering a bMPS hand-off)."""
        es = {e for v in pins for e in self.site_bonds[v]}
        return int(max((self.chi[e] for e in es), default=1))

    # ------------------------------------------------------------ net costs

    def bond_costs(self, machine: Machine, gauge_iters: int):
        """Byte-equivalent (inter tier) cost of cutting each bond over the
        remaining horizon: gate R-factor shipping + BP/gauge messages."""
        c = np.zeros(self.m)
        bps = machine.bytes_per_scalar
        for e in range(self.m):
            a, b = self.bond_ends(e)
            chi = int(self.chi[e])
            r = min(self.rest(a, e), self.rest(b, e), self.d * chi)
            ship = 2 * bps * r * self.d * chi                     # R over, R' back
            nmsg_gate = 2 * int(self.traffic[e])
            gauge = 2 * bps * chi * chi * gauge_iters * self.layers_remaining
            nmsg_gauge = 2 * gauge_iters * self.layers_remaining
            c[e] = machine.byte_equiv(ship * self.traffic[e] + gauge, nmsg_gate + nmsg_gauge)
        return c

    def nets(self, machine: Machine, gauge_iters=5, chi_b=32, n_bmps_passes=2,
             include_rows=True, chi_aware=True):
        """Return (pins_list, cost_array). With ``chi_aware=False`` bond costs are
        the remaining gate counts and row nets get unit cost — the circuit-graph-
        only (χ-blind) model of the prior art (brief §2.4)."""
        pins = [list(self.bond_ends(e)) for e in range(self.m)]
        if chi_aware:
            cost = list(self.bond_costs(machine, gauge_iters))
        else:
            cost = [float(t) + 1e-3 for t in self.traffic]
        if include_rows:
            for line in self.rows() + self.cols():
                pins.append(list(line))
                if chi_aware:
                    D = self.line_D(line)
                    cost.append(machine.byte_equiv(
                        n_bmps_passes * 2 * bmps_handoff_bytes(chi_b, D, machine), n_bmps_passes * 2))
                else:
                    cost.append(1.0)
        return pins, np.asarray(cost, dtype=float)

    @staticmethod
    def net_lambda(pi, pins):
        return len({int(pi[v]) for v in pins})
