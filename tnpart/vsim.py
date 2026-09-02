"""Virtual distributed replay of a trace under a placement (brief §1.4, Part VII).

Bulk-synchronous model: every circuit round is one ledger round; the gauge
sweep at the end of each layer is ``gauge_iters`` rounds; the final
boundary-MPS observable pass is one *serial* round per swept row/column
(the compress sweep is a sequential chain — placement cannot parallelise it,
only avoid its hand-offs). Placement may be static (decided once from the
circuit graph at χ=1 — the prior-art regime), oracle-static (from the final
χ), or dynamic: at layer boundaries the remaining trace is replayed under the
current and the candidate placement and the candidate is accepted when its
modeled remaining wall time plus migration time is lower (brief §2.7(1)).
"""
from __future__ import annotations

import numpy as np

from .costmodel import (Ledger, Machine, Topology, bmps_handoff_bytes, bmps_site_cost,
                        gate_cost, gauge_bond_cost)
from .partition import evaluate, run_placer
from .peps_sim import Trace


class _Ctx:
    """Everything the accounting needs that does not change during a replay."""

    def __init__(self, trace, P, topo, machine, gauge_iters, chi_b, n_bmps_passes, obs_factor, suffix,
                 obs_mode="bmps", bp_iters=20):
        self.trace, self.P, self.topo, self.machine = trace, P, topo, machine
        self.gauge_iters, self.chi_b = gauge_iters, chi_b
        self.n_bmps_passes, self.obs_factor, self.suffix = n_bmps_passes, obs_factor, suffix
        self.obs_mode, self.bp_iters = obs_mode, bp_iters
        self.d, self.bps = trace.d, machine.bytes_per_scalar
        sidx = {s: k for k, s in enumerate(trace.sites)}
        self.ends = np.array([(sidx[a], sidx[b]) for a, b in trace.bonds], dtype=int)
        self.R = len(trace.rounds)
        # first round index of every layer, plus R
        self.layer_starts = [r for r in range(self.R) if trace.rounds[r].ridx == 0] + [self.R]
        self.gF = trace.graph_final(full_traffic=False, suffix=suffix)


def _account_rounds(ctx: _Ctx, pi, r0, r1, ledger, stats=None):
    """Ledger rounds r0..r1-1 (gate rounds + gauge sweeps at layer ends)."""
    tr, d, bps, m, ends = ctx.trace, ctx.d, ctx.bps, ctx.trace.m, ctx.ends
    for r in range(r0, r1):
        snap = tr.rounds[r]
        chi_next = tr.rounds[r + 1].chi if r + 1 < ctx.R else tr.chi_final   # χ after this round
        ledger.new_round(snap.layer, snap.ridx, "gate" if snap.kind == "2q" else "1q")
        resident = np.bincount(pi, weights=snap.w * bps, minlength=ctx.P)   # tensors at rest
        if snap.kind == "1q":
            for v in range(tr.n):
                ledger.add_flops(pi[v], 2 * d * snap.w[v])
        else:
            for e in snap.bonds_acted:
                a, b = ends[e]
                chi = int(snap.chi[e])
                ra, rb = max(1, snap.w[a] // (d * chi)), max(1, snap.w[b] // (d * chi))
                gc = gate_cost(d, chi, ra, rb, ctx.machine, chi_new=int(chi_next[e]))
                pa, pb = pi[a], pi[b]
                ex = pa if (pa == pb or ra >= rb) else pb          # execute beside the larger Q
                if pa != pb:                                        # ship the other side's R, get R' back
                    other, r_o = (pb, gc.r_b) if ex == pa else (pa, gc.r_a)
                    ledger.add_msg(other, ex, bps * r_o * d * chi)
                    ledger.add_msg(ex, other, bps * r_o * d * gc.chi_new)
                ledger.add_flops(ex, gc.gemm)
                ledger.add_flops(ex, gc.svd, "svd")
                resident[ex] += bps * (gc.r_a * d) * (d * gc.r_b)           # θ workspace
        for p in range(ctx.P):
            ledger.set_resident(p, resident[p])
        ledger.close_round()
        if stats is not None:
            part_bytes = np.bincount(pi, weights=snap.w * bps, minlength=ctx.P)
            stats["peak"] = max(stats["peak"], float(part_bytes.max()))
            stats["imb"].append(float(part_bytes.max() / (part_bytes.sum() / ctx.P) - 1.0))
            stats["cut"].append(int(np.sum(pi[ends[:, 0]] != pi[ends[:, 1]])))
        last = (r + 1 == ctx.R) or (tr.rounds[r + 1].layer != snap.layer)
        if last and ctx.gauge_iters > 0:
            chi_a = tr.rounds[r + 1].chi if r + 1 < ctx.R else tr.chi_final
            w_a = tr.rounds[r + 1].w if r + 1 < ctx.R else tr.w_final
            resident = np.bincount(pi, weights=w_a * bps, minlength=ctx.P)
            for _ in range(ctx.gauge_iters):
                ledger.new_round(snap.layer, snap.ridx, "gauge")
                for e in range(m):
                    a, b = ends[e]
                    chi = int(chi_a[e])
                    gcst = gauge_bond_cost(w_a[a], w_a[b], chi, ctx.machine)
                    ledger.add_flops(pi[a], gcst.qr_a)
                    ledger.add_flops(pi[b], gcst.qr_b)
                    ledger.add_flops(pi[a], gcst.svd, "svd")
                    if pi[a] != pi[b]:
                        ledger.add_msg(pi[a], pi[b], bps * chi * chi)
                        ledger.add_msg(pi[b], pi[a], bps * chi * chi)
                for p in range(ctx.P):
                    ledger.set_resident(p, resident[p])
                ledger.close_round()


def _bmps_phase(ctx: _Ctx, pi, ledger):
    """Boundary-MPS contraction: rows swept from both ends, then columns,
    repeated ``obs_factor`` times (plaquette environments). One *serial* ledger
    round per line; hand-offs where consecutive sites / lines change rank."""
    g, chi_b, machine, bps = ctx.gF, ctx.chi_b, ctx.machine, ctx.bps
    tensors = np.bincount(pi, weights=ctx.trace.w_final * bps, minlength=ctx.P)
    for _ in range(ctx.obs_factor):
        for lines in (g.rows(), g.cols()):
            for pss in range(ctx.n_bmps_passes):
                order = lines if pss % 2 == 0 else lines[::-1]
                prev = None
                for line in order:
                    ledger.new_round(ctx.trace.n_layers, pss, "bmps")
                    D = g.line_D(line)
                    hb = bmps_handoff_bytes(chi_b, D, machine)
                    bc = bmps_site_cost(chi_b, D, machine, d=ctx.d)
                    for v in line:
                        ledger.add_flops(pi[v], bc.contract)
                        ledger.add_flops(pi[v], bc.svd, "svd")
                    for p in range(ctx.P):                         # tensors + the boundary MPS held
                        ledger.set_resident(p, tensors[p] + hb * sum(1 for v in line if pi[v] == p))
                    for u, v in zip(line[:-1], line[1:]):
                        if pi[u] != pi[v]:                     # canonize + compress sweeps
                            ledger.add_msg(pi[u], pi[v], hb)
                            ledger.add_msg(pi[v], pi[u], hb)
                    if prev is not None:
                        for u, v in zip(prev, line):
                            if pi[u] != pi[v]:                 # MPS tensor follows the sweep
                                ledger.add_msg(pi[u], pi[v], hb)
                    prev = line
                    ledger.close_round(serial=True)


def _bp_phase(ctx: _Ctx, pi, ledger):
    """BP observables (brief §3.2, E1): ``bp_iters`` message rounds on the final
    network — each is a gauge-style round (χ² message per cut bond, reduced-QR
    per bond) — then one local contraction per site. Cheap compute, message-
    heavy: the regime where the placement's cut actually shows in the time."""
    tr, d, bps, ends = ctx.trace, ctx.d, ctx.bps, ctx.ends
    chi, w = tr.chi_final, tr.w_final
    resident = np.bincount(pi, weights=w * bps, minlength=ctx.P)
    for _ in range(ctx.bp_iters):
        ledger.new_round(tr.n_layers, 0, "bp")
        for e in range(tr.m):
            a, b = ends[e]
            c = int(chi[e])
            gcst = gauge_bond_cost(w[a], w[b], c, ctx.machine)
            ledger.add_flops(pi[a], gcst.qr_a)
            ledger.add_flops(pi[b], gcst.qr_b)
            ledger.add_flops(pi[a], gcst.svd, "svd")
            if pi[a] != pi[b]:
                ledger.add_msg(pi[a], pi[b], bps * c * c)
                ledger.add_msg(pi[b], pi[a], bps * c * c)
        for p in range(ctx.P):
            ledger.set_resident(p, resident[p])
        ledger.close_round()
    ledger.new_round(tr.n_layers, 1, "bp")
    for v in range(tr.n):                                  # local RDM: contract site with its messages
        ledger.add_flops(pi[v], 2 * w[v] * int(max(chi[ctx.gF.site_bonds[v]], default=1)) * d)
    ledger.close_round()


def _observables(ctx: _Ctx, pi, ledger):
    if ctx.obs_mode == "bp":
        _bp_phase(ctx, pi, ledger)
    else:
        _bmps_phase(ctx, pi, ledger)


def _remaining_time(ctx: _Ctx, pi, r0):
    led = Ledger(ctx.P, ctx.topo, ctx.machine)
    _account_rounds(ctx, pi, r0, ctx.R, led)
    _observables(ctx, pi, led)
    return led.summary()["t_wall"]


def _migration(ctx: _Ctx, pi_old, pi_new, w, ledger=None):
    """Migration cost (seconds) of moving every tensor whose rank changes; if a
    ledger is given the moves are recorded as a 'migrate' round."""
    moved = np.nonzero(pi_new != pi_old)[0]
    led = ledger if ledger is not None else Ledger(ctx.P, ctx.topo, ctx.machine)
    led.new_round(-1, -1, "migrate")
    for v in moved:
        led.add_msg(pi_old[v], pi_new[v], ctx.bps * w[v])
    rec = led.close_round()
    return rec.t_comm, float(sum(ctx.bps * w[v] for v in moved if ctx.topo.tier(pi_old[v], pi_new[v]) != "local"))


def replay(trace: Trace, placer="kahypar_km1_chi", P=4, *, ranks_per_node=4, machine=None,
           gauge_iters=None, chi_b=None, n_bmps_passes=2, obs_factor=3, include_rows=True,
           dynamic=None, oracle=False, seed=0, placer_kw=None, obs_mode="bmps", bp_iters=20):
    """obs_mode: 'bmps' (boundary-MPS plaquette environments, serial, compute-heavy)
    or 'bp' (belief-propagation messages on the final network, message-heavy)."""
    machine = machine or Machine()
    topo = Topology(P, ranks_per_node)
    gauge_iters = trace.gauge_iters if gauge_iters is None else gauge_iters
    chi_b = (trace.chi_b or 32) if chi_b is None else chi_b
    ck = dict(machine=machine, gauge_iters=gauge_iters, chi_b=chi_b, n_bmps_passes=n_bmps_passes,
              include_rows=include_rows, **(placer_kw or {}))
    suffix = trace.traffic_suffix()
    ctx = _Ctx(trace, P, topo, machine, gauge_iters, chi_b, n_bmps_passes, obs_factor, suffix,
               obs_mode=obs_mode, bp_iters=bp_iters)

    # ---- initial placement: from the χ=1 circuit graph (static) or the final χ (oracle)
    g0 = trace.graph_final(full_traffic=True, suffix=suffix) if oracle else trace.graph_at(0, suffix)
    pi, t_search = run_placer(placer, g0, P, seed=seed, **ck)
    ev0 = evaluate(g0, pi, P=P, **ck)

    ledger = Ledger(P, topo, machine)
    stats = dict(peak=0.0, imb=[], cut=[])
    n_repart, n_tried, mig_bytes, t_mig = 0, 0, 0.0, 0.0
    every = (dynamic or {}).get("every", 1)
    min_gain = (dynamic or {}).get("min_gain", 0.0)

    for li in range(len(ctx.layer_starts) - 1):
        r0, r1 = ctx.layer_starts[li], ctx.layer_starts[li + 1]
        layer = trace.rounds[r0].layer
        if dynamic and layer > 0 and layer % every == 0:
            g = trace.graph_at(r0, suffix)
            pi_new, dt = run_placer(placer, g, P, seed=seed + layer, **ck)
            t_search += dt
            n_tried += 1
            if np.any(pi_new != pi):
                t_cur = _remaining_time(ctx, pi, r0)
                t_new = _remaining_time(ctx, pi_new, r0)
                t_move, b_move = _migration(ctx, pi, pi_new, g.w)
                if t_new + t_move < (1.0 - min_gain) * t_cur:
                    _migration(ctx, pi, pi_new, g.w, ledger=ledger)
                    pi, n_repart, mig_bytes, t_mig = pi_new, n_repart + 1, mig_bytes + b_move, t_mig + t_move
        _account_rounds(ctx, pi, r0, r1, ledger, stats)

    _observables(ctx, pi, ledger)
    evF = evaluate(trace.graph_final(True, suffix), pi, P=P, **ck)
    meta = trace.meta
    workload = f"{trace.family} {trace.Lx}x{trace.Ly}"
    if trace.family == "brickwork":
        workload += f" depth{meta.get('depth')}"
    elif trace.family == "tfim":
        workload += f" n{meta.get('n_steps')} dt{meta.get('dt')}" + (f" dis{meta['disorder']}" if meta.get("disorder") else "")
    if meta.get("seed"):
        workload += f" s{meta['seed']}"

    out = dict(
        workload=workload, family=trace.family, Lx=trace.Lx, Ly=trace.Ly, n=trace.n, D=trace.D, chi_b=chi_b,
        depth=trace.n_layers, seed_circ=meta.get("seed"), max_bond=trace.max_bond,
        disc_total=trace.disc_total, logfid_proxy=trace.logfid_proxy, mae_z=trace.mae_z,
        fidelity=trace.fidelity, infidelity=(None if trace.fidelity is None else 1.0 - trace.fidelity),
        maxerr_z=trace.maxerr_z, t_local_evolve=trace.t_evolve,
        placer=placer, P=P, ranks_per_node=ranks_per_node, n_nodes=topo.n_nodes, obs_mode=obs_mode,
        dynamic=bool(dynamic), oracle=oracle, t_search=t_search, n_repart=n_repart, n_tried=n_tried,
        mig_bytes=mig_bytes, t_mig=t_mig,
        peak_bytes=stats["peak"], imb_mean=float(np.mean(stats["imb"])), imb_max=float(np.max(stats["imb"])),
        imb_init=ev0["imbalance"], imb_final=evF["imbalance"],
        cut_bonds_init=ev0["cut_bonds"], cut_bonds_final=evF["cut_bonds"], cut_bonds_mean=float(np.mean(stats["cut"])),
        comm_model_init=ev0["comm"], comm_model_final=evF["comm"],
        line_handoffs_final=evF["line_handoffs"],
    )
    out.update(ledger.summary())
    out["bytes_cut"] = out["bytes_inter"] + out["bytes_intra"]
    out["peak_bytes"] = out["peak_resident"]          # tensors + workspace, from the ledger
    out["peak_tensor_bytes"] = stats["peak"]
    out["oom"] = out["oom_rounds"] > 0
    return out
