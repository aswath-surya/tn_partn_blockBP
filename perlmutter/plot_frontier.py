#!/usr/bin/env python
"""Measured Pareto frontiers from frontier_job.sbatch runs (results_perlmutter/mpi_fr_*.json).

  ../.venv/bin/python perlmutter/plot_frontier.py [dir-or-files]

Panels: (A) Σ discarded weight vs wall and vs time-in-MPI (slowest rank) per placer across D
        at fixed (L, P) — the measured F3/F2;  (B) strong scaling: wall and MPI fraction vs P
        at fixed (L, D) per placer — F5.  Medians over repeats with min/max bars.
Writes results_perlmutter/fig_frontier_measured.png and prints the tables.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RP = os.path.join(HERE, "..", "results_perlmutter")
STYLE = {"rows": ("tab:gray", "s"), "cols": ("tab:gray", "D"), "tiles": ("tab:olive", "^"), "random": ("k", "x"),
         "kahypar_cut": ("tab:blue", "o"), "kahypar_km1": ("tab:cyan", "o"), "kahypar_km1_chi": ("tab:red", "*")}


def load(paths):
    rows = []
    for p in paths:
        r = json.load(open(p))
        slow = max(r["per_rank"], key=lambda t: t["total"])
        r["comm"] = slow["gate_comm"] + slow["gauge_comm"] + slow["sync"]
        r["comp"] = slow["gate_comp"] + slow["gauge_comp"]
        r["L"] = r["spec"]["Lx"]
        r["n_nodes"] = r.get("n_nodes") or -(-r["P"] // r.get("ranks_per_node", 4))
        rows.append(r)
    return rows


def med(xs):
    return float(np.median(xs)), float(np.min(xs)), float(np.max(xs))


def agg(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = dict(n=len(rs), wall=med([r["wall"] for r in rs]), comm=med([r["comm"] for r in rs]),
                      comp=med([r["comp"] for r in rs]), disc=float(np.median([r["disc_total"] for r in rs])),
                      msgs=int(np.median([r["n_msgs_max"] for r in rs])), MB=float(np.median([r["bytes_max"] for r in rs])) / 1e6,
                      cut=rs[0]["cut_bonds"], max_chi=rs[0]["max_chi"])
    return out


def main():
    args = sys.argv[1:] or [RP]
    paths = []
    for a in args:
        paths += sorted(glob.glob(os.path.join(a, "mpi_*.json"))) if os.path.isdir(a) else [a]
    rows = load(paths)
    if not rows:
        sys.exit(f"no mpi_*.json in {args}")
    # ---- (A) D-sweep: for each (L, P, n_nodes) with >1 distinct D
    A = agg(rows, ("L", "P", "n_nodes", "placer", "D"))
    groupsA = defaultdict(dict)
    for (L, P, nn, pl, D), v in A.items():
        groupsA[(L, P, nn)].setdefault(pl, {})[D] = v
    groupsA = {k: v for k, v in groupsA.items() if any(len(d) > 1 for d in v.values())}
    # ---- (B) node sweep: for each (L, D) with >1 distinct P
    B = agg(rows, ("L", "D", "placer", "P", "n_nodes"))
    groupsB = defaultdict(dict)
    for (L, D, pl, P, nn), v in B.items():
        groupsB[(L, D)].setdefault(pl, {})[(P, nn)] = v
    groupsB = {k: v for k, v in groupsB.items() if any(len(d) > 1 for d in v.values())}

    print("== (A) D-sweep: L P nodes placer D | maxχ Σdisc | wall med[min,max] comp comm MPI% msgs MB cut")
    for (L, P, nn), pls in sorted(groupsA.items()):
        for pl, byD in sorted(pls.items()):
            for D, v in sorted(byD.items()):
                w, c, m = v["wall"], v["comp"], v["comm"]
                print(f"  {L:2d} {P:2d} {nn:2d} {pl:16s} D={D:3d} | {v['max_chi']:3d} {v['disc']:9.2e} | {w[0]:8.2f}[{w[1]:7.2f},{w[2]:7.2f}] "
                      f"{c[0]:7.2f} {m[0]:7.2f} {100*m[0]/max(w[0],1e-9):5.1f} {v['msgs']:6d} {v['MB']:8.1f} {v['cut']:4d}")
    print("== (B) node-sweep: L D placer P(nodes) | wall med  efficiency vs smallest P  MPI%")
    for (L, D), pls in sorted(groupsB.items()):
        for pl, byP in sorted(pls.items()):
            base = min(byP)
            for (P, nn), v in sorted(byP.items()):
                eff = byP[base]["wall"][0] * base[0] / (v["wall"][0] * P)
                print(f"  {L:2d} {D:3d} {pl:16s} P={P:2d}({nn}n) | {v['wall'][0]:8.2f}  eff={eff:5.2f}  {100*v['comm'][0]/max(v['wall'][0],1e-9):5.1f}")

    nA, nB = len(groupsA), len(groupsB)
    if nA + nB == 0:
        sys.exit("nothing to plot: need >1 D at fixed (L,P) or >1 P at fixed (L,D)")
    fig, axes = plt.subplots(1, 2 * nA + nB, figsize=(4.6 * (2 * nA + nB), 3.9), squeeze=False)
    axes = axes.ravel(); i = 0
    for (L, P, nn), pls in sorted(groupsA.items()):
        for ykey, ylab in (("wall", "wall (s)"), ("comm", "time in MPI, slowest rank (s)")):
            ax = axes[i]; i += 1
            for pl, byD in sorted(pls.items()):
                Ds = sorted(byD); c, mk = STYLE.get(pl, ("m", "o"))
                x = [max(byD[D]["disc"], 1e-16) for D in Ds]; y = [byD[D][ykey][0] for D in Ds]
                yerr = [[byD[D][ykey][0] - byD[D][ykey][1] for D in Ds], [byD[D][ykey][2] - byD[D][ykey][0] for D in Ds]]
                ax.errorbar(x, y, yerr=yerr, marker=mk, color=c, capsize=2, label=pl)
                for D, xx, yy in zip(Ds, x, y):
                    ax.annotate(f"D{D}", (xx, yy), fontsize=6)
            ax.set(xscale="log", yscale="log", xlabel="Σ discarded weight (↓ = more accurate)", ylabel=ylab,
                   title=f"measured F3: L={L} P={P} ({nn} nodes)")
            ax.legend(fontsize=7)
    for (L, D), pls in sorted(groupsB.items()):
        ax = axes[i]; i += 1
        for pl, byP in sorted(pls.items()):
            Ps = sorted(byP); c, mk = STYLE.get(pl, ("m", "o"))
            ax.plot([p for p, _ in Ps], [byP[k]["wall"][0] for k in Ps], marker=mk, color=c, label=f"{pl} wall")
            ax.plot([p for p, _ in Ps], [byP[k]["comm"][0] for k in Ps], marker=mk, color=c, ls="--", alpha=0.6, label=f"{pl} MPI")
        ax.set(xscale="log", yscale="log", xlabel="ranks P (1 GPU each)", ylabel="seconds", title=f"F5 scaling: L={L} D={D}")
        ax.set_xticks(sorted({p for pls_ in pls.values() for p, _ in pls_}), [str(p) for p in sorted({p for pls_ in pls.values() for p, _ in pls_})])
        ax.legend(fontsize=6)
    fig.tight_layout()
    out = os.path.join(RP if os.path.isdir(args[0]) else os.path.dirname(args[0]) or ".", "fig_frontier_measured.png")
    fig.savefig(out, dpi=130); print("figure ->", out)


if __name__ == "__main__":
    main()
