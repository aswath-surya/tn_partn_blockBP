#!/usr/bin/env python
"""Aggregate Phase-1 runs (results_perlmutter/mpi_p1_*.json after pull.sh).

  ../.venv/bin/python perlmutter/plot_phase1.py [dir-or-files]

Groups by (L, D, placer, n_nodes×ranks_per_node); reports medians with min/max
over repeats (brief §7: never means), the 2-node / 1-node wall-time ratio (the
inter-node penalty), and per-phase compute vs time-in-MPI. Writes
results_perlmutter/fig_phase1.png and prints the table.
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


def load(paths):
    rows = []
    for p in paths:
        r = json.load(open(p))
        sp = r["spec"]
        rpn = r.get("ranks_per_node", 4)
        r["n_nodes"] = r.get("n_nodes") or -(-r["P"] // rpn)
        r["layout"] = f"{r['n_nodes']}x{rpn}"
        r["key"] = (sp["Lx"], r["D"], r["placer"], r["P"])
        # decomposition from the slowest rank (maxes over different ranks are not additive)
        slow = max(r["per_rank"], key=lambda t: t["total"])
        r["comm"] = slow["gate_comm"] + slow["gauge_comm"] + slow["sync"]
        r["comp"] = slow["gate_comp"] + slow["gauge_comp"]
        rows.append(r)
    return rows


def med(xs):
    return float(np.median(xs)), float(np.min(xs)), float(np.max(xs))


def main():
    args = sys.argv[1:] or [RP]
    paths = []
    for a in args:
        paths += sorted(glob.glob(os.path.join(a, "mpi_*.json"))) if os.path.isdir(a) else [a]
    rows = load(paths)
    if not rows:
        sys.exit(f"no mpi_*.json found in {args}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["key"], r["layout"])].append(r)
    print(f"{'L':>3s} {'D':>3s} {'placer':16s} {'P':>2s} {'layout':7s} {'n':>2s} {'wall med [min,max] s':>24s} {'comp':>7s} {'comm':>7s} {'comm%':>6s} {'msgs':>6s} {'MB':>7s} {'cut':>4s}")
    table = {}
    for (key, layout), rs in sorted(groups.items()):
        w = med([r["wall"] for r in rs]); c = med([r["comp"] for r in rs]); m = med([r["comm"] for r in rs])
        table[(key, layout)] = w
        print(f"{key[0]:3d} {key[1]:3d} {key[2]:16s} {key[3]:2d} {layout:7s} {len(rs):2d} {w[0]:9.3f} [{w[1]:7.3f},{w[2]:7.3f}] "
              f"{c[0]:7.3f} {m[0]:7.3f} {100*m[0]/max(w[0],1e-12):6.1f} {int(np.median([r['n_msgs_max'] for r in rs])):6d} "
              f"{np.median([r['bytes_max'] for r in rs])/1e6:7.2f} {rs[0]['cut_bonds']:4d}")
    # inter-node penalty: layouts with >1 node vs the 1-node layout at equal total ranks
    print("\ninter-node penalty (wall multi-node / wall 1-node, medians):")
    keys = sorted({k for k, _ in table})
    for key in keys:
        one = [(lay, t) for (k, lay), t in table.items() if k == key and lay.startswith("1x")]
        multi = [(lay, t) for (k, lay), t in table.items() if k == key and not lay.startswith("1x")]
        for lay1, t1 in one:
            for lay2, t2 in multi:
                print(f"  L={key[0]} D={key[1]} {key[2]:16s} P={key[3]} {lay2} / {lay1}: {t2[0]/t1[0]:.3f}")
    # figure: one panel per (L, D); bars per placer, one bar per layout, comm hatched
    LDs = sorted({(k[0], k[1], k[3]) for k, _ in table})          # (L, D, P) panels
    fig, axes = plt.subplots(1, len(LDs), figsize=(4.5 * len(LDs), 3.8), squeeze=False)
    for ax, LD in zip(axes.ravel(), LDs):
        placers = sorted({k[2] for k, _ in table if (k[0], k[1], k[3]) == LD})
        layouts = sorted({lay for k, lay in table if (k[0], k[1], k[3]) == LD})
        wdt = 0.8 / len(layouts)
        for j, lay in enumerate(layouts):
            for i, pl in enumerate(placers):
                rs = groups.get(((LD[0], LD[1], pl, LD[2]), lay))
                if not rs:
                    continue
                w = med([r["wall"] for r in rs]); m = med([r["comm"] for r in rs])
                x = i + (j - len(layouts) / 2 + 0.5) * wdt
                ax.bar(x, w[0] - m[0], wdt, color=f"C{j}", label=lay if i == 0 else None)
                ax.bar(x, m[0], wdt, bottom=w[0] - m[0], color=f"C{j}", alpha=0.4, hatch="//")
                ax.errorbar(x, w[0], yerr=[[w[0] - w[1]], [w[2] - w[0]]], color="k", capsize=2, lw=0.8)
        ax.set(xticks=range(len(placers)), xticklabels=placers, ylabel="wall (s); hatched = time in MPI",
               title=f"L={LD[0]} D={LD[1]} P={LD[2]}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8, title="nodes×ranks")
    fig.tight_layout()
    out = os.path.join(RP if os.path.isdir(args[0]) else os.path.dirname(args[0]) or ".", "fig_phase1.png")
    fig.savefig(out, dpi=130)
    print("figure ->", out)


if __name__ == "__main__":
    main()
