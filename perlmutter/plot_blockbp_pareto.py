#!/usr/bin/env python
"""Pareto fronts from blockbp_pareto.sbatch outputs (results_perlmutter/blockbp_{A,B,C}_*.json).

  ../.venv/bin/python perlmutter/plot_blockbp_pareto.py [dir]

Points = (run, strategy). Cost axes: seconds per sweep, production message bytes
16·L·2·χ_m² summed over block edges. Accuracy axes: |Z/Z_exact − 1| (A, B) or best Δ (C).
Fronts: non-dominated points in (cost, error) per sweep family. Writes fig_blockbp_pareto.png
and prints the front tables.
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RP = os.path.join(HERE, "..", "results_perlmutter")


def points(paths):
    pts = []
    for p in paths:
        d = json.load(open(p)); a = d["args"]
        fam = os.path.basename(p).split("_")[1][0]
        ref = d.get("exact_logZ")
        for r in d["results"]:
            lz = complex(*r["best_logZ"])
            err = abs(np.exp(lz - complex(*ref)) - 1) if ref is not None else float("nan")
            L = d["max_boundary"]
            pts.append(dict(fam=fam, file=os.path.basename(p), bq=a["bq"], bt=a["bt"], chi=r["strategy"]["chi_m"],
                            n=a["n"], strat=r["strategy"]["name"], err=err, delta=r["best_delta"], conv=r["converged"],
                            sweeps=r["sweeps"], s_per_sweep=r["wall"] / max(r["sweeps"], 1),
                            msg_bytes=16 * L * 2 * r["strategy"]["chi_m"] ** 2 * d["n_edges"]))
    return pts


def front(pts, xkey, ykey):
    P = sorted([p for p in pts if np.isfinite(p[ykey])], key=lambda p: (p[xkey], p[ykey]))
    out, best = [], np.inf
    for p in P:
        if p[ykey] < best:
            out.append(p); best = p[ykey]
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else RP
    pts = points(sorted(glob.glob(os.path.join(d, "blockbp_[ABC]_*.json"))))
    if not pts:
        sys.exit(f"no blockbp_[ABC]_*.json in {d}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, fam, ykey, ylab in ((axes[0], "A", "err", "|Z/Z_exact − 1| (amplitude, n=16)"),
                                (axes[1], "B", "err", "|Z − 1| (norm network, 64 tiles)"),
                                (axes[2], "C", "delta", "best Δ (amplitude, 64 tiles)")):
        fp = [p for p in pts if p["fam"] == fam]
        for p in fp:
            ax.scatter(p["s_per_sweep"], max(p[ykey], 1e-16), s=12 + 3 * p["chi"], alpha=0.7,
                       marker={"parallel": "o", "colored": "s", "parallel_eta0.3": "^"}.get(p["strat"], "x"),
                       c=[[min(0.1 + 0.07 * p["bq"], 1.0), 0.2, max(1 - 0.05 * p["bq"], 0.0)]])
        fr = front(fp, "s_per_sweep", ykey)
        if fr:
            ax.plot([p["s_per_sweep"] for p in fr], [max(p[ykey], 1e-16) for p in fr], "k--", lw=1, label="front")
        ax.set(xscale="log", yscale="log", xlabel="seconds per sweep", ylabel=ylab, title=f"family {fam}")
        print(f"== family {fam}: front in (s/sweep, {ykey})  [size ∝ χ_m; colour ∝ bq; marker = strategy]")
        for p in fr:
            print(f"   bq={p['bq']:2d} bt={p['bt']:2d} χ_m={p['chi']:2d} {p['strat']:16s} {ykey}={p[ykey]:.2e} "
                  f"s/sweep={p['s_per_sweep']:.3f} sweeps={p['sweeps']:2d} conv={p['conv']} msgMB={p['msg_bytes']/1e6:.2f}")
    fig.tight_layout(); out = os.path.join(d, "fig_blockbp_pareto.png"); fig.savefig(out, dpi=130)
    print("figure ->", out)
    # second view: error vs production message bytes (the communication axis)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for p in pts:
        if p["fam"] in "AB" and np.isfinite(p["err"]):
            ax.scatter(p["msg_bytes"] / 1e6, max(p["err"], 1e-16), s=10 + 3 * p["chi"], alpha=0.7,
                       c="tab:blue" if p["fam"] == "A" else "tab:red")
    ax.set(xscale="log", yscale="log", xlabel="message bytes per sweep, MPS form (MB)", ylabel="error vs exact",
           title="accuracy vs communication (blue A, red B)")
    fig.tight_layout(); out2 = os.path.join(d, "fig_blockbp_pareto_comm.png"); fig.savefig(out2, dpi=130)
    print("figure ->", out2)


if __name__ == "__main__":
    main()
