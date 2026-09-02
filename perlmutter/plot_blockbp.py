#!/usr/bin/env python
"""Convergence plots for run_blockbp_mpi.py outputs (results_perlmutter/blockbp_*.json or results/).

  ../.venv/bin/python perlmutter/plot_blockbp.py [files...]

Per file: Δ vs sweep for every strategy (log scale), |Z/Z_ref − 1| vs sweep when an exact
reference exists (else the spread of log Z across strategies), and a summary table.
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


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(RP, "blockbp_*.json")))
    if not paths:
        sys.exit(f"no blockbp_*.json in {RP}")
    for p in paths:
        d = json.load(open(p))
        a = d["args"]
        ref = d.get("exact_logZ")
        print(f"== {os.path.basename(p)}: {a['network']} n={a['n']} depth={a['depth']} tiles {a['bq']}x{a['bt']} "
              f"P={d['P']} blocks={d['n_blocks']} edges={d['n_edges']} max∂={d['max_boundary']} χ_m={a['chi_m']}")
        print(f"   {'strategy':26s} {'sweeps':>6s} {'Δ_final':>9s} {'Δ_best':>9s} {'conv':>4s} {'|Z/Zref-1|':>10s} {'log|Z|':>8s} {'t[s]':>6s}")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for r in d["results"]:
            s = r["strategy"]; name = s["name"]
            dl = np.array(r["delta"]); lz = np.array(r["logZ"])
            axes[0].semilogy(np.arange(1, len(dl) + 1), np.maximum(dl, 1e-9), label=name, lw=1)
            if ref is not None:
                err = np.abs(np.exp(lz[:, 0] + 1j * lz[:, 1]) / np.exp(ref[0] + 1j * ref[1]) - 1)
                axes[1].semilogy(np.arange(1, len(err) + 1), np.maximum(err, 1e-16), lw=1)
                e_final = err[-1]
            else:
                axes[1].plot(np.arange(1, len(lz) + 1), lz[:, 0], lw=1)
                e_final = float("nan")
            print(f"   {name:26s} {r['sweeps']:6d} {dl[-1]:9.2e} {r['best_delta']:9.2e} {'yes' if r['converged'] else 'no':>4s} "
                  f"{e_final:10.2e} {lz[-1, 0]:8.3f} {r['wall']:6.0f}")
        axes[0].set(xlabel="sweep", ylabel="Δ (message distance)", title="convergence")
        axes[0].axhline(a["tol"], color="k", ls=":", lw=0.8)
        axes[1].set(xlabel="sweep", ylabel="|Z/Z_exact − 1|" if ref is not None else "Re log Z",
                    title="accuracy" if ref is not None else "log Z (no exact reference)")
        axes[0].legend(fontsize=6)
        fig.suptitle(os.path.basename(p))
        fig.tight_layout()
        out = p.replace(".json", ".png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print("   figure ->", out)


if __name__ == "__main__":
    main()
