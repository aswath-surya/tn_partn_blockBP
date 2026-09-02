"""Pareto-frontier figures (brief Part VI numbering) from replay results."""
from __future__ import annotations

from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "rows": ("tab:gray", "s"), "cols": ("tab:gray", "D"), "tiles": ("tab:olive", "^"),
    "random": ("k", "x"), "kahypar_cut": ("tab:blue", "o"), "kahypar_km1": ("tab:cyan", "o"),
    "kahypar_km1_chi": ("tab:red", "*"), "ilp": ("tab:green", "P"),
}
MB = 1e6


def err_metric(r):
    """1 − fidelity when the exact overlap exists (n ≤ 20), else Σ discarded weight."""
    if r.get("fidelity") is not None:
        return max(1.0 - r["fidelity"], 1e-16)
    return max(r["disc_total"], 1e-16)


def err_label(rs):
    return "1 − fidelity (exact)" if any(r.get("fidelity") is not None for r in rs) else "Σ discarded weight"


def _ok(results, P=None, obs_mode=None):
    """Successful results; by default only the largest P (multi-node) and, if
    given, one observable mode — one panel per workload keeps figures legible."""
    res = [r for r in results if "error" not in r]
    if P == "max":
        P = max(r["P"] for r in res)
    if P is not None:
        res = [r for r in res if r["P"] == P]
    if obs_mode is not None:
        res = [r for r in res if r.get("obs_mode", "bmps") == obs_mode]
    return res


def _group(results, keys):
    g = defaultdict(list)
    for r in results:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def _panel_grid(n, ncols=3, size=(4.2, 3.4)):
    ncols = min(ncols, max(n, 1))
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def _pt(ax, x, y, r, s=40):
    c, mk = STYLE.get(r["placer"], ("m", "o"))
    hollow = r.get("dynamic") and mk not in ("x",)
    ax.scatter([x], [y], marker=mk, s=s, facecolors="none" if hollow else c, edgecolors=c,
               linewidths=1.2, label=_label(r))


def _label(r):
    return r["placer"] + (" (dyn)" if r.get("dynamic") else "") + (" (oracle)" if r.get("oracle") else "")


def fig_chi_heterogeneity(traces, path):
    """§2.4 prerequisite: χ_e spread vs depth, and the w_v spread at the end."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for tr in traces:
        st = np.array(tr.chi_stats_by_layer())
        lab = f"{tr.family} {tr.Lx}x{tr.Ly} D={tr.D}"
        l, = ax1.plot(st[:, 0], st[:, 2], "-o", ms=3, label=lab)
        ax1.fill_between(st[:, 0], st[:, 1], st[:, 4], color=l.get_color(), alpha=0.15)
        w = np.sort(tr.w_final)[::-1] * 16 / MB
        ax2.plot(np.arange(len(w)), w, "-", color=l.get_color(), label=lab)
    ax1.set(xlabel="layer", ylabel="χ_e (median; band = min..max)", title="bond-dimension heterogeneity vs depth")
    ax2.set(xlabel="site rank", ylabel="tensor bytes (MB)", yscale="log", title="final w_v = d·∏χ (sorted)")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_f4(results, path, P="max", obs_mode="bp"):
    """F4: cross-rank communication vs memory imbalance — the placement objective."""
    res = _ok(results, P, obs_mode)
    groups = _group(res, ("workload", "D", "P", "obs_mode"))
    fig, axes = _panel_grid(len(groups))
    for ax, (key, rs) in zip(axes, sorted(groups.items())):
        for r in rs:
            _pt(ax, r["imb_mean"], r["bytes_cut"] / MB, r)
        ax.set(title=f"{key[0]} D={key[1]} P={key[2]} {key[3]}", xlabel="mean memory imbalance",
               ylabel="cross-rank bytes (MB, inter+intra)", yscale="log")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_f3(results, path, P="max", obs_mode="bp"):
    """F3: truncation error vs communication, sweeping D (static placers)."""
    res = [r for r in _ok(results, P, obs_mode) if not r["dynamic"] and not r.get("oracle")]
    groups = _group(res, ("workload", "P", "obs_mode"))
    fig, axes = _panel_grid(len(groups))
    for ax, (key, rs) in zip(axes, sorted(groups.items())):
        for pl, prs in _group(rs, ("placer",)).items():
            prs = sorted(prs, key=lambda r: r["D"])
            c, mk = STYLE.get(pl[0], ("m", "o"))
            x = [err_metric(r) for r in prs]
            ax.plot(x, [r["bytes_cut"] / MB for r in prs], "-", marker=mk, color=c, label=pl[0])
            for r in prs:
                ax.annotate(f"D{r['D']}", (err_metric(r), r["bytes_cut"] / MB), fontsize=6)
        ax.set(title=f"{key[0]} P={key[1]} {key[2]}", xlabel=err_label(rs), ylabel="cross-rank bytes (MB)",
               xscale="log", yscale="log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_f2(results, path, P="max", obs_mode=None):
    """F2: modeled wall time (compute + communication stacked) per placer and D, with fidelity proxy."""
    res = [r for r in _ok(results, P, obs_mode) if not r.get("oracle")]
    groups = _group(res, ("workload", "P", "obs_mode"))
    fig, axes = _panel_grid(len(groups), size=(5.2, 3.6))
    for ax, (key, rs) in zip(axes, sorted(groups.items())):
        Ds = sorted({r["D"] for r in rs})
        labs = sorted({_label(r) for r in rs})
        wdt = 0.8 / len(labs)
        for k, lab in enumerate(labs):
            for i, D in enumerate(Ds):
                sub = [r for r in rs if _label(r) == lab and r["D"] == D]
                if not sub:
                    continue
                r = sub[0]
                c, _ = STYLE.get(r["placer"], ("m", "o"))
                x = i + (k - len(labs) / 2) * wdt
                ax.bar(x, r["t_comp"], wdt, color=c, alpha=0.9, label=lab if i == 0 else None)
                ax.bar(x, r["t_comm"], wdt, bottom=r["t_comp"], color=c, alpha=0.35, hatch="//")
        ax.set(xticks=range(len(Ds)), xticklabels=[f"D={D}" for D in Ds], ylabel="modeled seconds (hatched = comm)",
               title=f"{key[0]} P={key[1]} {key[2]}", yscale="log")
        ax2 = ax.twinx()
        fid = [(D, np.mean([err_metric(r) for r in rs if r["D"] == D])) for D in Ds]
        ax2.plot(range(len(Ds)), [max(f, 1e-16) for _, f in fid], "k--o", ms=3)
        ax2.set(ylabel=err_label(rs), yscale="log")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_f6(results, path, P="max", obs_mode="bp"):
    """F6: partition-search time vs modeled simulation time."""
    res = _ok(results, P, obs_mode)
    fig, ax = plt.subplots(figsize=(5, 4))
    for r in res:
        _pt(ax, max(r["t_search"], 1e-5), r["t_wall"], r, s=25)
    ax.set(xlabel="partition search time (s, laptop)", ylabel="modeled simulation wall (s)", xscale="log", yscale="log")
    hs = [plt.Line2D([], [], color=c, marker=mk, ls="") for _, (c, mk) in STYLE.items()]
    ax.legend(hs, list(STYLE), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
