"""BlockBP on a random 1+1D circuit network with many spacetime tiles: two GIFs
(block arrows; explicit MPS messages) in results/blockbp_circuit{,_mps}.gif.

  ../.venv/bin/python make_blockbp_circuit_gif.py [--network norm|amplitude] [--n 24] [--depth 14] [--bq 4] [--bt 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from tnpart.blockbp import BlockBP  # noqa: E402
from tnpart.circuits1d import amplitude_tn, circuit_blocks, circuit_key, norm_tn, random_circuit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="norm", choices=["norm", "amplitude"])
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--bq", type=int, default=4); ap.add_argument("--bt", type=int, default=4)
    ap.add_argument("--chi-m", type=int, default=4); ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    circ = random_circuit(a.n, a.depth, seed=a.seed, p_gate=0.6)
    tn = amplitude_tn(circ) if a.network == "amplitude" else norm_tn(circ)
    base_block = circuit_blocks(a.bq, a.bt, a.depth)
    if a.network == "norm":
        def block_of(t):
            qs = [int(g[1:]) for g in t.tags if g.startswith("I") and g[1:].isdigit()]
            rs = [int(g.split("_")[1]) for g in t.tags if g.startswith("ROUND_")]
            r = -1 if "PSI0" in t.tags else rs[0]
            return (min(qs) // a.bq, (r + 1) // a.bt)
    else:
        block_of = base_block
    Z_exact = 1.0 + 0j if a.network == "norm" else complex(circ.amplitude("0" * a.n))
    bp = BlockBP(tn, block_of, chi_m=a.chi_m, order_key=circuit_key(tn), init="random", seed=a.seed,
                 optimize="greedy", tol=1e-9, max_iter=a.sweeps, dense_limit=2 ** 24)

    # ---- spacetime positions: x = qubit, y = round (PSI0 at −1, BRA at depth); bra layer nudged
    pos, blk = {}, {}
    for tid, t in tn.tensor_map.items():
        qs = [int(g[1:]) for g in t.tags if g.startswith("I") and g[1:].isdigit()]
        rs = [int(g.split("_")[1]) for g in t.tags if g.startswith("ROUND_")]
        r = -1 if "PSI0" in t.tags else (a.depth if ("BRA" in t.tags and a.network == "amplitude") else rs[0])
        x = np.mean(qs); y = float(r)
        if a.network == "norm" and "BRA" in t.tags:
            x += 0.18; y += 0.22
        pos[tid] = np.array([x, y]); blk[tid] = block_of(t)
    name2tids = {ind: tuple(tids) for ind, tids in tn.ind_map.items()}
    cut = [ind for ind, tids in name2tids.items() if blk[tids[0]] != blk[tids[1]]]
    K = len(bp.blocks)
    nq = max(b[0] for b in bp.blocks) + 1
    cols = plt.cm.tab20(np.linspace(0, 1, 20))
    color = {A: cols[(A[0] * 7 + A[1] * 3) % 20] for A in bp.blocks}
    cent = {A: np.mean([pos[t] for t in pos if blk[t] == A], axis=0) for A in bp.blocks}
    rect = {A: (A[0] * a.bq - 0.5, A[1] * a.bt - 1.5, a.bq, a.bt) for A in bp.blocks}

    frames = [(0, {k: 1.0 for k in bp.m}, float("nan"), abs(bp.Z() / Z_exact - 1))]
    for it in range(1, a.sweeps + 1):
        d = bp.iterate()
        frames.append((it, dict(bp.last_edge_delta), d, abs(bp.Z() / Z_exact - 1)))
        print(f"  sweep {it:2d}: Δ={d:.2e}  |Z_bp/Z−1|={frames[-1][3]:.2e}", flush=True)
    print(f"{a.network}: {len(pos)} tensors, {len(name2tids)} bonds, {K} blocks ({nq}×{K//nq}), {len(cut)} cut bonds, "
          f"max boundary {max(len(v) for v in bp.order.values())}")

    def base(ax, i):
        it, ed, d, e = frames[i]
        for A in bp.blocks:
            x0, y0, w, h = rect[A]
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor=color[A], alpha=0.13, edgecolor=color[A], lw=0.8))
        for ind, (u, v) in name2tids.items():
            c = "k" if ind in cut else "0.8"
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=c, lw=1.0 if c == "k" else 0.5, zorder=1)
        for t in pos:
            ax.scatter(*pos[t], s=9, color=color[blk[t]], edgecolor="none", zorder=3)
        ax.set_xlim(-1, a.n); ax.set_ylim(-1.8, a.depth + 0.8); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("qubit →", fontsize=8); ax.set_ylabel("time →", fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)

    def panels(axd, axe, i):
        its = [f[0] for f in frames[1:i + 1]]
        axd.semilogy(its, [max(f[2], 1e-12) for f in frames[1:i + 1]], "k-o", ms=3)
        axd.set_xlim(0, a.sweeps); axd.set_ylim(1e-9, 3); axd.set_title("Δ: average change of the messages", fontsize=9)
        axd.set_xlabel("sweep", fontsize=8); axd.tick_params(labelsize=7)
        axe.semilogy([f[0] for f in frames[:i + 1]], [max(f[3], 1e-16) for f in frames[:i + 1]], "r-o", ms=3)
        axe.set_xlim(0, a.sweeps); axe.set_ylim(1e-8, 10)
        axe.set_title("|Z_BlockBP / Z_exact − 1|" + (" (Z_exact = 1)" if a.network == "norm" else ""), fontsize=9)
        axe.set_xlabel("sweep", fontsize=8); axe.tick_params(labelsize=7)

    lab = f"{a.network} network of a random {a.n}-qubit, depth-{a.depth} circuit; {K} tiles of {a.bq} qubits × {a.bt} layers"
    # ---- render 1: arrows
    fig = plt.figure(figsize=(12, 6)); axg = fig.add_axes([0.03, 0.06, 0.6, 0.88])
    axd = fig.add_axes([0.70, 0.58, 0.28, 0.34]); axe = fig.add_axes([0.70, 0.10, 0.28, 0.34])

    def draw(i):
        it, ed, d, e = frames[i]
        axg.clear(); axd.clear(); axe.clear(); base(axg, i)
        if i >= 1:
            for (A, B), val in ed.items():
                p, q = cent[A], cent[B]
                off = np.array([-(q[1] - p[1]), q[0] - p[0]]); off = 0.25 * off / (np.linalg.norm(off) + 1e-12)
                axg.add_patch(FancyArrowPatch(p + off, q + off, arrowstyle="-|>", mutation_scale=8 + 5 * min(val, 1),
                                              lw=0.4 + 4 * min(val, 1), color=color[A], alpha=0.3 + 0.7 * min(val, 1),
                                              zorder=4, shrinkA=8, shrinkB=8, connectionstyle="arc3,rad=0.12"))
        axg.set_title(lab + (f"   — sweep {it}" if i >= 1 else "   — the partition"), fontsize=9, loc="left")
        panels(axd, axe, i); return []

    out = os.path.join("results", f"blockbp_circuit_{a.network}.gif")
    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(out, writer=PillowWriter(fps=1.2))
    print("gif ->", out)

    # ---- render 2: MPS messages
    fig2 = plt.figure(figsize=(12, 6)); ax2 = fig2.add_axes([0.03, 0.06, 0.6, 0.88])
    ax2d = fig2.add_axes([0.70, 0.58, 0.28, 0.34]); ax2e = fig2.add_axes([0.70, 0.10, 0.28, 0.34])

    def sites(A, B):
        pts = []
        for nm in bp.order[frozenset((A, B))]:
            u, v = name2tids[nm]
            mid = (pos[u] + pos[v]) / 2; dvec = pos[v] - pos[u]
            nrm = np.array([-dvec[1], dvec[0]]); nrm /= (np.linalg.norm(nrm) + 1e-12)
            if np.dot(nrm, cent[B] - mid) < 0:
                nrm = -nrm
            pts.append(mid + 0.28 * nrm)
        return np.array(pts)

    def draw2(i):
        it, ed, d, e = frames[i]
        ax2.clear(); ax2d.clear(); ax2e.clear(); base(ax2, i)
        if i >= 1:
            for (A, B), val in ed.items():
                P = sites(A, B); al = 0.25 + 0.75 * min(val, 1.0)
                if len(P) > 1:
                    ax2.plot(P[:, 0], P[:, 1], color=color[A], lw=0.8 + 2.0 * min(val, 1.0), alpha=al, zorder=4)
                ax2.scatter(P[:, 0], P[:, 1], marker="s", s=10 + 22 * min(val, 1.0), color=color[A], edgecolor="k",
                            lw=0.3, alpha=al, zorder=5)
        ax2.set_title(lab + (f"   — sweep {it}: squares = MPS sites, one per cut bond" if i >= 1 else "   — the partition"),
                      fontsize=9, loc="left")
        panels(ax2d, ax2e, i); return []

    out2 = os.path.join("results", f"blockbp_circuit_{a.network}_mps.gif")
    FuncAnimation(fig2, draw2, frames=len(frames), blit=False).save(out2, writer=PillowWriter(fps=1.2))
    print("gif ->", out2)


if __name__ == "__main__":
    main()
