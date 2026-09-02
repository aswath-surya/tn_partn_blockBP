"""Animate BlockBP on a generic graph: results/blockbp_generic.gif (~1 min to render).

  ../.venv/bin/python make_blockbp_gif.py

A random geometric graph carries a random positive tensor network; blocks come from
modularity communities; the real BlockBP (tnpart.blockbp) runs sweep by sweep. Arrow
width = that directed message's change in the current sweep; right panels track the
global Δ and the error of the Bethe estimate against the exact contraction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
import networkx as nx
import numpy as np
import quimb.tensor as qtn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from tnpart.blockbp import BlockBP  # noqa: E402

SEED, N, RADIUS, CHI_M, SWEEPS = 7, 34, 0.30, 4, 12
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "blockbp_generic.gif")

def main():
    # ---- graph + tensor network
    rng = np.random.default_rng(SEED)
    G = nx.random_geometric_graph(N, RADIUS, seed=SEED)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    pos = nx.get_node_attributes(G, "pos")
    ind = {tuple(sorted(e)): f"e{min(e)}_{max(e)}" for e in G.edges}
    tensors = []
    for v in G.nodes:
        inds = [ind[tuple(sorted((v, u)))] for u in G.neighbors(v)]
        data = rng.uniform(0.2, 1.0, size=(2,) * len(inds))
        tensors.append(qtn.Tensor(data, inds=inds, tags={f"n{v}"}))
    tn = qtn.TensorNetwork(tensors)
    Z_exact = complex(tn.contract(optimize="greedy"))

    # ---- blocks: modularity communities
    comms = list(nx.community.greedy_modularity_communities(G, resolution=1.1))
    block_of_node = {v: k for k, c in enumerate(comms) for v in c}
    tagmap = {f"n{v}": block_of_node[v] for v in G.nodes}
    bp = BlockBP(tn, tagmap, chi_m=CHI_M, init="random", seed=SEED, optimize="greedy", tol=1e-9, max_iter=SWEEPS)
    K = len(bp.blocks)
    cent = {A: np.mean([pos[v] for v in G.nodes if block_of_node[v] == A], axis=0) for A in bp.blocks}
    cut = [e for e in G.edges if block_of_node[e[0]] != block_of_node[e[1]]]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:K]

    # ---- run and record
    frames = []            # (sweep, edge_delta dict, Δ, err)
    err0 = abs(bp.Z() / Z_exact - 1)
    frames.append((0, {k: 1.0 for k in bp.m}, float("nan"), err0))
    for it in range(1, SWEEPS + 1):
        d = bp.iterate()
        frames.append((it, dict(bp.last_edge_delta), d, abs(bp.Z() / Z_exact - 1)))
    print(f"graph: {G.number_of_nodes()} tensors, {G.number_of_edges()} bonds, {K} blocks, {len(cut)} cut bonds, "
          f"tree={bp.is_tree}; exact log|Z|={np.log(abs(Z_exact)):.3f}")
    for it, _, d, e in frames:
        print(f"  sweep {it:2d}: Δ={d:.2e}  |Z_bp/Z−1|={e:.2e}")

    # ---- drawing
    fig = plt.figure(figsize=(11, 5.2))
    axg = fig.add_axes([0.02, 0.04, 0.56, 0.9])
    axd = fig.add_axes([0.66, 0.58, 0.32, 0.36])
    axe = fig.add_axes([0.66, 0.10, 0.32, 0.36])
    captions = ["1. a tensor network on a random graph, cut into blocks (communities)",
                "2. one message per direction on every block edge; width = how much it just changed",
                "3. sweeps: each block contracts itself with its incoming messages and sends the result on"]


    def draw(i):
        it, ed, d, e = frames[i]
        axg.clear(); axd.clear(); axe.clear()
        for A in bp.blocks:
            pts = np.array([pos[v] for v in G.nodes if block_of_node[v] == A])
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                axg.add_patch(Polygon(pts[hull.vertices], closed=True, facecolor=colors[A], alpha=0.12, edgecolor="none"))
        for u, v in G.edges:
            c = "0.75" if (u, v) not in cut else "k"
            axg.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=c, lw=0.8 if c == "0.75" else 1.6, zorder=1)
        for v in G.nodes:
            axg.scatter(*pos[v], s=70, color=colors[block_of_node[v]], edgecolor="k", lw=0.5, zorder=3)
        if i >= 1:
            for (A, B), val in ed.items():
                a, b = cent[A], cent[B]
                off = np.array([-(b[1] - a[1]), b[0] - a[0]]); off = 0.02 * off / (np.linalg.norm(off) + 1e-12)
                w = 0.4 + 5.0 * min(val, 1.0)
                axg.add_patch(FancyArrowPatch(a + off, b + off, arrowstyle="-|>", mutation_scale=10 + 6 * min(val, 1),
                                              lw=w, color=colors[A], alpha=0.35 + 0.65 * min(val, 1), zorder=4,
                                              shrinkA=12, shrinkB=12, connectionstyle="arc3,rad=0.15"))
        for A in bp.blocks:
            axg.text(*cent[A], f"{A}", ha="center", va="center", fontsize=9, weight="bold",
                     bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=colors[A]))
        axg.set_xlim(-0.05, 1.05); axg.set_ylim(-0.05, 1.05); axg.set_xticks([]); axg.set_yticks([])
        for s in axg.spines.values():
            s.set_visible(False)
        axg.set_title(captions[min(i, 2)] + (f"   — sweep {it}" if i >= 2 else ""), fontsize=10, loc="left")
        its = [f[0] for f in frames[1:i + 1]]
        axd.semilogy(its, [max(f[2], 1e-12) for f in frames[1:i + 1]], "k-o", ms=3)
        axd.set_xlim(0, SWEEPS); axd.set_ylim(1e-9, 2); axd.set_title("Δ: average change of the messages", fontsize=9)
        axd.set_xlabel("sweep", fontsize=8); axd.tick_params(labelsize=7)
        axe.semilogy([f[0] for f in frames[:i + 1]], [max(f[3], 1e-16) for f in frames[:i + 1]], "r-o", ms=3)
        axe.set_xlim(0, SWEEPS); axe.set_ylim(1e-6, 10); axe.set_title("|Z_BlockBP / Z_exact − 1|", fontsize=9)
        axe.set_xlabel("sweep", fontsize=8); axe.tick_params(labelsize=7)
        return []


    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(OUT, writer=PillowWriter(fps=1.2))
    print("gif ->", OUT, f"({os.path.getsize(OUT)/1e6:.2f} MB, {len(frames)} frames)")

    # ---- second render: messages drawn as MPSs — one site per cut bond, bars = MPS bonds
    OUT2 = OUT.replace(".gif", "_mps.gif")
    name2edge = {v: k for k, v in ind.items()}
    fig2 = plt.figure(figsize=(11, 5.2))
    ax2 = fig2.add_axes([0.02, 0.04, 0.56, 0.9])
    ax2d = fig2.add_axes([0.66, 0.58, 0.32, 0.36])
    ax2e = fig2.add_axes([0.66, 0.10, 0.32, 0.36])

    def mps_sites(A, B):
        """site positions of m_{A→B}: cut-bond midpoints in the message's bond order,
        shifted a little toward the receiving block B."""
        pts = []
        for nm in bp.order[frozenset((A, B))]:
            u, v = name2edge[nm]
            mid = (np.array(pos[u]) + np.array(pos[v])) / 2
            d = np.array(pos[v]) - np.array(pos[u]); n = np.array([-d[1], d[0]]); n /= (np.linalg.norm(n) + 1e-12)
            if np.dot(n, cent[B] - mid) < 0:
                n = -n
            pts.append(mid + 0.022 * n)
        return np.array(pts)

    def draw2(i):
        it, ed, d, e = frames[i]
        ax2.clear(); ax2d.clear(); ax2e.clear()
        for A in bp.blocks:
            pts = np.array([pos[v] for v in G.nodes if block_of_node[v] == A])
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                ax2.add_patch(Polygon(pts[hull.vertices], closed=True, facecolor=colors[A], alpha=0.10, edgecolor="none"))
        for u, v in G.edges:
            c = "0.8" if (u, v) not in cut else "0.3"
            ax2.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=c, lw=0.7 if c == "0.8" else 1.2, zorder=1)
        for v in G.nodes:
            ax2.scatter(*pos[v], s=45, color=colors[block_of_node[v]], edgecolor="k", lw=0.4, zorder=3)
        if i >= 1:
            for (A, B), val in ed.items():
                P = mps_sites(A, B)
                a = 0.25 + 0.75 * min(val, 1.0)
                if len(P) > 1:
                    ax2.plot(P[:, 0], P[:, 1], color=colors[A], lw=1.2 + 2.5 * min(val, 1.0), alpha=a, zorder=4)  # MPS bonds
                ax2.scatter(P[:, 0], P[:, 1], marker="s", s=26 + 40 * min(val, 1.0), color=colors[A], edgecolor="k",
                            lw=0.5, alpha=a, zorder=5)                                                          # MPS sites
        for A in bp.blocks:
            ax2.text(*cent[A], f"{A}", ha="center", va="center", fontsize=9, weight="bold",
                     bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=colors[A]))
        ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.05, 1.05); ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
        cap = ["1. blocks and cut bonds", "2. each message is an MPS: one site (square) per cut bond, bars = its bonds (χ_m)",
               "3. size/opacity = how much that message changed this sweep"]
        ax2.set_title(cap[min(i, 2)] + (f"   — sweep {it}" if i >= 2 else ""), fontsize=10, loc="left")
        its = [f[0] for f in frames[1:i + 1]]
        ax2d.semilogy(its, [max(f[2], 1e-12) for f in frames[1:i + 1]], "k-o", ms=3)
        ax2d.set_xlim(0, SWEEPS); ax2d.set_ylim(1e-9, 2); ax2d.set_title("Δ: average change of the messages", fontsize=9)
        ax2d.set_xlabel("sweep", fontsize=8); ax2d.tick_params(labelsize=7)
        ax2e.semilogy([f[0] for f in frames[:i + 1]], [max(f[3], 1e-16) for f in frames[:i + 1]], "r-o", ms=3)
        ax2e.set_xlim(0, SWEEPS); ax2e.set_ylim(1e-6, 10); ax2e.set_title("|Z_BlockBP / Z_exact − 1|", fontsize=9)
        ax2e.set_xlabel("sweep", fontsize=8); ax2e.tick_params(labelsize=7)
        return []

    anim2 = FuncAnimation(fig2, draw2, frames=len(frames), blit=False)
    anim2.save(OUT2, writer=PillowWriter(fps=1.2))
    print("gif ->", OUT2, f"({os.path.getsize(OUT2)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
