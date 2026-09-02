"""Block sizing for BlockBP on a random 1+1D circuit: how large a spacetime tile
(bq qubits × bt layers) can one rank own before per-iteration compute or memory
becomes the limit, and what fraction of an iteration the MPS messages cost.

  ../.venv/bin/python blockbp_sizing.py [--machine results_perlmutter/machine_calibrated.json] \\
      [--bq 4 8 12 16] [--bt 4 8 12 16] [--chi-m 4 8 16] [--target-ms 2000] [--mem-gb 30]

Per tile it builds a real random circuit with the tile in the middle (neighbours on all
sides), attaches random incoming messages, and asks cotengra for the exact-contraction
cost and width of one outgoing message. Message bytes are given both as the MPS the
production code would send (16·L·2·χ_m²) and as the dense vector the convergence
runner sends (16·2^L). Times use the calibrated Machine (p_gemm, c_svd, tiers).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from tnpart.blockbp import BlockBP  # noqa: E402
from tnpart.circuits1d import amplitude_tn, circuit_blocks, circuit_key, random_circuit  # noqa: E402
from tnpart.costmodel import Machine  # noqa: E402


def tile_cost(bq, bt, chi_m, p_gate=0.6, seed=0, small_factor=10.0, machine=None):
    """Cost of one BlockBP iteration for the central tile of a 3×3 tile arrangement."""
    n, depth = 3 * bq, 3 * bt - 2
    circ = random_circuit(n, depth, seed=seed, p_gate=p_gate)
    tn = amplitude_tn(circ)
    bp = BlockBP(tn, circuit_blocks(bq, bt, depth), chi_m=chi_m, order_key=circuit_key(tn), init="random",
                 dense_limit=2 ** 40)
    A = (1, 1)
    out = dict(bq=bq, bt=bt, chi_m=chi_m, n_nbrs=len(bp.nbrs[A]))
    flops, width, L = 0.0, 0, {}
    for B in sorted(bp.nbrs[A], key=str):
        inds = bp.order[frozenset((A, B))]
        L[B] = len(inds)
        tnAB = bp._block_with_messages(A, exclude=B)
        tree = tnAB.contraction_tree(output_inds=inds, optimize="auto-hq")
        flops += float(tree.contraction_cost()) * 8            # complex MAC → 8 real flops
        width = max(width, float(tree.contraction_width()))
    Ls = list(L.values())
    out.update(L_t=max(Ls), L_q=min(Ls), L_max=max(Ls), width_log2=width, flops_iter=flops)
    m = machine or Machine()
    out["t_comp_ms"] = 1e3 * flops / m.p_gemm * small_factor
    out["mem_MB"] = 16 * 2 ** width / 1e6
    msgs_mps = sum(16 * l * 2 * chi_m ** 2 for l in Ls)
    msgs_dense = sum(16 * 2 ** l for l in Ls)
    out["msg_mps_KB"], out["msg_dense_KB"] = msgs_mps / 1e3, msgs_dense / 1e3
    # 4 neighbours: t-cuts cross nodes (rows of q-blocks share a node), q-cuts stay on the node
    t_inter = sum(16 * l * 2 * chi_m ** 2 / m.bw("inter") + m.lat("inter") for l in Ls[:2])
    t_intra = sum(16 * l * 2 * chi_m ** 2 / m.bw("intra") + m.lat("intra") for l in Ls[2:])
    out["t_comm_ms"] = 1e3 * (t_inter + t_intra)
    out["comm_frac"] = out["t_comm_ms"] / (out["t_comm_ms"] + out["t_comp_ms"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default=None)
    ap.add_argument("--bq", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--bt", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--chi-m", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--target-ms", type=float, default=2000.0, help="max compute per iteration per rank")
    ap.add_argument("--mem-gb", type=float, default=30.0)
    ap.add_argument("--small-factor", type=float, default=10.0, help="GEMM inefficiency for small dense contractions")
    ap.add_argument("--out", default="results/blockbp_sizing.json")
    args = ap.parse_args()
    m = Machine.from_json(args.machine) if args.machine else Machine()
    print(f"cost model (local run; constants from {m.name}):  p_gemm={m.p_gemm:.3g}  inter {m.lat('inter')*1e6:.1f}us/{m.bw('inter')/1e9:.0f}GB/s  "
          f"intra {m.lat('intra')*1e6:.1f}us/{m.bw('intra')/1e9:.0f}GB/s")
    print(f"{'bq':>3s} {'bt':>3s} {'χm':>3s} {'L_t':>4s} {'L_q':>4s} {'width':>6s} {'GF/iter':>8s} {'comp ms':>8s} {'mem MB':>8s} "
          f"{'msg MPS KB':>10s} {'msg dense KB':>12s} {'comm ms':>8s} {'comm%':>6s}")
    rows = []
    for bq in args.bq:
        for bt in args.bt:
            for chi in args.chi_m:
                try:
                    r = tile_cost(bq, bt, chi, small_factor=args.small_factor, machine=m)
                except Exception as e:
                    print(f"{bq:3d} {bt:3d} {chi:3d}  ERROR {type(e).__name__}: {e}")
                    continue
                rows.append(r)
                print(f"{bq:3d} {bt:3d} {chi:3d} {r['L_t']:4d} {r['L_q']:4d} {r['width_log2']:6.1f} {r['flops_iter']/1e9:8.2f} "
                      f"{r['t_comp_ms']:8.1f} {r['mem_MB']:8.1f} {r['msg_mps_KB']:10.1f} {r['msg_dense_KB']:12.1f} "
                      f"{r['t_comm_ms']:8.3f} {100*r['comm_frac']:6.2f}", flush=True)
    ok = [r for r in rows if r["t_comp_ms"] <= args.target_ms and r["mem_MB"] <= args.mem_gb * 1e3]
    if ok:
        best = max(ok, key=lambda r: (r["bq"] * r["bt"], -r["chi_m"]))
        print(f"\nlargest tile within {args.target_ms:.0f} ms/iter and {args.mem_gb:.0f} GB: "
              f"bq={best['bq']} bt={best['bt']} χ_m={best['chi_m']}  (comp {best['t_comp_ms']:.0f} ms, "
              f"mem {best['mem_MB']:.0f} MB, comm {100*best['comm_frac']:.2f}%)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(machine=m.name, rows=rows), open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
