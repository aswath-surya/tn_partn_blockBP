#!/usr/bin/env python
"""Calibrate the cost model from pulled Phase-0 JSONs (laptop side, after pull.sh).

  ../.venv/bin/python perlmutter/plot_phase0.py [results_perlmutter/phase0_<tag>.json ...]

Writes results_perlmutter/machine_calibrated.json (fields of tnpart.costmodel.Machine)
and results_perlmutter/fig_phase0.png. Derivations:
  p_gemm  = best complex128 GEMM rate at n >= 1024 (real FLOP/s)
  c_svd   = median over n of  t_svd(n) / (equivalent FLOPs / p_gemm)   (same for c_qr, c_eigh)
  bw_tier = best pingpong bandwidth over sizes;  lat_tier = pingpong time at 8 B
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


def calibrate(doc):
    loc = doc["local_by_node"][0]
    m = {}
    c128 = loc.get("complex128", [])
    big = [r for r in c128 if r["n"] >= 1024] or c128
    p_gemm = max(r["gemm_gflops"] for r in big) * 1e9
    m["p_gemm"] = p_gemm
    for key, name in (("svd_sq", "c_svd"), ("qr_tall", "c_qr"), ("eigh", "c_eigh")):
        ratios = [r[f"{key}_s"] / (r[f"{key}_gflops_equiv"] * 1e9 * r[f"{key}_s"] / p_gemm) for r in big]
        m[name] = float(np.median(ratios))
    net = doc["net"]
    for tier in ("intra", "inter"):
        rows = net["pingpong"].get(tier)
        if not rows:
            continue
        m[f"bw_{tier}"] = max(r["bw_GBs"] for r in rows) * 1e9
        m[f"lat_{tier}"] = min(r["t_s"] for r in rows if r["bytes"] <= 64)
    if net["allreduce"]:
        m["allreduce_bus_GBs"] = max(r["bus_bw_GBs"] for r in net["allreduce"])
    m["source"] = doc["tag"]
    m["gpu"] = doc["gpu"]
    m["cplx"], m["bytes_per_scalar"], m["hbm_bytes"], m["mem_headroom"], m["nic_share"] = 4.0, 16, 40e9, 3.0, 1.0
    m["name"] = f"perlmutter-{'a100' if doc['gpu'] else 'cpu'}-{doc['tag']}"
    return m


def plot(doc, path):
    loc = doc["local_by_node"][0]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.8))
    for dt, rows in loc.items():
        n = [r["n"] for r in rows]
        a1.plot(n, [r["gemm_gflops"] for r in rows], "-o", label=f"gemm {dt}")
        a1.plot(n, [r["svd_sq_gflops_equiv"] for r in rows], "--s", label=f"svd n×n {dt}")
        a1.plot(n, [r["qr_tall_gflops_equiv"] for r in rows], ":^", label=f"qr 4n×n {dt}")
    a1.set(xscale="log", yscale="log", xlabel="n", ylabel="GFLOP/s (real, equiv.)", title="local rates")
    a1.legend(fontsize=7)
    for tier, rows in doc["net"]["pingpong"].items():
        b = [r["bytes"] for r in rows]
        a2.plot(b, [r["t_s"] * 1e6 for r in rows], "-o", label=tier)
        a3.plot(b, [r["bw_GBs"] for r in rows], "-o", label=tier)
    a2.set(xscale="log", yscale="log", xlabel="bytes", ylabel="one-way time (µs)", title="pingpong latency")
    a3.set(xscale="log", xlabel="bytes", ylabel="GB/s", title="pingpong bandwidth")
    a2.legend(); a3.legend()
    fig.suptitle(f"Phase 0 — {doc['tag']} ({'GPU' if doc['gpu'] else 'CPU'}, {doc['size']} ranks)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(RP, "phase0_*.json")))
    if not paths:
        sys.exit(f"no phase0_*.json in {RP}; run perlmutter/pull.sh first")
    for p in paths:
        doc = json.load(open(p))
        m = calibrate(doc)
        print(f"== {p}")
        for k, v in m.items():
            print(f"  {k:18s} {v:.4g}" if isinstance(v, float) else f"  {k:18s} {v}")
        tag = doc["tag"]
        plot(doc, os.path.join(RP, f"fig_phase0_{tag}.png"))
        out = os.path.join(RP, f"machine_{tag}.json")
        json.dump(m, open(out, "w"), indent=1)
        print(f"  -> {out}, fig_phase0_{tag}.png")
    # default calibration: local rates from the most recent GPU run; each network tier from whichever
    # run measured it with the most points (the GPU network benchmark can be missing/partial)
    docs = [json.load(open(p)) for p in paths]
    gpu = [d for d in docs if d["gpu"]]
    m = calibrate(gpu[-1] if gpu else docs[-1])
    for tier in ("intra", "inter"):
        cands = [d for d in docs if len(d["net"]["pingpong"].get(tier, [])) >= 3]
        if cands:
            d = max(cands, key=lambda d: len(d["net"]["pingpong"][tier]))
            rows = d["net"]["pingpong"][tier]
            m[f"bw_{tier}"] = max(r["bw_GBs"] for r in rows) * 1e9
            m[f"lat_{tier}"] = min(r["t_s"] for r in rows if r["bytes"] <= 64)
            m[f"net_{tier}_source"] = d["tag"] + (" (GPU buffers)" if d["gpu"] else " (host buffers)")
        else:
            m.pop(f"bw_{tier}", None); m.pop(f"lat_{tier}", None)
    json.dump(m, open(os.path.join(RP, "machine_calibrated.json"), "w"), indent=1)
    print("default calibration ->", os.path.join(RP, "machine_calibrated.json"))
    for k, v in m.items():
        print(f"  {k:18s} {v:.4g}" if isinstance(v, float) else f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
