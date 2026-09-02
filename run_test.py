"""Quick end-to-end test run (see README.md 'The test run').

    cd tn_partitioning && ../.venv/bin/python -u run_test.py [--quick]

Produces results/traces/*.pkl, results/results.jsonl and results/fig_*.png.
"""
import os
import sys
import time

# one BLAS thread per process: the fork pool provides the parallelism, and
# Accelerate/OpenBLAS oversubscription otherwise stalls everything
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tnpart.plots import fig_chi_heterogeneity, fig_f2, fig_f3, fig_f4, fig_f6  # noqa: E402
from tnpart.sweep import load_trace, run_replays, run_traces  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PLACERS = ["rows", "cols", "tiles", "random", "kahypar_cut", "kahypar_km1", "kahypar_km1_chi", "ilp"]


def trace_specs(quick):
    """Workloads chosen so D spans low→high fidelity and, for the disordered
    TFIM with a finite cutoff, χ_e is heterogeneous across the lattice."""
    specs = [dict(family="brickwork", Lx=4, Ly=4, depth=dp, D=D, seed=0, gauge_iters=5, chi_b=32)
             for dp in (2, 3) for D in (2, 4, 8)]
    specs += [dict(family="tfim", Lx=4, Ly=4, n_steps=8, dt=0.2, D=D, seed=0, gauge_iters=5, chi_b=32,
                   disorder=0.9, cutoff=1e-6, obs="X") for D in (2, 4, 8)]
    specs += [dict(family="tfim", Lx=6, Ly=6, n_steps=8, dt=0.2, D=4, seed=0, gauge_iters=5, chi_b=32,
                   disorder=0.9, cutoff=1e-6, obs="X")]
    # sustained χ heterogeneity (w_v spread ~10³): strong disorder, small dt, coarse cutoff.
    # D > 8 → no bMPS/exact reference; error metric is the discarded weight.
    specs += [dict(family="tfim", Lx=6, Ly=6, n_steps=10, dt=0.1, D=D, seed=0, gauge_iters=5, chi_b=32,
                   disorder=1.0, cutoff=1e-4, obs="X") for D in (4, 8, 16)]
    specs += [dict(family="tfim", Lx=8, Ly=8, n_steps=10, dt=0.1, D=16, seed=0, gauge_iters=5, chi_b=32,
                   disorder=1.0, cutoff=1e-4, obs="X")]
    if not quick:
        specs += [dict(family="tfim", Lx=6, Ly=6, n_steps=10, dt=0.15, D=32, seed=0, gauge_iters=5, chi_b=32,
                       disorder=1.0, cutoff=1e-4, obs="X")]
        specs += [dict(family="tfim", Lx=8, Ly=8, n_steps=10, dt=0.1, D=D, seed=s, gauge_iters=5, chi_b=32,
                       disorder=1.0, cutoff=1e-4, obs="X") for D in (8, 32) for s in (0, 1)]
        # model points matching perlmutter/frontier_job.sbatch (D-sweep at L=8; node-sweep at L=12, D=16)
        specs += [dict(family="tfim", Lx=8, Ly=8, n_steps=10, dt=0.1, D=4, seed=0, gauge_iters=5, chi_b=32,
                       disorder=1.0, cutoff=1e-4, obs="X"),
                  dict(family="tfim", Lx=12, Ly=12, n_steps=10, dt=0.1, D=16, seed=0, gauge_iters=5, chi_b=32,
                       disorder=1.0, cutoff=1e-4, obs="X")]
        specs += [dict(family="tfim", Lx=6, Ly=6, n_steps=8, dt=0.2, D=8, seed=0, gauge_iters=5, chi_b=32,
                       disorder=0.9, cutoff=1e-6, obs="X")]
        specs += [dict(family="tfim", Lx=4, Ly=4, n_steps=8, dt=0.2, D=D, seed=0, gauge_iters=5, chi_b=32,
                       obs="X") for D in (4, 8)]                       # clean TFIM control (uniform χ)
        specs += [dict(family="brickwork", Lx=6, Ly=6, depth=2, D=D, seed=0, gauge_iters=5, chi_b=32)
                  for D in (4, 8)]
        specs += [dict(family="brickwork", Lx=4, Ly=4, depth=3, D=8, seed=s, gauge_iters=5, chi_b=32)
                  for s in (1, 2)]
    return specs


def replay_specs(machine=None):
    rs = []
    for obs_mode in ("bmps", "bp"):
        for P in (2, 4, 8, 16):
            for pl in PLACERS:
                rs.append(dict(placer=pl, P=P, dynamic=None, seed=0, obs_mode=obs_mode, machine=machine))
                if pl != "ilp":
                    rs.append(dict(placer=pl, P=P, dynamic=dict(every=1), seed=0, obs_mode=obs_mode, machine=machine))
            rs.append(dict(placer="kahypar_km1_chi", P=P, dynamic=None, oracle=True, seed=0, obs_mode=obs_mode,
                           machine=machine))
    return rs


def applicable(tr, rs):
    if rs["placer"] == "ilp":
        return tr.n <= 16 and rs["P"] <= 4
    return True


def print_table(results):
    from collections import defaultdict
    g = defaultdict(list)
    for r in results:
        if "error" in r:
            print("ERROR", r["trace"], r.get("placer"), r.get("P"), r["error"])
            continue
        g[(r["workload"], r["D"], r["P"], r["obs_mode"])].append(r)
    for key in sorted(g):
        rs = sorted(g[key], key=lambda r: r["t_wall"])
        fid = rs[0].get('fidelity')
        print(f"\n=== {key[0]} D={key[1]} P={key[2]} obs={key[3]}  "
              f"(max χ={rs[0]['max_bond']}, Σdisc={rs[0]['disc_total']:.2e}, "
              f"fidelity={'n/a' if fid is None else f'{fid:.4f}'}, mae_z={rs[0]['mae_z']})")
        print(f"{'placer':26s} {'inter MB':>9s} {'intra MB':>9s} {'msgs':>7s} {'t_wall':>9s} {'comm%':>6s} "
              f"{'imb':>6s} {'peakMB':>8s} {'cut':>5s} {'repart':>7s} {'search s':>9s}")
        for r in rs:
            nm = r["placer"] + ("+dyn" if r["dynamic"] else "") + ("+oracle" if r.get("oracle") else "")
            print(f"{nm:26s} {r['bytes_inter']/1e6:9.3f} {r['bytes_intra']/1e6:9.3f} {r['msgs_inter']:7d} "
                  f"{r['t_wall']:9.4f} {100*r['comm_frac']:6.1f} {r['imb_mean']:6.2f} {r['peak_bytes']/1e6:8.2f} "
                  f"{r['cut_bonds_mean']:5.1f} {r['n_repart']:3d}/{r.get('n_tried', 0):<3d} {r['t_search']:9.4f}")


def main():
    quick = "--quick" in sys.argv
    machine = None
    if "--machine" in sys.argv:                       # e.g. --machine results_perlmutter/machine_calibrated.json
        from tnpart.costmodel import Machine
        machine = Machine.from_json(sys.argv[sys.argv.index("--machine") + 1])
        print("cost model:", machine, flush=True)
    workers = 8
    os.makedirs(RES, exist_ok=True)
    t0 = time.perf_counter()
    paths = run_traces(trace_specs(quick), os.path.join(RES, "traces"), workers=workers)
    print(f"traces done in {time.perf_counter() - t0:.1f}s", flush=True)
    traces = [load_trace(p) for p in paths]
    for tr in traces:
        st = tr.chi_stats_by_layer()
        print(f"  {tr.family} {tr.Lx}x{tr.Ly} D={tr.D}: χ(min/med/max) per layer "
              + " ".join(f"{s[1]}/{s[2]:.0f}/{s[4]}" for s in st)
              + f"  F={tr.fidelity} disc={tr.disc_total:.2e} mae_z={tr.mae_z} t={ {k: round(v, 1) for k, v in tr.timings.items()} }", flush=True)
    fig_chi_heterogeneity(traces, os.path.join(RES, "fig_chi_heterogeneity.png"))
    t1 = time.perf_counter()
    results = run_replays(paths, replay_specs(machine), os.path.join(RES, "results.jsonl"), workers=workers,
                          filter_fn=applicable)
    print(f"replays done in {time.perf_counter() - t1:.1f}s", flush=True)
    print_table(results)
    fig_f4(results, os.path.join(RES, "fig_F4_comm_vs_imbalance.png"))
    fig_f4(results, os.path.join(RES, "fig_F4_comm_vs_imbalance_bmps.png"), obs_mode="bmps")
    fig_f3(results, os.path.join(RES, "fig_F3_trunc_vs_comm.png"))
    fig_f2(results, os.path.join(RES, "fig_F2_D_vs_fidelity_vs_time.png"))
    fig_f6(results, os.path.join(RES, "fig_F6_search_vs_wall.png"))
    print(f"figures in {RES}; total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
