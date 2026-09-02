"""Sweep driver: traces in a fork pool (physics, expensive) then replays in a
fork pool (placement, cheap). Traces are cached as pickles; results go to JSONL.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

from .circuits import brickwork_2d, ghz_2d, trotter_tfim_2d
from .peps_sim import evolve_and_trace
from .vsim import replay

TRACE_KEYS = ("family", "Lx", "Ly", "depth", "n_steps", "dt", "J", "h", "seed", "two_qubit",
              "disorder", "obs", "D", "cutoff", "gauge_iters", "chi_b")


def make_circuit(spec):
    fam, Lx, Ly = spec["family"], spec["Lx"], spec["Ly"]
    if fam == "brickwork":
        return brickwork_2d(Lx, Ly, spec["depth"], seed=spec.get("seed", 0),
                            two_qubit=spec.get("two_qubit", "haar"))
    if fam == "tfim":
        return trotter_tfim_2d(Lx, Ly, spec["n_steps"], dt=spec.get("dt", 0.1),
                               J=spec.get("J", 1.0), h=spec.get("h", 1.0),
                               disorder=spec.get("disorder", 0.0), seed=spec.get("seed", 0))
    if fam == "ghz":
        return ghz_2d(Lx, Ly)
    raise ValueError(fam)


def trace_key(spec):
    sub = {k: spec[k] for k in TRACE_KEYS if k in spec}
    h = hashlib.sha1(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:8]
    dis = f"_dis{spec['disorder']}" if spec.get("disorder") else ""
    return f"{spec['family']}_{spec['Lx']}x{spec['Ly']}{dis}_D{spec['D']}_s{spec.get('seed', 0)}_{h}"


def _trace_job(args):
    spec, out_dir = args
    path = os.path.join(out_dir, trace_key(spec) + ".pkl")
    if os.path.exists(path):
        return path
    t0 = time.perf_counter()
    circ = make_circuit(spec)
    tr = evolve_and_trace(circ, spec["D"], cutoff=spec.get("cutoff", 1e-10),
                          gauge_iters=spec.get("gauge_iters", 5), chi_b=spec.get("chi_b"),
                          exact=spec.get("exact", "auto"),
                          obs=spec.get("obs", "X" if spec["family"] == "tfim" else "Z"))
    with open(path, "wb") as f:
        pickle.dump(tr, f)
    print(f"[trace] {trace_key(spec)}: max χ={tr.max_bond} disc={tr.disc_total:.3e} "
          f"mae_z={tr.mae_z} t={time.perf_counter() - t0:.1f}s "
          f"(evolve {tr.timings.get('evolve', 0):.1f}s, bmps {tr.timings.get('bmps_z', 0):.1f}s)", flush=True)
    return path


def run_traces(specs, out_dir, workers=8):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, out_dir) for s in specs]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(jobs))) as pool:
        paths = list(pool.map(_trace_job, jobs, chunksize=1))
    return paths


def load_trace(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- replays

_TRACES = {}


def _replay_job(args):
    key, rspec = args
    tr = _TRACES[key]
    try:
        out = replay(tr, **rspec)
    except Exception as e:  # keep the sweep alive, record the failure
        out = dict(error=f"{type(e).__name__}: {e}", trace=key, **{k: v for k, v in rspec.items() if k != "dynamic"},
                   dynamic=bool(rspec.get("dynamic")))
    out["trace"] = os.path.basename(key)
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def run_replays(trace_paths, replay_specs, out_path, workers=8, filter_fn=None, verbose=True):
    global _TRACES
    _TRACES = {p: load_trace(p) for p in trace_paths}     # built in the parent → inherited by fork
    jobs = [(p, rs) for p in trace_paths for rs in replay_specs
            if filter_fn is None or filter_fn(_TRACES[p], rs)]
    ctx = mp.get_context("fork")
    results, t0 = [], time.perf_counter()
    with ctx.Pool(min(workers, len(jobs))) as pool:
        for i, res in enumerate(pool.imap_unordered(_replay_job, jobs, chunksize=4), 1):
            results.append(res)
            if verbose and (i % 50 == 0 or i == len(jobs)):
                print(f"[replay] {i}/{len(jobs)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=_json_default) + "\n")
    return results


def load_results(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
