"""Distributed SU-PEPS run (mpi4py). One rank per GPU/core; placement computed on
rank 0 from the χ=1 circuit graph with any tnpart placer and broadcast.

  laptop:     mpirun -n 4 ../.venv/bin/python run_mpi.py --family tfim --L 6 --n-steps 10 --dt 0.1 \\
                  --disorder 1.0 --cutoff 1e-4 --D 16 --placer tiles --ranks-per-node 2 --validate --tag l4
  Perlmutter: srun -n 8 --ntasks-per-node=4 --gpus-per-task=1 python run_mpi.py ... --gpu --ranks-per-node 4

Writes results/mpi_<tag>.json (timers per rank, χ stats, discarded weight, optional
exact fidelity). ``--validate`` gathers the state to rank 0 (n ≤ 20 only) and also
dumps the Λ_e so two runs can be compared bit-for-bit (placement independence).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from tnpart.mpi_backend import DistributedPEPS  # noqa: E402
from tnpart.partition import evaluate, run_placer  # noqa: E402
from tnpart.sweep import make_circuit  # noqa: E402
from tnpart.tngraph import TNGraph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="tfim", choices=["tfim", "brickwork", "ghz"])
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--Ly", type=int, default=None)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--disorder", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--cutoff", type=float, default=1e-4)
    ap.add_argument("--gauge-iters", type=int, default=5)
    ap.add_argument("--placer", default="tiles")
    ap.add_argument("--ranks-per-node", type=int, default=4)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, P = comm.Get_rank(), comm.Get_size()
    Ly = args.Ly or args.L
    spec = dict(family=args.family, Lx=args.L, Ly=Ly, depth=args.depth, n_steps=args.n_steps, dt=args.dt,
                disorder=args.disorder, seed=args.seed)
    circ = make_circuit(spec)                       # deterministic from the spec on every rank

    # ---- placement on rank 0 from the χ=1 circuit graph (static, brief §2.4 baseline regime)
    if rank == 0:
        from tnpart.circuits import grid_bonds, grid_sites
        sites, bonds = grid_sites(args.L, Ly), grid_bonds(args.L, Ly)
        traffic = circ.bond_traffic()
        g0 = TNGraph(args.L, Ly, 2, sites, bonds, np.full(len(sites), 2), np.ones(len(bonds), dtype=int),
                     np.array([traffic[b] for b in bonds]), layers_remaining=circ.depth)
        pi, t_search = run_placer(args.placer, g0, P, seed=0)
        ev = evaluate(g0, pi, P=P)
        print(f"run_mpi: {args.family} {args.L}x{Ly} D={args.D} P={P} ranks/node={args.ranks_per_node} "
              f"placer={args.placer} cut={ev['cut_bonds']} search={t_search:.3f}s gpu={args.gpu}", flush=True)
    else:
        pi = None
    pi = comm.bcast(pi, root=0)

    if args.gpu:
        import cupy as cp
        vis = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
        cp.cuda.Device(0 if len(vis) == 1 else rank % args.ranks_per_node).use()   # --gpus-per-task=1 → one visible GPU
        xp = cp
    else:
        xp = np

    sim = DistributedPEPS(circ, args.D, pi, comm, cutoff=args.cutoff, gauge_iters=args.gauge_iters, xp=xp)
    comm.Barrier()
    t0 = time.perf_counter()
    sim.run(progress=args.progress)
    comm.Barrier()
    wall = time.perf_counter() - t0
    rep = sim.report()

    if args.validate:
        G_all, lam = sim.gather_state()
        if rank == 0:
            rep["lam_checksum"] = float(sum(float(np.sum(l)) for l in lam))
            rep["lam"] = [l.tolist() for l in lam]
            rep["chi"] = sim.chi.tolist()
            if circ.n_qubits <= 20:
                from tnpart.peps_sim import exact_statevector
                psi_p = sim.dense_state(G_all)
                psi_e = exact_statevector(circ).ravel()
                rep["fidelity"] = float(abs(np.vdot(psi_e, psi_p)) ** 2)

    if rank == 0:
        rep.update(spec=spec, placer=args.placer, pi=[int(x) for x in pi], wall=wall,
                   ranks_per_node=args.ranks_per_node, n_nodes=-(-P // args.ranks_per_node), gpu=args.gpu,
                   cut_bonds=int(ev["cut_bonds"]), omp_threads=os.environ.get("OMP_NUM_THREADS"),
                   host=__import__("socket").gethostname())
        os.makedirs(args.out, exist_ok=True)
        tag = args.tag or f"{args.family}{args.L}_D{args.D}_P{P}_{args.placer}"
        path = os.path.join(args.out, f"mpi_{tag}.json")
        json.dump(rep, open(path, "w"), indent=1)
        print(f"wall {wall:.3f}s | max over ranks: gate comp {rep['gate_comp_max']:.3f} comm {rep['gate_comm_max']:.3f} "
              f"| gauge comp {rep['gauge_comp_max']:.3f} comm {rep['gauge_comm_max']:.3f} | sync {rep['sync_max']:.3f} "
              f"| msgs {rep['n_msgs_max']} bytes {rep['bytes_max']/1e6:.2f} MB | max χ {rep['max_chi']} "
              f"disc {rep['disc_total']:.3e}" + (f" | fidelity {rep['fidelity']:.6f}" if "fidelity" in rep else ""),
              flush=True)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
