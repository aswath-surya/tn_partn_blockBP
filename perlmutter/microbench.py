#!/usr/bin/env python
"""Phase-0 microbenchmarks (brief §7.1) — standalone, no tnpart import.

Pins the cost-model constants: p_gemm (achievable zgemm/cgemm), c_svd, c_qr, c_eigh
(time relative to GEMM-rate at equal FLOPs), and per-tier point-to-point latency /
bandwidth plus all-reduce bus bandwidth. Rank 0 writes results/phase0_<tag>.json.

  Perlmutter:  srun -n 8 --ntasks-per-node=4 --gpus-per-task=1 \\
                 python perlmutter/microbench.py --gpu --ranks-per-node 4 --tag $SLURM_JOB_ID
  Laptop:      mpirun -n 4 ../.venv/bin/python perlmutter/microbench.py --ranks-per-node 2 --quick

Tier definition: ranks p, q are "intra" iff p//ranks_per_node == q//ranks_per_node
(SLURM block distribution). Pairs measured: (0,1) intra, (0, ranks_per_node) inter.
"""
import argparse
import json
import os
import platform
import socket
import time

import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()


def median_time(fn, reps, sync=None, warmup=1):
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


# ------------------------------------------------------------------ local linear algebra

def bench_local(xp, sizes, reps, sync, dtype):
    """GEMM (n×n), SVD (n×n and 4n×n), QR (4n×n), eigh (n×n Hermitian).
    FLOPs reported are *real* FLOPs for complex data (8 per complex MAC)."""
    out = []
    cplx = 8 if np.dtype(dtype).kind == "c" else 2
    rng = np.random.default_rng(0)
    for n in sizes:
        rec = {"n": n, "dtype": np.dtype(dtype).name}
        A = xp.asarray((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(dtype))
        B = xp.asarray((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(dtype))
        T = xp.asarray((rng.standard_normal((4 * n, n)) + 1j * rng.standard_normal((4 * n, n))).astype(dtype))
        H = A @ A.conj().T
        t, _ = median_time(lambda: A @ B, reps, sync)
        rec["gemm_s"], rec["gemm_gflops"] = t, cplx / 2 * 2 * n ** 3 / t / 1e9      # 2n³ MACs
        t, _ = median_time(lambda: xp.linalg.svd(A, full_matrices=False), max(1, reps // 2), sync)
        rec["svd_sq_s"], rec["svd_sq_gflops_equiv"] = t, cplx / 2 * 4 * n ** 3 / t / 1e9  # 4·M·N·min
        t, _ = median_time(lambda: xp.linalg.svd(T, full_matrices=False), max(1, reps // 2), sync)
        rec["svd_tall_s"], rec["svd_tall_gflops_equiv"] = t, cplx / 2 * 4 * (4 * n) * n * n / t / 1e9
        t, _ = median_time(lambda: xp.linalg.qr(T), reps, sync)
        rec["qr_tall_s"], rec["qr_tall_gflops_equiv"] = t, cplx / 2 * 2 * (4 * n) * n * n / t / 1e9
        t, _ = median_time(lambda: xp.linalg.eigh(H), max(1, reps // 2), sync)
        rec["eigh_s"], rec["eigh_gflops_equiv"] = t, cplx / 2 * 4 * n ** 3 / t / 1e9
        out.append(rec)
        if RANK == 0:
            print(f"  n={n:5d} {rec['dtype']}: gemm {rec['gemm_gflops']:8.1f} GF/s | svd(sq) {rec['svd_sq_s']*1e3:8.2f} ms "
                  f"| svd(4n×n) {rec['svd_tall_s']*1e3:8.2f} ms | qr(4n×n) {rec['qr_tall_s']*1e3:8.2f} ms "
                  f"| eigh {rec['eigh_s']*1e3:8.2f} ms", flush=True)
    return out


# ------------------------------------------------------------------ network

def pingpong(xp, peer_a, peer_b, nbytes, reps):
    """Round-trip/2 between two ranks; other ranks wait at the barriers."""
    n = max(1, nbytes // 8)
    buf = xp.zeros(n, dtype=np.float64)
    COMM.Barrier()
    t = None
    if RANK in (peer_a, peer_b):
        other = peer_b if RANK == peer_a else peer_a
        for _ in range(2):                                         # warm-up
            if RANK == peer_a:
                COMM.Send(buf, dest=other); COMM.Recv(buf, source=other)
            else:
                COMM.Recv(buf, source=other); COMM.Send(buf, dest=other)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            if RANK == peer_a:
                COMM.Send(buf, dest=other); COMM.Recv(buf, source=other)
            else:
                COMM.Recv(buf, source=other); COMM.Send(buf, dest=other)
            ts.append((time.perf_counter() - t0) / 2)
        t = float(np.median(ts))
    COMM.Barrier()
    return COMM.bcast(t, root=peer_a)


def allreduce_bw(xp, nbytes, reps):
    n = max(1, nbytes // 8)
    buf, out = xp.ones(n, dtype=np.float64), xp.zeros(n, dtype=np.float64)
    COMM.Barrier()
    t, _ = median_time(lambda: COMM.Allreduce(buf, out, op=MPI.SUM), reps)
    t = COMM.allreduce(t, op=MPI.MAX)
    return t, 2 * (SIZE - 1) / SIZE * nbytes / t          # NCCL-style bus bandwidth


def alltoall_time(xp, nbytes_per_pair, reps):
    n = max(1, nbytes_per_pair // 8)
    s, r = xp.ones(n * SIZE, dtype=np.float64), xp.zeros(n * SIZE, dtype=np.float64)
    COMM.Barrier()
    t, _ = median_time(lambda: COMM.Alltoall(s, r), reps)
    return COMM.allreduce(t, op=MPI.MAX)


def bench_net(xp, ranks_per_node, sizes, reps):
    pairs = {}
    if ranks_per_node > 1 and SIZE > 1:
        pairs["intra"] = (0, 1)
    if SIZE > ranks_per_node:
        pairs["inter"] = (0, ranks_per_node)
    out = {"pairs": pairs, "pingpong": {}, "allreduce": [], "alltoall": []}
    for tier, (a, b) in pairs.items():
        rows = []
        for nb in sizes:
            t = pingpong(xp, a, b, nb, reps if nb < 2 ** 24 else max(3, reps // 5))
            rows.append({"bytes": nb, "t_s": t, "bw_GBs": nb / t / 1e9})
            if RANK == 0:
                print(f"  {tier} pingpong {nb:>11d} B: {t*1e6:9.2f} us  {nb/t/1e9:7.2f} GB/s", flush=True)
        out["pingpong"][tier] = rows
    if SIZE > 1:
        for nb in (2 ** 20, 2 ** 25, 2 ** 28):
            t, bw = allreduce_bw(xp, nb, 5)
            out["allreduce"].append({"bytes": nb, "t_s": t, "bus_bw_GBs": bw / 1e9})
            if RANK == 0:
                print(f"  allreduce {nb:>10d} B over {SIZE} ranks: {t*1e3:8.2f} ms  bus {bw/1e9:6.2f} GB/s", flush=True)
        for nb in (2 ** 12, 2 ** 20, 2 ** 24):
            t = alltoall_time(xp, nb, 5)
            out["alltoall"].append({"bytes_per_pair": nb, "t_s": t})
            if RANK == 0:
                print(f"  alltoall {nb:>9d} B/pair: {t*1e3:8.2f} ms", flush=True)
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="use cupy arrays (GPU-aware MPI for the network part)")
    ap.add_argument("--ranks-per-node", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="small sizes / few reps (laptop smoke test)")
    ap.add_argument("--max-n", type=int, default=2048, help="largest matrix size (4096 is fine on GPU, ~30 min on CPU)")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.gpu:
        import cupy as cp
        xp = cp
        vis = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
        dev = cp.cuda.Device(0 if len(vis) == 1 else RANK % args.ranks_per_node)   # --gpus-per-task=1 → one visible GPU
        dev.use()
        sync = cp.cuda.Stream.null.synchronize
        gpu_name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode() if isinstance(
            cp.cuda.runtime.getDeviceProperties(dev.id)["name"], bytes) else str(cp.cuda.runtime.getDeviceProperties(dev.id)["name"])
    else:
        xp, sync, gpu_name = np, None, None

    sizes = [128, 256, 512] if args.quick else [n for n in (256, 512, 1024, 2048, 4096) if n <= args.max_n]
    reps = 3 if args.quick else 7
    net_sizes = [8, 4096, 2 ** 16, 2 ** 20, 2 ** 23] if args.quick else [8, 4096, 2 ** 16, 2 ** 20, 2 ** 24, 2 ** 27] + ([] if args.gpu else [2 ** 30])

    info = dict(rank=RANK, host=socket.gethostname(), node=RANK // args.ranks_per_node, gpu=gpu_name,
                python=platform.python_version(), numpy=np.__version__,
                mpi=MPI.Get_library_version().strip()[:80],
                gpu_support=os.environ.get("MPICH_GPU_SUPPORT_ENABLED"))
    infos = COMM.gather(info, root=0)
    if RANK == 0:
        print(f"phase0 tag={args.tag} ranks={SIZE} ranks/node={args.ranks_per_node} gpu={args.gpu} "
              f"hosts={sorted({i['host'] for i in infos})}", flush=True)
        print("== local linear algebra (rank 0) ==", flush=True)
    local = {}
    for dtype in ((np.complex128, np.complex64) if not args.quick else (np.complex128,)):
        res = bench_local(xp, sizes, reps, sync, dtype)
        local[np.dtype(dtype).name] = res
    # one measurement per node is enough; keep rank 0 and the first rank of each other node
    locals_all = COMM.gather(local if RANK % args.ranks_per_node == 0 else None, root=0)
    if RANK == 0:
        print("== network ==", flush=True)
    net = bench_net(xp, args.ranks_per_node, net_sizes, reps * 10)
    if RANK == 0:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"phase0_{args.tag}.json")
        with open(path, "w") as f:
            json.dump(dict(tag=args.tag, size=SIZE, ranks_per_node=args.ranks_per_node, gpu=args.gpu,
                           ranks=infos, local_by_node=[x for x in locals_all if x], net=net), f, indent=1)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
