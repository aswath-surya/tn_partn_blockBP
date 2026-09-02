# Running tn_partitioning on NERSC Perlmutter (multi-node MPI + GPU, from a laptop)

Self-contained recipe. Replace `<user>` (NERSC username) and `<PROJECT>` (allocation, e.g. `m1234`, from
https://iris.nersc.gov). Local repo: `~/Desktop/workspaces/workspace1/VSCodeFiles_Thinkpad/tn_partitioning`
(`<REPO>` below); remote dir: `~/tn_partitioning`. Sources: docs.nersc.gov (connect/mfa, python/nersc-python,
python/parallel-python, mpi/cray-mpich, python/using-python-perlmutter, jobs/policy), read 2026-08-26.

## 0. One-time: passwordless ssh via sshproxy (24 h key)
Install sshproxy from https://portal.nersc.gov/cfs/mfa/ (macOS installer). Get a key (prompt `Password+OTP:` —
NERSC password immediately followed by the 6-digit OTP, no space):
```bash
# writes ~/.ssh/nersc, nersc-cert.pub (valid 24 h)
sshproxy -u <user>
```
Make plain ssh/scp/rsync use it — `~/.ssh/config` is a file (create if missing):
```bash
cat >> ~/.ssh/config <<'EOT'
Host perlmutter*.nersc.gov saul*.nersc.gov dtn*.nersc.gov
    User <user>
    IdentityFile ~/.ssh/nersc
    IdentitiesOnly yes
EOT
chmod 600 ~/.ssh/config
```
Verify (must print a login node name with no prompt) and check expiry:
```bash
ssh -o BatchMode=yes perlmutter.nersc.gov hostname
# re-run sshproxy when expired
ssh-keygen -L -f ~/.ssh/nersc-cert.pub | grep Valid
```
Gotcha: if ssh silently asks for a password again, the key expired → `sshproxy -u <user>`.
zsh gotcha: interactive zsh does not treat `#` as a comment, so never paste a trailing `# …` (e.g. `# ~10 min` →
`zsh: not enough directory stack entries`). One-time fix: `echo 'setopt interactive_comments' >> ~/.zshrc`.

## 1. Upload the repo (laptop → Perlmutter)
```bash
cd <REPO> && bash perlmutter/push.sh
# = rsync -av --exclude '.venv' --exclude '__pycache__' --exclude results --exclude results_perlmutter \
#     --exclude logs --exclude refs --exclude '*.pkl'  <REPO>/  perlmutter.nersc.gov:~/tn_partitioning/
```
Run from the laptop (from inside ssh it fails with "change_dir … No such file"). Re-run after every local edit.

## 2. One-time environment on Perlmutter (login node, ~10 min)
```bash
ssh perlmutter.nersc.gov 'bash -l ~/tn_partitioning/perlmutter/setup_perlmutter.sh'
# must print "deps ok" + Cray MPICH version
ssh perlmutter.nersc.gov 'bash -l ~/tn_partitioning/perlmutter/verify.sh'
```
What `setup_perlmutter.sh` does, and why (all from the NERSC docs):
- `module load conda cudatoolkit craype-accel-nvidia80` → **conda** env (NERSC's recommendation; venv is not
  documented) at `$HOME/envs/tnp`, pinned `requirements.txt`. NERSC suggests `/global/common/software/<PROJECT>/`
  for parallel apps, but its quota is small and overflowed on the first try (`Disk quota exceeded` mid-`conda`);
  use it only after a quota ticket: `TNP_COMMON=<PROJECT> bash -l …`. Check usage with `showquota`.
- **mpi4py built through the Cray compiler wrapper with the GPU transport layer linked:**
  `MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py`, run *after*
  the accel modules are loaded. Docs: "Do NOT conda/pip install mpi4py at NERSC" — a plain install looks fine
  and silently does not work. Linking GTL now means one build serves CPU runs and GPU-aware (CuPy) runs.
- `pip install cupy-cuda12x` matched to the `cudatoolkit` module's major version.
- Writes `~/.tnp_env` (the env prefix) which every batch script sources.
Optional GPU sanity check (10-min interactive allocation; only before a `GPU=1` run):
```bash
ssh perlmutter.nersc.gov
salloc -C gpu -q interactive -N 1 --ntasks-per-node=4 -c 32 --gpus-per-task=1 -t 00:10:00 -A <PROJECT>
module load conda cudatoolkit craype-accel-nvidia80; source ~/.tnp_env; conda activate $TNP_ENV
export MPICH_GPU_SUPPORT_ENABLED=1
srun -n 4 python -c 'import cupy, mpi4py.MPI as M; print(M.COMM_WORLD.rank, cupy.cuda.runtime.getDeviceProperties(0)["name"])'
exit; exit
```

## 3. Machine facts that matter (GPU nodes — this repo uses them, not the CPU nodes)
- GPU node = 4× A100 40 GB (NVLink between them) + 1× AMD EPYC 7763 (64 physical cores; Slurm shows 128
  hyperthreads), 256 GB host RAM, Slingshot-11 (4 NICs). `-c` per task = `2*floor(64/ntasks_per_node)` → **`-c 32` for
  4 tasks/node**, and `export SLURM_CPU_BIND=cores`.
- **One MPI rank per GPU, launched by `srun`** (mpi4py "must be launched via srun"; never `mpirun`, never
  `python script.py` alone). `--gpus-per-task=1` makes exactly one GPU visible per task (`CUDA_VISIBLE_DEVICES`
  has one entry → device index 0); the code handles that.
- GPU-aware MPI: `export MPICH_GPU_SUPPORT_ENABLED=1` at run time — only valid if mpi4py was built with GTL
  linked (step 2); otherwise `MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked`.
  `cudatoolkit` must be loaded at run time too ("otherwise the executable may hang").
- Queues (GPU): `-q debug` ≤ 8 nodes, ≤ 30 min, ≤ 2 running, starts in minutes; `-q interactive` ≤ 4 nodes, ≤ 4 h;
  `-q regular` unlimited nodes, ≤ 48 h, longer wait. `-A <PROJECT>` needed only if you are in several projects.
- BLAS threads: with 4 ranks/node use `OMP_NUM_THREADS=16` (= 64 cores / 4), and keep it **identical across the
  1-node and 2-node layouts** (Phase 1 compares them; different thread counts would confound the result).
- Several `srun` steps with *different node layouts* run back-to-back inside one allocation
  (`srun -N 1 -n 4 …`, then `srun -N 2 --ntasks-per-node=2 …`); that is how the 1×4-vs-2×2 comparison uses the
  same hardware in the same job.

## 4. The batch script (`perlmutter/pareto_job.sbatch`) — one submission does everything
```bash
#SBATCH -J tnp-pareto
#SBATCH -C gpu
#SBATCH -q regular              # debug is capped at 30 min; this needs ~45-60 min
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -o logs/pareto_%j.out
#SBATCH -e logs/pareto_%j.err
set -uo pipefail; cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
module load conda cudatoolkit craype-accel-nvidia80; source ~/.tnp_env; conda activate "$TNP_ENV"
export MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 SLURM_CPU_BIND=cores MPLBACKEND=Agg
# step 0  Phase 0 microbenchmarks (8 ranks = 2x4): GEMM/SVD/QR/eigh rates, NVLink-vs-Slingshot pingpong 8 B-1 GB, all-reduce
srun -N 2 -n 8 --ntasks-per-node=4 --gpus-per-task=1 python -u perlmutter/microbench.py --ranks-per-node 4 --tag cpu_$SLURM_JOB_ID
srun -N 2 -n 8 --ntasks-per-node=4 --gpus-per-task=1 python -u perlmutter/microbench.py --gpu --ranks-per-node 4 --tag gpu_$SLURM_JOB_ID
# step 1  Phase 1: identical 4-rank SU-PEPS runs as 1x4 (cut bonds on NVLink) and 2x2 (cut crosses Slingshot)
#         for each L:D:steps config, placer and repeat:   run_case 1 4 …   then   run_case 2 2 …
#         run_case = srun -N $N -n $((N*RPN)) --ntasks-per-node=$RPN --gpus-per-task=1 python -u run_mpi.py \
#                    --family tfim --L $L --n-steps $S --dt 0.1 --disorder 1.0 --cutoff 1e-4 --D $D \
#                    --placer $PL --ranks-per-node $RPN [--gpu] --tag p1_N${N}x${RPN}_L${L}_D${D}_${PL}_r${R}_$SLURM_JOB_ID
# step 2  scaling point: the same on 8 ranks (2x4)
```
Every step writes one JSON to `results/` (`phase0_*.json`, `mpi_p1_*.json`) and is wrapped in `|| echo FAILED …`
so one failure never kills the job. Knobs: `CONFIGS="8:16:10 8:32:10 12:16:10"` (L:D:steps), `PLACERS="tiles rows
kahypar_km1_chi random"`, `REPS=3`, `GPU=0|1` (CuPy tensors + GPU-aware MPI; default CPU). `run_mpi.py` records
per-rank timers (compute vs time-in-MPI per phase), messages, bytes, cut bonds, χ stats, host, threads.

## 5. Submit, watch, fetch — all from the laptop
```bash
ssh perlmutter.nersc.gov 'cd ~/tn_partitioning && sbatch -A <PROJECT> perlmutter/pareto_job.sbatch && squeue -u $USER'
# PD queued, R running, gone = finished
ssh perlmutter.nersc.gov 'squeue -u $USER'
# per-srun-step state; FAILED/TIMEOUT show here
ssh perlmutter.nersc.gov 'sacct -j <jobid> --format=JobID%14,State,Elapsed,NodeList%22'
# progress / FAILED lines
ssh perlmutter.nersc.gov 'tail -40 ~/tn_partitioning/logs/pareto_<jobid>.out'
# runs finished so far
ssh perlmutter.nersc.gov 'grep -c "^wrote" ~/tn_partitioning/logs/pareto_<jobid>.out'
```
Keep `$USER` inside single quotes so it expands on Perlmutter. Quick variants on the debug queue (≤ 30 min):
`sbatch -A <PROJECT> perlmutter/phase0_micro.sbatch` (15 min) and `perlmutter/phase1_1v2.sbatch` submitted twice
(`sbatch …` and `sbatch -N 2 --ntasks-per-node=2 …`). Knobs go in front: `CONFIGS="8:16:10" REPS=5 GPU=1 sbatch …`.

Read-only pull loop (laptop; leave running; Ctrl-C when done; never modifies the remote):
```bash
bash perlmutter/watch_pull.sh
# = every 30 s: rsync results/*.json,*.png -> <REPO>/results_perlmutter/ ; logs/ -> <REPO>/logs_perlmutter/
```
One-shot: `bash perlmutter/pull.sh`.

## 6. Post-process locally (from results_perlmutter/)
```bash
# -> results_perlmutter/machine_calibrated.json (p_gemm, c_svd, c_qr, bw/lat per tier) + fig_phase0_*.png
../.venv/bin/python perlmutter/plot_phase0.py
# -> table + fig_phase1.png: medians [min,max], slowest-rank compute vs MPI, per-placer 2x2/1x4 ratio
../.venv/bin/python perlmutter/plot_phase1.py
# frontiers F2/F3/F4/F6 under the calibrated cost model (~10 min; --quick ~2 min)
../.venv/bin/python -u run_test.py --machine results_perlmutter/machine_calibrated.json
```
Frontier validation = measured Phase-1 placer ordering / comm fractions vs the model's `t_wall` / `comm_frac` for
the same (L, D, P, placer) in `results/results.jsonl`.

## 7. Failure modes, with fixes
| symptom | cause | fix |
|---|---|---|
| `Permission denied` / password prompt on ssh | sshproxy key missing or expired | `sshproxy -u <user>`; `ls ~/.ssh/nersc*` |
| `rsync: change_dir … No such file` | ran the laptop-side rsync while ssh'd in | run it in a local terminal |
| `module: command not found` in a script run over ssh | non-login shell, Lmod not initialised | scripts source `/usr/share/lmod/lmod/init/bash`; or call with `bash -l` |
| `conda: command not found` in the job | `module load conda` missing before `conda activate` | it is in the sbatch; check the log's first lines |
| `ImportError: mpi4py` or "mpi4py programs just won't work" | pip/conda mpi4py without the Cray wrapper | re-run step 2 (`MPICC="cc -shared" … --no-binary=mpi4py`) |
| `MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked` | mpi4py built without `cudatoolkit craype-accel-nvidia80` loaded | re-run step 2; or drop `MPICH_GPU_SUPPORT_ENABLED=1` for CPU-only runs |
| job hangs at the first `srun` | `cudatoolkit` not loaded at run time (docs' warning) | it is in the sbatch; check `module list` in the log |
| CuPy step `FAILED`, CPU steps fine | cupy wheel ≠ cudatoolkit major version | `pip install cupy-cuda12x` (or 11x) matching `nvcc --version`; CPU results already suffice for calibration |
| `srun: error: … gpus-per-task` on CPU-only nodes | script assumes GPU nodes | keep `-C gpu`; this repo's steps all run on GPU nodes |
| `[Errno 122] Disk quota exceeded` under `/global/common/software/…` during setup | project software quota is small | `rm -rf` that env dir; use the default `$HOME/envs/tnp` (re-run setup without `TNP_COMMON`); `showquota` |
| `sbatch: error: … account` | no default allocation | `sbatch -A <PROJECT> …` |
| job pending long in `regular` | queue load | run the debug-queue variants meanwhile (≤ 30 min, ≤ 8 nodes, 2 running max) |
| wall times noisy between repeats | shared network (brief §7) | `REPS=5`; plots use medians with min/max, never means |
