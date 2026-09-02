#!/bin/bash
#SBATCH -J tnp-phase0
#SBATCH -C gpu
#SBATCH -q debug                 # 2 GPU nodes, <=30 min, starts fast
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1 --gpu-bind=none
#SBATCH -c 32
#SBATCH -t 00:15:00
#SBATCH -o logs/phase0_%j.out
#SBATCH -e logs/phase0_%j.err
# Phase-0 microbenchmarks (brief §7.1): pins p_gemm, c_svd/c_qr and NVLink-vs-Slingshot
# latency/bandwidth for the cost model. 8 ranks = 4 GPUs x 2 nodes; ranks 0-3 on node 0.
# Usage (from ~/tn_partitioning):  sbatch -A <project> perlmutter/phase0_micro.sbatch
# Output: results/phase0_<jobid>.json  -> pull.sh -> perlmutter/plot_phase0.py (laptop)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
module load conda cudatoolkit craype-accel-nvidia80
[ -f "$HOME/.tnp_env" ] && source "$HOME/.tnp_env"
conda activate "${TNP_ENV:-$HOME/envs/tnp}"
export MPICH_GPU_SUPPORT_ENABLED=1
# GPU-aware sends between tasks on one node use CUDA IPC, which needs every peer GPU visible: hence
# --gpu-bind=none on every srun (device = rank % 4 in the code). Fallback if IPC still fails: GPU_IPC=0.
[ "${GPU_IPC:-1}" = "0" ] && export MPICH_GPU_IPC_ENABLED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SLURM_CPU_BIND=cores
echo "job $SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# If GPU-aware sends fail ("MPICH ... cudaErrorInvalidValue"), retry with --gpu-bind=none.
srun -n 8 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none \
  python -u perlmutter/microbench.py --gpu --ranks-per-node 4 --tag "$SLURM_JOB_ID"
echo "done $(date)"
