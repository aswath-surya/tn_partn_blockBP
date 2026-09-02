#!/bin/bash
#SBATCH -J tnp-bbp-pareto
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 00:25:00
#SBATCH -o logs/bbp_pareto_%j.out
#SBATCH -e logs/bbp_pareto_%j.err
# BlockBP Pareto sweeps in ≤25 min on the debug queue (8 nodes × 8 ranks = 64 tiles, CPU compute).
#   sbatch -A <project> perlmutter/blockbp_pareto.sbatch
# A  exact amplitude, n=16:      χ_m × tile-size grid            -> error vs χ_m / tile / time
# B  norm network, n=16·bq:      χ_m × bq (exact Z = 1)          -> error vs χ_m / tile / time at 64 tiles
# C  amplitude network at scale: bq × χ_m, 5 strategies          -> Δ, schedule agreement, time
# Every run writes results/blockbp_<tag>.json; pull + perlmutter/plot_blockbp_pareto.py
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
module load conda cudatoolkit craype-accel-nvidia80
[ -f "$HOME/.tnp_env" ] && source "$HOME/.tnp_env"
conda activate "${TNP_ENV:-$HOME/envs/tnp}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 SLURM_CPU_BIND=cores MPLBACKEND=Agg
J=$SLURM_JOB_ID; NT=$(( SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE ))
echo "job $J nodes=$SLURM_JOB_NUM_NODES ranks=$NT start $(date)"
run() { # run NRANKS ARGS...
  local nr=$1; shift
  srun -n "$nr" --ntasks-per-node=$(( nr < SLURM_NTASKS_PER_NODE ? nr : SLURM_NTASKS_PER_NODE )) \
       python -u run_blockbp_mpi.py "$@" || echo "FAILED: $*"
}
echo "== A: exact amplitude n=16 depth=14, χ_m × tiles $(date)"
for chi in 2 4 8 16; do
  run 16 --n 16 --depth 14 --bq 4 --bt 4 --chi-m $chi --preset fast --budget 8 --max-sweeps 30 --exact --tag "A_q4t4_chi${chi}_$J"
  run 8  --n 16 --depth 14 --bq 8 --bt 4 --chi-m $chi --preset fast --budget 8 --max-sweeps 30 --exact --tag "A_q8t4_chi${chi}_$J"
  run 8  --n 16 --depth 14 --bq 4 --bt 8 --chi-m $chi --preset fast --budget 8 --max-sweeps 30 --exact --tag "A_q4t8_chi${chi}_$J"
done
echo "== B: norm network at 64 tiles (exact Z=1), χ_m × bq $(date)"
for bq in 6 8; do for chi in 2 4 8; do
  run $NT --n $(( 16 * bq )) --depth 22 --bq $bq --bt 6 --chi-m $chi --preset fast --network norm --budget 10 --max-sweeps 30 --exact --tag "B_q${bq}t6_chi${chi}_$J"
done; done
echo "== C: amplitude network at 64 tiles, strategies $(date)"
for bq in 8 12; do for chi in 4 8; do
  run $NT --n $(( 16 * bq )) --depth 22 --bq $bq --bt 6 --chi-m $chi --preset agree --budget 10 --max-sweeps 30 --tag "C_q${bq}t6_chi${chi}_$J"
done; done
echo "all done $(date)"
