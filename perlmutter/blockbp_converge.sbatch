#!/bin/bash
#SBATCH -J tnp-blockbp
#SBATCH -C gpu
#SBATCH -q debug                 # QUICK default: 8 nodes (debug cap), ~5 min. FULL: see below
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -o logs/blockbp_%j.out
#SBATCH -e logs/blockbp_%j.err
# BlockBP convergence experiment on a random non-brickwork 1+1D circuit, one spacetime
# tile per rank, MPS messages over MPI, a library of convergence strategies per run.
#   QUICK (default, ~5 min):  sbatch -A <project> perlmutter/blockbp_converge.sbatch
#   FULL  (10 nodes, 1 h):    FULL=1 sbatch -A <project> -q regular -N 10 -t 01:00:00 perlmutter/blockbp_converge.sbatch
# Steps: (1) small instance with the exact amplitude, all ranks; (2) amplitude network at scale;
#        (3) the same circuit's norm network (control: must converge to Z=1).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
module load conda cudatoolkit craype-accel-nvidia80
[ -f "$HOME/.tnp_env" ] && source "$HOME/.tnp_env"
conda activate "${TNP_ENV:-$HOME/envs/tnp}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}
export SLURM_CPU_BIND=cores MPLBACKEND=Agg
NT=$(( SLURM_JOB_NUM_NODES * ${SLURM_NTASKS_PER_NODE:-4} ))
J=$SLURM_JOB_ID
if [ "${FULL:-0}" = "1" ]; then
  BQ=${BQ:-10}; BT=${BT:-8}; PRESET=${PRESET:-full}; BUDGET=${BUDGET:-90}; SWEEPS=${SWEEPS:-80}; CHI=${CHI:-8}
else
  BQ=${BQ:-8};  BT=${BT:-6}; PRESET=${PRESET:-basic}; BUDGET=${BUDGET:-15}; SWEEPS=${SWEEPS:-40}; CHI=${CHI:-8}
fi
NQ=$(( NT / 4 )); N=$(( NQ * BQ )); DEPTH=$(( 4 * BT - 2 ))     # NQ q-blocks × 4 t-blocks = NT blocks
echo "job $J nodes=$SLURM_JOB_NUM_NODES ranks=$NT full=${FULL:-0} n=$N depth=$DEPTH tiles ${BQ}x${BT} chi_m=$CHI preset=$PRESET budget=${BUDGET}s start $(date)"
S="srun -n $NT --ntasks-per-node=${SLURM_NTASKS_PER_NODE:-4} --gpus-per-task=1 --gpu-bind=none"

echo "== step 1: small instance with exact amplitude (n=16, depth=14, 4x4 tiles = 16 blocks) $(date)"
$S python -u run_blockbp_mpi.py --n 16 --depth 14 --bq 4 --bt 4 --chi-m "$CHI" --preset "$PRESET" \
   --budget "$BUDGET" --max-sweeps "$SWEEPS" --exact --tag "small_$J" || echo "STEP1 FAILED"

echo "== step 2: amplitude network at scale $(date)"
$S python -u run_blockbp_mpi.py --n "$N" --depth "$DEPTH" --bq "$BQ" --bt "$BT" --chi-m "$CHI" --preset "$PRESET" \
   --budget "$BUDGET" --max-sweeps "$SWEEPS" --tag "amp_$J" || echo "STEP2 FAILED"

echo "== step 3: norm network control $(date)"
$S python -u run_blockbp_mpi.py --n "$N" --depth "$DEPTH" --bq "$BQ" --bt "$BT" --chi-m "$CHI" --preset basic \
   --network norm --budget "$BUDGET" --max-sweeps "$SWEEPS" --exact --tag "norm_$J" || echo "STEP3 FAILED"
echo "all done $(date)"
