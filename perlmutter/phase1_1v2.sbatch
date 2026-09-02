#!/bin/bash
#SBATCH -J tnp-phase1
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -o logs/phase1_%j.out
#SBATCH -e logs/phase1_%j.err
# Phase 1 — THE decisive measurement (brief §7.2): identical 4-rank job on
#   (A) 1 node, 4 ranks  -> every cut bond crosses NVLink / shared memory
#   (B) 2 nodes, 2+2     -> the cut between rank pairs crosses Slingshot
# Submit the SAME script twice; the command line overrides the node layout:
#   sbatch -A <proj> perlmutter/phase1_1v2.sbatch                              # (A)
#   sbatch -A <proj> -N 2 --ntasks-per-node=2 perlmutter/phase1_1v2.sbatch     # (B)
# Per-rank compute is identical in both (same placement, same OMP threads), so the
# wall-time delta is the inter-node penalty. Repeats: medians (brief §7 variability).
# Env knobs: CONFIGS="L:D:steps ..."  PLACERS="tiles rows ..."  REPS=3  GPU=0|1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
module load conda cudatoolkit craype-accel-nvidia80
[ -f "$HOME/.tnp_env" ] && source "$HOME/.tnp_env"
conda activate "${TNP_ENV:-$HOME/envs/tnp}"
export MPICH_GPU_SUPPORT_ENABLED=1
# GPU-aware sends between tasks on one node use CUDA IPC, which needs every peer GPU visible: hence
# --gpu-bind=none on every srun (device = rank % 4 in the code). Fallback if IPC still fails: GPU_IPC=0.
[ "${GPU_IPC:-1}" = "0" ] && export MPICH_GPU_IPC_ENABLED=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${OMP_NUM_THREADS:-16}
export SLURM_CPU_BIND=cores
RPN=${SLURM_NTASKS_PER_NODE:-4}
NT=$(( SLURM_JOB_NUM_NODES * RPN ))
CONFIGS=${CONFIGS:-"8:16:10 8:32:10 12:16:10"}
PLACERS=${PLACERS:-"tiles rows kahypar_km1_chi random"}
REPS=${REPS:-3}
GPUFLAG=""; [ "${GPU:-0}" = "1" ] && GPUFLAG="--gpu"
echo "job $SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ranks/node=$RPN ranks=$NT omp=$OMP_NUM_THREADS gpu=${GPU:-0} start $(date)"
for cfg in $CONFIGS; do
  IFS=: read -r L D STEPS <<< "$cfg"
  for pl in $PLACERS; do
    for rep in $(seq 1 "$REPS"); do
      tag="p1_N${SLURM_JOB_NUM_NODES}x${RPN}_L${L}_D${D}_${pl}_r${rep}_${SLURM_JOB_ID}"
      srun -n "$NT" --ntasks-per-node="$RPN" ${GPU:+--gpus-per-task=1 --gpu-bind=none} \
        python -u run_mpi.py --family tfim --L "$L" --n-steps "$STEPS" --dt 0.1 --disorder 1.0 \
          --cutoff 1e-4 --D "$D" --placer "$pl" --ranks-per-node "$RPN" $GPUFLAG --tag "$tag"
    done
  done
done
echo "done $(date)"
