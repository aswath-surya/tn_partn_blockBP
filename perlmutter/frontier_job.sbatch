#!/bin/bash
#SBATCH -J tnp-frontier
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1 --gpu-bind=none
#SBATCH -t 02:00:00
#SBATCH -o logs/frontier_%j.out
#SBATCH -e logs/frontier_%j.err
# MEASURED Pareto frontiers (brief Part VI / §7.3-7.4), one 4-node allocation:
#   sweep A  D-sweep at L=8, P=8 (2x4): Σdiscarded-weight vs wall / time-in-MPI per placer  (F2, F3 measured)
#   sweep B  node-sweep at L=12, D=16: P=4 (1x4), 8 (2x4), 16 (4x4): strong scaling + MPI fraction (F5)
# Submit:  sbatch -A <project> perlmutter/frontier_job.sbatch
# Knobs:   DS="4 8 16 32"  PLACERS="tiles rows kahypar_km1_chi random"  REPS=3  GPU=0|1  QUICK=1 (debug-queue size)
# GPU=1 first time: test on the debug queue —  GPU=1 QUICK=1 sbatch -A <project> -q debug -t 00:30:00 perlmutter/frontier_job.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
module load conda cudatoolkit craype-accel-nvidia80
[ -f "$HOME/.tnp_env" ] && source "$HOME/.tnp_env"
conda activate "${TNP_ENV:-$HOME/envs/tnp}"
export MPICH_GPU_SUPPORT_ENABLED=1
# GPU-aware sends between tasks on one node use CUDA IPC, which needs every peer GPU visible: hence
# --gpu-bind=none on every srun (device = rank % 4 in the code). Fallback if IPC still fails: GPU_IPC=0.
[ "${GPU_IPC:-1}" = "0" ] && export MPICH_GPU_IPC_ENABLED=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${OMP_NUM_THREADS:-16}
export SLURM_CPU_BIND=cores MPLBACKEND=Agg
GPU=${GPU:-0}; GPUFLAG=""; [ "$GPU" = "1" ] && GPUFLAG="--gpu"
if [ "${QUICK:-0}" = "1" ]; then DS=${DS:-"8 16"}; REPS=${REPS:-1}; PLACERS=${PLACERS:-"tiles kahypar_km1_chi"}; else DS=${DS:-"4 8 16 32"}; REPS=${REPS:-3}; PLACERS=${PLACERS:-"tiles rows kahypar_km1_chi random"}; fi
[ "$GPU" = "1" ] && [ "${QUICK:-0}" != "1" ] && DS=${DS_GPU:-"8 16 32 64"}
J=$SLURM_JOB_ID; NN=$SLURM_JOB_NUM_NODES
echo "job $J nodes=$NN ($SLURM_JOB_NODELIST) omp=$OMP_NUM_THREADS gpu=$GPU quick=${QUICK:-0} DS=[$DS] placers=[$PLACERS] reps=$REPS start $(date)"

run_case() {  # run_case NODES RPN L D STEPS PLACER REP SWEEP
  local N=$1 RPN=$2 L=$3 D=$4 S=$5 PL=$6 R=$7 SW=$8; local NT=$(( N * RPN ))
  srun -N "$N" -n "$NT" --ntasks-per-node="$RPN" --gpus-per-task=1 --gpu-bind=none python -u run_mpi.py \
      --family tfim --L "$L" --n-steps "$S" --dt 0.1 --disorder 1.0 --cutoff 1e-4 --D "$D" \
      --placer "$PL" --ranks-per-node "$RPN" $GPUFLAG --tag "fr_${SW}_N${N}x${RPN}_L${L}_D${D}_${PL}_r${R}_$J" \
      || echo "FAILED $SW N${N}x${RPN} L=$L D=$D $PL r$R"
}

echo "== sweep A: D-sweep, L=8, P=8 (2x4) $(date)"
for D in $DS; do for pl in $PLACERS; do for r in $(seq 1 "$REPS"); do
  run_case 2 4 8 "$D" 10 "$pl" "$r" A
done; done; echo "   D=$D done $(date)"; done

echo "== sweep B: node-sweep, L=12, D=16: 1x4, 2x4, 4x4 $(date)"
for pl in $PLACERS; do for r in $(seq 1 "$REPS"); do
  run_case 1 4 12 16 10 "$pl" "$r" B
  run_case 2 4 12 16 10 "$pl" "$r" B
  [ "$NN" -ge 4 ] && run_case 4 4 12 16 10 "$pl" "$r" B
done; done
echo "all done $(date)"
