#!/bin/bash
#SBATCH -J tnp-pareto
#SBATCH -C gpu
#SBATCH -q regular                  # debug is capped at 30 min; this job needs ~60-75 min
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 32                       # 2*floor(64/4) logical CPUs per task (NERSC formula)
#SBATCH --gpus-per-task=1 --gpu-bind=none
#SBATCH -t 01:30:00
#SBATCH -o logs/pareto_%j.out
#SBATCH -e logs/pareto_%j.err
# ONE job, ONE 2-node allocation, everything the frontiers need (brief §7.1-7.3):
#   step 1  Phase 1 decisive measurement: identical 4-rank runs as 1x4 (NVLink) and 2x2 (Slingshot)
#   step 2  Phase 2 scaling point: 8 ranks = 2x4
#   step 0  Phase 0 microbenchmarks (last: GPU up to n=4096, CPU capped at 2048 — a 4096 CPU SVD is 42 s)
# Submit:   cd ~/tn_partitioning && sbatch -A <project> perlmutter/pareto_job.sbatch
# Knobs:    CONFIGS="L:D:steps ..."  PLACERS="..."  REPS=3  GPU=0|1 (GPU=1 -> cupy tensors, GPU-aware MPI)
# Afterwards on the laptop: pull.sh -> plot_phase0.py, plot_phase1.py, run_test.py --machine ...
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
module load conda cudatoolkit craype-accel-nvidia80
[ -f "$HOME/.tnp_env" ] && source "$HOME/.tnp_env"
conda activate "${TNP_ENV:-$HOME/envs/tnp}"
export MPICH_GPU_SUPPORT_ENABLED=1
# GPU-aware sends between tasks on one node use CUDA IPC, which needs every peer GPU visible: hence
# --gpu-bind=none on every srun (device = rank % 4 in the code). Fallback if IPC still fails: GPU_IPC=0.
[ "${GPU_IPC:-1}" = "0" ] && export MPICH_GPU_IPC_ENABLED=0          # mpi4py was built with GTL linked (setup_perlmutter.sh)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${OMP_NUM_THREADS:-16}
export SLURM_CPU_BIND=cores MPLBACKEND=Agg
CONFIGS=${CONFIGS:-"8:16:10 8:32:10 12:16:10"}
PLACERS=${PLACERS:-"tiles rows kahypar_km1_chi random"}
REPS=${REPS:-3}
GPU=${GPU:-0}; GPUFLAG=""; [ "$GPU" = "1" ] && GPUFLAG="--gpu"
J=$SLURM_JOB_ID
echo "job $J nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) omp=$OMP_NUM_THREADS gpu=$GPU start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

run_case() {  # run_case NODES RPN LABEL L D STEPS PLACER REP
  local N=$1 RPN=$2 LAB=$3 L=$4 D=$5 S=$6 PL=$7 R=$8
  local NT=$(( N * RPN ))
  srun -N "$N" -n "$NT" --ntasks-per-node="$RPN" --gpus-per-task=1 --gpu-bind=none python -u run_mpi.py \
      --family tfim --L "$L" --n-steps "$S" --dt 0.1 --disorder 1.0 --cutoff 1e-4 --D "$D" \
      --placer "$PL" --ranks-per-node "$RPN" $GPUFLAG --tag "p1_${LAB}_L${L}_D${D}_${PL}_r${R}_$J" \
      || echo "FAILED $LAB L=$L D=$D $PL r$R"
}

echo "== step 1: Phase 1, 1x4 vs 2x2 $(date)"
for cfg in $CONFIGS; do
  IFS=: read -r L D S <<< "$cfg"
  for pl in $PLACERS; do
    for r in $(seq 1 "$REPS"); do
      run_case 1 4 N1x4 "$L" "$D" "$S" "$pl" "$r"     # all cut bonds on NVLink / shared memory
      run_case 2 2 N2x2 "$L" "$D" "$S" "$pl" "$r"     # cut between rank pairs crosses Slingshot
    done
  done
  echo "   config $cfg done $(date)"
done

echo "== step 2: scaling point 2x4 = 8 ranks $(date)"
for cfg in $CONFIGS; do
  IFS=: read -r L D S <<< "$cfg"
  for pl in $PLACERS; do
    for r in $(seq 1 "$REPS"); do
      run_case 2 4 N2x4 "$L" "$D" "$S" "$pl" "$r"
    done
  done
done
echo "== step 0: Phase 0 microbenchmarks (2x4) $(date)"
srun -N 2 -n 8 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none python -u perlmutter/microbench.py \
     --gpu --max-n 4096 --ranks-per-node 4 --tag "gpu_$J" || echo "STEP0-GPU FAILED (cupy/GPU-aware MPI)"
srun -N 2 -n 8 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none python -u perlmutter/microbench.py \
     --max-n 2048 --ranks-per-node 4 --tag "cpu_$J" || echo "STEP0-CPU FAILED"
echo "all done $(date)"
