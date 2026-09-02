#!/bin/bash
# One-time environment setup on a Perlmutter LOGIN node (~10 min):
#   bash ~/tn_partitioning/perlmutter/setup_perlmutter.sh                      # env -> $HOME/envs/tnp (default)
#   TNP_COMMON=m1234 bash ~/tn_partitioning/perlmutter/setup_perlmutter.sh     # env -> /global/common/software/m1234/$USER/tnp
# The project software space has a SMALL quota (NERSC: file a ticket to enlarge); it overflowed on 2026-08-26,
# so $HOME (40 GB) is the default. Check with `showquota`.
# Follows docs.nersc.gov (python/nersc-python, parallel-python, cray-mpich, using-python-perlmutter):
#  * conda env on $HOME/envs/tnp (or /global/common/software/<project>/<user>/tnp with TNP_COMMON=<project>)
#  * mpi4py built against Cray MPICH with the GPU Transport Layer linked (accel modules loaded first),
#    so the same build works for CPU runs and for GPU-aware (CuPy) runs
#  * CuPy wheel matched to the cudatoolkit module version
set -euo pipefail
type module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash   # non-interactive ssh: init Lmod
module load conda
module load cudatoolkit craype-accel-nvidia80        # GTL gets linked into mpi4py below
CUDA_MAJOR=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p'); CUDA_MAJOR=${CUDA_MAJOR:-12}
if [ -n "${TNP_COMMON:-}" ]; then ENV_PREFIX=/global/common/software/$TNP_COMMON/$USER/tnp; else ENV_PREFIX=$HOME/envs/tnp; fi
export TNP_ENV=$ENV_PREFIX
echo "env prefix: $ENV_PREFIX   cuda major: $CUDA_MAJOR"
PYVER=3.12   # the pins in requirements.txt (scipy 1.18, numpy 2.4.6) need >=3.12; matches the laptop venv
if [ -d "$ENV_PREFIX" ] && ! "$ENV_PREFIX/bin/python" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" 2>/dev/null; then
  echo "existing env at $ENV_PREFIX has the wrong Python — recreating with $PYVER"
  conda env remove -y -p "$ENV_PREFIX" || rm -rf "$ENV_PREFIX"
fi
if [ ! -d "$ENV_PREFIX" ]; then conda create -y -p "$ENV_PREFIX" python=$PYVER; fi
conda activate "$ENV_PREFIX"
pip install --upgrade pip
pip install --no-cache-dir -r "$(dirname "$0")/requirements.txt"
# mpi4py: MUST be built through the Cray compiler wrapper (docs: "Do NOT conda/pip install mpi4py at NERSC")
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
pip install --no-cache-dir "cupy-cuda${CUDA_MAJOR}x"
# remember the prefix for the batch scripts
echo "export TNP_ENV=$ENV_PREFIX" > "$HOME/.tnp_env"
echo
echo "done. Verify (login node):"
echo "  module load conda cudatoolkit craype-accel-nvidia80 && conda activate $ENV_PREFIX && \\"
echo "  bash -l ~/tn_partitioning/perlmutter/verify.sh"
echo "Verify CuPy + GPU-aware MPI on a GPU node (interactive):"
echo "  salloc -C gpu -q interactive -N 1 --ntasks-per-node=4 -c 32 --gpus-per-task=1 -t 00:10:00 -A <project>"
echo "  export MPICH_GPU_SUPPORT_ENABLED=1; srun -n 4 python -c 'import cupy, mpi4py.MPI as M; print(M.COMM_WORLD.rank, cupy.cuda.runtime.getDeviceProperties(0)[\"name\"])'"
