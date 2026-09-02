#!/bin/bash
# Rebuild ONLY mpi4py from source through the Cray wrapper (GTL linked), then show what it links.
#   ssh perlmutter.nersc.gov 'bash -l ~/tn_partitioning/perlmutter/rebuild_mpi4py.sh'
set -euo pipefail
type module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load conda cudatoolkit craype-accel-nvidia80
source "$HOME/.tnp_env"; conda activate "$TNP_ENV"
pip uninstall -y mpi4py || true
echo "== building mpi4py with MPICC='cc -shared' (1-3 min, expect 'Building wheel for mpi4py') =="
MPICC="cc -shared" pip install -v --no-cache-dir --no-binary=mpi4py mpi4py 2>&1 | grep -iE "Building wheel|Successfully|error|cc -shared" | head -20
echo "== ldd of the built extension =="
ldd "$(python -c 'import mpi4py,glob,os;print(glob.glob(os.path.dirname(mpi4py.__file__)+"/MPI*.so")[0])')" | grep -iE "mpi|gtl" || echo "(no libmpi linked — build did not go through cc)"
