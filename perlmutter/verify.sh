#!/bin/bash
# Check the Perlmutter environment from the laptop:  ssh perlmutter.nersc.gov 'bash -l ~/tn_partitioning/perlmutter/verify.sh'
type module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load conda cudatoolkit craype-accel-nvidia80 || { echo "MODULE LOAD FAILED — run: module avail conda cudatoolkit craype-accel"; exit 1; }
source "$HOME/.tnp_env" 2>/dev/null || { echo "no ~/.tnp_env — run setup_perlmutter.sh first"; exit 1; }
conda activate "$TNP_ENV" || exit 1
python - <<'PY'
import glob, os, subprocess, sys
import numpy, scipy, quimb, cotengra, kahypar
print("deps ok | python", sys.version.split()[0], "| numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| quimb", quimb.__version__, "| kahypar", kahypar.__version__)
import mpi4py
cfg = mpi4py.get_config()
so = glob.glob(os.path.join(os.path.dirname(mpi4py.__file__), "MPI*.so"))
ldd = subprocess.run(["ldd", so[0]], capture_output=True, text=True).stdout if so else ""
cray = any("libmpi" in l and ("cray" in l.lower() or "gnu" in l) for l in ldd.splitlines())
gtl = "gtl" in ldd.lower()
print(f"mpi4py {mpi4py.__version__} at {os.path.dirname(mpi4py.__file__)} | links Cray MPICH: {cray} | GTL (GPU-aware) linked: {gtl}")
print("  mpi.cfg:", cfg if cfg else "(empty — normal for MPICC= env-var builds and for wheels; not diagnostic)")
print("  ldd MPI*.so | grep -iE 'mpi|gtl':")
for l in ldd.splitlines():
    if "mpi" in l.lower() or "gtl" in l.lower():
        print("   ", l.strip())
if not any("mpi" in l.lower() for l in ldd.splitlines()):
    print("    (no libmpi in ldd → PyPI binary wheel that dlopens MPI at runtime, NOT the Cray build)")
if not cray:
    print("!! mpi4py is NOT linked against Cray MPICH — rerun setup (MPICC=\"cc -shared\" pip install --no-binary=mpi4py mpi4py)")
if not gtl:
    print("!! GTL not linked — GPU-aware MPI (GPU=1 runs) will fail; CPU runs are fine if MPICH_GPU_SUPPORT_ENABLED is unset")
try:
    import cupy; print("cupy", cupy.__version__, "(GPU function needs a GPU node)")
except Exception as e:
    print("cupy import failed (needed for GPU=1 runs only):", e)
PY
echo "env: $TNP_ENV"
