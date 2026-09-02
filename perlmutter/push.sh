#!/bin/bash
# Laptop -> Perlmutter (run from a LOCAL terminal, never while ssh'd in): code only, ~1 MB.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
rsync -av --exclude '.venv' --exclude '__pycache__' --exclude '.DS_Store' --exclude 'results' \
  --exclude 'results_perlmutter' --exclude 'logs_perlmutter' --exclude 'refs' --exclude 'logs' --exclude '*.pkl' \
  "$HERE/" perlmutter.nersc.gov:~/tn_partitioning/
