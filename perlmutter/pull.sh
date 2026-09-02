#!/bin/bash
# Perlmutter -> laptop, one shot (read-only): results/ JSON+PNG -> results_perlmutter/, logs/ -> logs_perlmutter/
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/results_perlmutter" "$HERE/logs_perlmutter"
rsync -av --include='*.json' --include='*.png' --include='*/' --exclude='*' perlmutter.nersc.gov:~/tn_partitioning/results/ "$HERE/results_perlmutter/"
rsync -av perlmutter.nersc.gov:~/tn_partitioning/logs/ "$HERE/logs_perlmutter/"
ls -t "$HERE/results_perlmutter" | head
