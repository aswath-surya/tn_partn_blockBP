#!/bin/bash
# Read-only pull loop (laptop; leave running; Ctrl-C when done). Never modifies the remote.
#   bash perlmutter/watch_pull.sh            (uses ~/.ssh/config Host perlmutter.nersc.gov)
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/results_perlmutter" "$HERE/logs_perlmutter"
while true; do
  rsync -a --include='*.json' --include='*.png' --include='*/' --exclude='*' \
    perlmutter.nersc.gov:~/tn_partitioning/results/ "$HERE/results_perlmutter/" 2>/dev/null
  rsync -a perlmutter.nersc.gov:~/tn_partitioning/logs/ "$HERE/logs_perlmutter/" 2>/dev/null
  echo "$(date +%H:%M:%S) pulled: $(ls "$HERE/results_perlmutter" | grep -c json) json, $(ls "$HERE/logs_perlmutter" 2>/dev/null | wc -l | tr -d ' ') logs"
  sleep 30
done
