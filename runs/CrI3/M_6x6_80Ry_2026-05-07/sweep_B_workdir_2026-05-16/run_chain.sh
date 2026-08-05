#!/usr/bin/env bash
# Drive the remaining sweep configs sequentially.
# Usage: ./run_chain.sh CFG1 R1 B1 CFG2 R2 B2 ...
cd "$(dirname "$0")"
while [ "$#" -ge 3 ]; do
  CFG="$1"; R="$2"; B="$3"
  shift 3
  echo "=== Driving $CFG ==="
  bash ./run_one.sh "$CFG" "$R" "$B" > "runlog_${CFG}.log" 2>&1
  echo "=== $CFG complete ==="
done
echo "=== SWEEP_B ALL DONE $(date) ===" | tee -a /tmp/bispinor_80ry_sweep_progress.log
