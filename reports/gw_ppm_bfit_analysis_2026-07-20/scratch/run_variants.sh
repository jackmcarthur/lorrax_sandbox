#!/bin/bash
# Run floor (owner ω̃-floor) then cap (high-Ω) variants sequentially on the pool.
set -u
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_ppm_bfit_analysis_2026-07-20
cd $REP

echo "=== FLOOR variant (LORRAX_OMEGA_FLOOR_RY=0.5) $(date) ==="
cd $REP/run_floor && rm -rf tmp
PORT=$((14000 + RANDOM % 3000))
NGPU=4 EXTRA_ENV="--env=LORRAX_OMEGA_FLOOR_RY=0.5 --env=LORRAX_COORD_PORT=$PORT" \
  bash $REP/scratch/mrun_bfit.sh "$(pwd)" python3 -u -m gw.gw_jax -i cohsex.in > floor.log 2>&1
echo "floor rc=$?  $(date)"

echo "=== CAP variant (LORRAX_OMEGA_CAP_RY=2.0) $(date) ==="
cd $REP/run_cap && rm -rf tmp
PORT=$((17000 + RANDOM % 3000))
NGPU=4 EXTRA_ENV="--env=LORRAX_OMEGA_CAP_RY=2.0 --env=LORRAX_COORD_PORT=$PORT" \
  bash $REP/scratch/mrun_bfit.sh "$(pwd)" python3 -u -m gw.gw_jax -i cohsex.in > cap.log 2>&1
echo "cap rc=$?  $(date)"
echo "=== DONE $(date) ==="
