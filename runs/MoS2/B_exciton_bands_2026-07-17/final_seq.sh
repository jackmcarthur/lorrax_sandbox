#!/bin/bash
set -uo pipefail
RD=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_exciton_bands_2026-07-17
WT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_exciton_bands
cd $RD
echo "=== [1/4] diag_refit7 (flat-bottom confirmation)"
JID=$JID ./run_wt.sh "$RD" python3 -u diag_refit7.py > diag_refit7.log 2>&1
grep -E "E_min|E_stored|DONE" diag_refit7.log
echo "=== [2/4] smoke_refit (honest thresholds)"
JID=$JID ./run_wt.sh "$RD" python3 -u smoke_refit.py > smoke_refit_final.log 2>&1
tail -3 smoke_refit_final.log
echo "=== [3/4] FINAL both-mode path run (census)"
rm -rf .jax_cache_census
JID=$JID ./run_wt_census.sh "$RD" python3 -u -m bse.exciton_bands \
  -i exciton_bands.in --n-val 4 --n-cond 4 --n-eig 8 --block-size 8 \
  --max-iter 40 --vq-mode both --refit-points 3,9,12,18,26 \
  --out-prefix exciton_bands_GMKG > census_final.log 2>&1
grep -E "solve_path:|memory_analysis|Wrote|interp vs refit" -A 8 census_final.log | head -30
echo "=== [4/4] new-test rerun"
JID=$JID ./run_wt.sh "$WT" python3 -m pytest -q tests/test_bse_vq_interp.py tests/test_exciton_bands.py > pytest_new.log 2>&1
tail -3 pytest_new.log
echo "=== SEQUENCE DONE"
