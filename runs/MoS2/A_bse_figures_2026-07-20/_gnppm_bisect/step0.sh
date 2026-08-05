#!/bin/bash
# STEP 0 driver: three runs at current code (6bd4dc9) to decide which subsystem to bisect.
set -uo pipefail
export JID=56238206
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gnppm_bisect/src
R=$BR/run_bisect.sh
echo "########## A: 1600 no-orbit (auto full-BZ fallback) -> expect GOOD"
JID=$JID SRC=$SRC LABEL=cur_1600_full CENT=centroids_frac_1600.txt FFB=0 bash $R || true
echo "########## B: 1496 recovered-D3h, cascade ON -> expect BAD (reproduce)"
JID=$JID SRC=$SRC LABEL=cur_1496_ibz  CENT=centroids_frac_1496.txt FFB=0 bash $R || true
echo "########## C: 1496 recovered-D3h, cascade OFF (FORCE_FULL_BZ=1) -> DECISIVE"
JID=$JID SRC=$SRC LABEL=cur_1496_full CENT=centroids_frac_1496.txt FFB=1 bash $R || true
echo "########## STEP0 DONE $(date +%s)"
