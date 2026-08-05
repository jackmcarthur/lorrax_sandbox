#!/bin/bash
set -uo pipefail
export JID=56238206
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gnppm_bisect/src
echo "########## CHK 4x4 (16 GPU) 1600 no-orbit"
JID=$JID SRC=$SRC LABEL=chk_1600_4x4 CENT=centroids_frac_1600.txt FFB=0 NNODES=4 NTASKS=16 GRES=4 bash $BR/run_bisect.sh || true
echo "########## CHK 2x2 (4 GPU) 1600 no-orbit"
JID=$JID SRC=$SRC LABEL=chk_1600_2x2 CENT=centroids_frac_1600.txt FFB=0 NNODES=1 NTASKS=4 GRES=4 bash $BR/run_bisect.sh || true
echo "########## CHK DONE $(date +%s)"
