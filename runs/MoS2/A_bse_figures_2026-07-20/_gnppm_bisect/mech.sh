#!/bin/bash
set -uo pipefail
export JID=56238206 NNODES=1 NTASKS=4 GRES=4
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gnppm_bisect/src
echo "########## P1: 6bd4dc9 + reshard-via-replicated PATCH @ 2x2 @ 1600 no-orbit (was +14.85 BAD)"
JID=$JID SRC=$SRC LABEL=patch_1600_2x2 CENT=centroids_frac_1600.txt FFB=0 bash $BR/run_bisect.sh || true
echo "########## P2: 6bd4dc9 + PATCH @ 2x2 @ 1496 cascade-ON (was -47.55 BAD)"
JID=$JID SRC=$SRC LABEL=patch_1496_2x2 CENT=centroids_frac_1496.txt FFB=0 bash $BR/run_bisect.sh || true
echo "########## MECH DONE $(date +%s)"
