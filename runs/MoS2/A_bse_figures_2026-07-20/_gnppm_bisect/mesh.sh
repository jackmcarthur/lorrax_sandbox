#!/bin/bash
# Decisive mesh-shape experiment: committed 6bd4dc9 on a 2x2 mesh (4 GPU),
# the SAME mesh run_fix used.  Isolates mesh-shape from WIP-vs-committed code.
set -uo pipefail
export JID=56238206
export NNODES=1 NTASKS=4 GRES=4   # -> 4 processes -> 2x2 mesh
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
B=$BR/bisect_one.sh
echo "########## M1: 6bd4dc9 + 1496 recovered-D3h + cascade ON + 2x2 mesh -> DECISIVE"
JID=$JID bash $B 6bd4dc9 base_1496_ibz_2x2 0 centroids_frac_1496.txt || true
echo "########## M2: 6bd4dc9 + 1600 no-orbit + 2x2 mesh -> control (non-orbit basis on 2x2)"
JID=$JID bash $B 6bd4dc9 base_1600_full_2x2 0 centroids_frac_1600.txt || true
echo "########## MESH DONE $(date +%s)"
