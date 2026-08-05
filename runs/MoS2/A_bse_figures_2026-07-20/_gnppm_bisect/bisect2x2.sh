#!/bin/bash
# Bisect the 2x2-mesh screened-Sigma_c catastrophe over the window.
# Fixture: 1600 no-orbit (cascade-free full-BZ core path) on a 2x2 mesh (4 GPU).
# Committed 6bd4dc9 @ 2x2 = +14.85 (BAD).  Find if/when it was GOOD.
set -uo pipefail
export JID=56238206
export NNODES=1 NTASKS=4 GRES=4
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
B=$BR/bisect_one.sh
for spec in \
  "565750a jax09after_0615" \
  "c7e6695 jax09mig_0613" \
  "0f355b7 pre_jax09_0517" ; do
  set -- $spec
  echo "########## $2 ($1) @ 2x2 @ 1600 no-orbit"
  JID=$JID bash $B "$1" "b2x2_$2" 0 centroids_frac_1600.txt || echo "  (run failed/skipped)"
done
echo "########## BISECT2x2 DONE $(date +%s)"
