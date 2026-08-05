#!/bin/bash
# Recent-vs-standing: does the 2x2 catastrophe exist at old commits?
# Fixture 1496 cascade-ON (closes, avoids the pre-06-16 write_ibz fallback crash).
# 6bd4dc9 @ 2x2 @ 1496 = -47.55 (BAD).  6bd4dc9 @ 2x2 @ 1600 = +14.85 (BAD).
set -uo pipefail
export JID=56238206
export NNODES=1 NTASKS=4 GRES=4
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
B=$BR/bisect_one.sh
for spec in \
  "9416992 s_0630_1496 centroids_frac_1496.txt" \
  "9416992 s_0630_1600 centroids_frac_1600.txt" \
  "fc9984e s_0616_1496 centroids_frac_1496.txt" \
  "6728ceb s_0511_1496 centroids_frac_1496.txt" ; do
  set -- $spec
  echo "########## $2 ($1) @ 2x2  cent=$3"
  JID=$JID bash $B "$1" "$2" 0 "$3" || echo "  (run failed/skipped $1)"
done
echo "########## STANDING DONE $(date +%s)"
