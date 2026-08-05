#!/bin/bash
# Committed-history anchor points for the bisect table.  All cascade ON.
set -uo pipefail
export JID=56238206
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
B=$BR/bisect_one.sh
echo "########## D1: 6bd4dc9 + full-D3h 1421 basis, cascade ON (alt orbit-closed basis)"
JID=$JID bash $B 6bd4dc9 base_fullD3h1421_ibz 0 centroids_fullD3h_1421.txt || true
echo "########## D2: 9416992 (2026-06-30 old anchor) + 1496, cascade ON"
JID=$JID bash $B 9416992 anchor_0630_1496_ibz 0 centroids_frac_1496.txt || true
echo "########## D3: 565750a (2026-06-15 ~5wk) + 1496, cascade ON"
JID=$JID bash $B 565750a anchor_0615_1496_ibz 0 centroids_frac_1496.txt || true
echo "########## ANCHORS DONE $(date +%s)"
