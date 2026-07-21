#!/bin/bash
# Centroid generation sweep for the band-range-weighted k-means A/B.
# Every set: N_c=1600, D3h orbit closure (recovered density point group).
# Only the k-means WEIGHT changes.
#   usage: JID=<jid> ./gen_centroids.sh
set -uo pipefail
JID="${JID:?set JID}"
BASE=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20
WD=$BASE/04_lorrax_gw_bandrange_2026-07-21/kmeans_wd
SH=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_bandrange_centroids_2026-07-21/run_shifter_br.sh
R=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_bandrange_centroids_2026-07-21

run_one () {   # $1 = tag, rest = extra kmeans args
  local tag="$1"; shift
  rm -f "$WD"/centroids_frac_*.txt
  JID=$JID NNODES=1 NTASKS=1 GRES=1 "$SH" "$WD" \
    python3 -u -m centroid.kmeans_cli 1600 --oversample 1.0 "$@" \
    > "$R/kmeans_$tag.log" 2>&1
  local rc=$?
  local out
  out=$(ls "$WD"/centroids_frac_*.txt 2>/dev/null | head -1)
  if [ -n "$out" ]; then
    mv "$out" "$WD/cent_$tag.txt"
    echo "$tag rc=$rc -> cent_$tag.txt ($(grep -vc '^#' "$WD/cent_$tag.txt") centroids)"
  else
    echo "$tag rc=$rc -> NO OUTPUT (see kmeans_$tag.log)"
  fi
}

# baseline: occupied charge density (the production weight to date)
run_one rho    --centroid-weight charge_density
# band-range sweep: occupied-only -> full sigma window (nval=26, ncond=74)
run_one br026  --centroid-weight band_range --weight-bands 0:26
run_one br052  --centroid-weight band_range --weight-bands 0:52
run_one br100  --centroid-weight band_range --weight-bands 0:100
run_one br200  --centroid-weight band_range --weight-bands 0:200
ls -la "$WD"/cent_*.txt
