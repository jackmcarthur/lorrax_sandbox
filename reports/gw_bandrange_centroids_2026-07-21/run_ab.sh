#!/bin/bash
# Band-range-weight A/B: identical cohsex.in / WFN / 16 GPU, ONLY the
# centroid file changes.  kin_ion.h5 + dipole.h5 are centroid-independent
# (WFN + pseudopotentials only) so they are shared from 03 -> the A/B is
# exactly the k-means weight.
#   usage: JID=<jid> ./run_ab.sh tag [tag ...]
set -uo pipefail
JID="${JID:?set JID}"
BASE=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20
NEW=$BASE/04_lorrax_gw_bandrange_2026-07-21
KM=$NEW/kmeans_wd
SRC03=$BASE/03_lorrax_gw_postfix_2026-07-21
SH=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_bandrange_centroids_2026-07-21/run_shifter_br.sh

for TAG in "$@"; do
  RD=$NEW/ab_$TAG
  mkdir -p "$RD"
  ln -sfn ../../qe/nscf/WFN.h5 "$RD/WFN.h5"
  ln -sfn "$SRC03/kin_ion.h5" "$RD/kin_ion.h5"
  ln -sfn "$SRC03/dipole.h5"  "$RD/dipole.h5"
  cp "$KM/cent_$TAG.txt" "$RD/centroids.txt"
  NC=$(grep -vc '^#' "$RD/centroids.txt")
  sed -e "s#^centroids_file = .*#centroids_file = centroids.txt#" \
      -e "s#^output_file = .*#output_file = eqp0.dat#" \
      "$SRC03/cohsex.in" > "$RD/cohsex.in"

  # GPU zombies from a previous GW step masquerade as a phdf5 "negative
  # offset" crash -- reap before every run (KNOWN_SANDBOX_ERRORS).
  srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:4 \
       bash -c 'pkill -9 -u $USER -f gw.gw_jax; exit 0' >/dev/null 2>&1

  echo "=== $TAG ($NC centroids) start $(date +%T)"
  JID=$JID NNODES=4 NTASKS=16 GRES=4 "$SH" "$RD" \
    python3 -u -m gw.gw_jax -i "$RD/cohsex.in" > "$RD/gw.out" 2>&1
  echo "=== $TAG end $(date +%T) rc=$?"
  tail -3 "$RD/gw.out"
  ls -la "$RD/eqp0.dat" "$RD/sigma_freq_debug.dat" 2>&1 | tail -2
done
