#!/bin/bash
# Full-band exciton-bands driver stages (the fix for the iQ 6/9/16-17 dips).
# 4 GPU single node, --px 2 --py 2, --eigh-backend off (cusolverMp trap on the
# single-process 2x2 mesh), --vq-mode interp, n_val/n_cond/n_eig match the
# delivered 640c run so the .dat overlays 1:1.  --a-band 28: the f-transform
# width a is tied to a low-bandwidth conduction band (abs band 28) so the
# selected conduction caches sit in the f'~1 region (a large default a from the
# dispersive top guard band 39 would compress eps_c(k+Q); same lesson as agent
# B's clean (26,34) SP window a_band=28).
#   usage: JID=<jobid> ./run_driver.sh smoke|final
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
MODE="${1:?smoke|final}"
RD="$(cd "$(dirname "$0")" && pwd)"

if [ "$MODE" = smoke ]; then
  IN=exciton_smoke_fullbasis.in; PREFIX=exciton_smoke_fullbasis; LOG=smoke_fullbasis.log
else
  IN=exciton_bands_fullbasis.in; PREFIX=exciton_bands_fullbasis; LOG=final_fullbasis.log
fi

echo "=== driver($MODE) start $(date +%s) $(date)"
JID=$JID "$RD/run_wt4.sh" "$RD" \
  python3 -u -m bse.exciton_bands -i "$IN" \
    --n-val 4 --n-cond 4 --n-eig 8 --block-size 8 --max-iter 40 \
    --vq-mode interp --eigh-backend off --a-band 28 \
    --px 2 --py 2 --out-prefix "$PREFIX" > "$RD/$LOG" 2>&1
echo "=== driver($MODE) end $(date +%s) $(date) rc=$?"
grep -aE "^Wrote|full-band htransform|\[gate\]|cold |warm |TOTAL" "$RD/$LOG" | tail -14
