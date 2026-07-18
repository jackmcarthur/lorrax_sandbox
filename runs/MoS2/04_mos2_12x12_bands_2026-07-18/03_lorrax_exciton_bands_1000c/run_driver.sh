#!/bin/bash
# Exciton-bands driver stages, 1000-centroid variant.  Mirrors the sibling's
# final 640c invocation (4 GPU single node, --px 2 --py 2, --eigh-backend off
# [cusolverMp trap on single-process 2x2 mesh], --refit-r-chunk 256 [40G HBM],
# max_iter 40 [it80 probe showed 0.000000 eV delta], n_eig 8).
# vq-mode=interp: the htransform refit ground truth stays REPRESENTATION-
# LIMITED at 12x12 for ANY reachable n_mu (needs ns*n_mu >= nk*nb = 11520
# -> n_mu >= 5760); spot checks are the on-grid DENSE stored-tile truth
# (diag_dense_head pattern), not refit.
#   usage: JID=<jobid> ./run_driver.sh smoke|final
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
MODE="${1:?smoke|final}"
RD="$(cd "$(dirname "$0")" && pwd)"

if [ "$MODE" = smoke ]; then
  IN=exciton_smoke_nw_1000c.in; PREFIX=exciton_smoke_1000c; LOG=smoke_1000c.log
else
  IN=exciton_bands_nw_1000c.in; PREFIX=exciton_bands_1000c; LOG=final_1000c.log
fi

echo "=== driver($MODE) start $(date +%s) $(date)"
JID=$JID "$RD/run_wt4_census.sh" "$RD" \
  python3 -u -m bse.exciton_bands -i "$IN" \
    --n-val 4 --n-cond 4 --n-eig 8 --block-size 8 --max-iter 40 \
    --vq-mode interp --refit-r-chunk 256 --eigh-backend off \
    --px 2 --py 2 --out-prefix "$PREFIX" > "$RD/$LOG" 2>&1
echo "=== driver($MODE) end $(date +%s) $(date) rc=$?"
grep -aE "^Wrote|cold |warm |TOTAL" "$RD/$LOG" | tail -12
