#!/bin/bash
# Exciton-bands driver, 8v8c BSE window.  Interp basis selected by NBAND_TAG:
#   NBAND_TAG=40  -> exciton_{40}_8v8c.in  (WORKING basis; the deliverable)
#   NBAND_TAG=80  -> exciton_{80}_8v8c.in  (over-packed; expected to ABORT at the
#                    tightened on-grid gate — the reconciliation demonstration)
# 4 GPU single node, --px 2 --py 2, --eigh-backend off (matches run 08).
#   usage: JID=<jid> NBAND_TAG=<40|80> A_BAND=<n> ./run_exciton.sh smoke|final
set -uo pipefail
JID="${JID:?set JID}"; A_BAND="${A_BAND:?set A_BAND}"; NBAND_TAG="${NBAND_TAG:-40}"
MODE="${1:?smoke|final}"
RD="$(cd "$(dirname "$0")" && pwd)"

if [ "$MODE" = smoke ]; then
  IN=exciton_smoke_${NBAND_TAG}_8v8c.in; PREFIX=exciton_smoke_${NBAND_TAG}interp_8v8c
  LOG=smoke_${NBAND_TAG}interp_8v8c.log
else
  IN=exciton_${NBAND_TAG}_8v8c.in;       PREFIX=exciton_bands_${NBAND_TAG}interp_8v8c
  LOG=final_${NBAND_TAG}interp_8v8c.log
fi

echo "=== driver($MODE) NBAND_TAG=$NBAND_TAG A_BAND=$A_BAND start $(date +%s) $(date)"
JID="$JID" NGPU=4 "$RD/run10.sh" "$RD" \
  python3 -u -m bse.exciton_bands -i "$IN" \
    --n-val 8 --n-cond 8 --n-eig 8 --block-size 8 --max-iter 40 \
    --vq-mode interp --eigh-backend off --a-band "$A_BAND" \
    --px 2 --py 2 --out-prefix "$PREFIX" > "$RD/$LOG" 2>&1
rc=$?
echo "=== driver($MODE) end $(date +%s) $(date) rc=$rc"
grep -aE "^Wrote|htransform:|f-transform|\[gate\]|\[warn\]|cold |warm |TOTAL|solve_path:|AssertionError|Error" "$RD/$LOG" | tail -18
exit $rc
