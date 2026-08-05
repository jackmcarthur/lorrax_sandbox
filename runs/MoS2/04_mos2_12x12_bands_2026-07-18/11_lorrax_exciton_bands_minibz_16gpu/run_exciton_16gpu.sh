#!/bin/bash
# 16 GPU / 4 node exciton-bands driver — OFF (point-value head) and ON
# (--head-minibz-average) variants.  --px 4 --py 4 = 16 GLOBAL devices (square
# mesh: cusolverMp requires p==q).  Driver config EXACTLY matches dir 10's
# 4-GPU deliverable (exciton_bands_40interp_8v8c.dat: nband=40 basis, 8v8c BSE
# window, --a-band 33, --max-iter 40, --vq-mode interp) EXCEPT the mesh
# (--px 4 --py 4 vs --px 2 --py 2), the eigh backend (cusolverMp FFI vs the
# single-node native 'off' — the FFI needs one process per device, impossible
# on the 2x2 single-process driver), and the ON flag.  The eigh backend must
# not change the physics, only how the C_q eigh is computed → the OFF run
# still validates against dir 10 numerics.
#   usage: JID=<jid> [EIGH=cusolvermp|slate|off] ./run_exciton_16gpu.sh off|on
set -uo pipefail
JID="${JID:?set JID}"; MODE="${1:?off|on}"; EIGH="${EIGH:-cusolvermp}"
RD="$(cd "$(dirname "$0")" && pwd)"
IN=exciton_40_8v8c.in
if [ "$MODE" = on ]; then
  FLAG="--head-minibz-average"; PREFIX=exciton_bands_16gpu_on;  LOG=run16_on.log
elif [ "$MODE" = off ]; then
  FLAG="";                      PREFIX=exciton_bands_16gpu_off; LOG=run16_off.log
else
  echo "MODE must be off|on"; exit 2
fi
echo "=== 16gpu($MODE) EIGH=$EIGH start $(date) JID=$JID ==="
JID="$JID" "$RD/run11.sh" "$RD" \
  python3 -u -m bse.exciton_bands -i "$IN" \
    --n-val 8 --n-cond 8 --n-eig 8 --block-size 8 --max-iter 40 \
    --vq-mode interp --eigh-backend "$EIGH" --a-band 33 \
    --px 4 --py 4 $FLAG --out-prefix "$PREFIX" > "$RD/$LOG" 2>&1
rc=$?
echo "=== 16gpu($MODE) end $(date) rc=$rc ==="
grep -aE "^\[dist\]|^Wrote|htransform|f-transform|\[gate\]|\[prep\]|\[warn\]|cusolverMp|cusolvermp|SLATE|cold |warm |TOTAL|solve_path:|evs sharding|memory_analysis|AssertionError|Error|Traceback" "$RD/$LOG" | tail -35
exit $rc
