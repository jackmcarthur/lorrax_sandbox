#!/bin/bash
# Gate the regenerated FULL-BZ zeta against the production V_qmunu tiles.
# Single GPU; the q-chunk is small because a 48-q zeta chunk is 15.9 GB at
# n_mu = 2412 / ngkmax = 8603.  See reports/.../zeta_gate.py.
#
#   usage: JID=<jid> ./run_zeta_gate.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/09_mos2_exciton_smooth_2026-07-21
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_exciton_smooth_2026-07-21
SH=$R/run_shifter.sh

echo "=== zeta gate start $(date +%T)"
JID=$JID NNODES=1 NTASKS=1 GRES=1 "$SH" "$R" \
  python3 -u $REP/zeta_gate.py \
    --restart $R/tmp_fullbz/isdf_tensors_2412.h5 \
    --zeta    $R/tmp_fullbz/zeta_q.h5 \
    --q-chunk 4 \
    --out     $REP/zeta_gate.json > "$REP/logs/zeta_gate.log" 2>&1
rc=$?
echo "=== zeta gate end $(date +%T) rc=$rc"
tail -25 "$REP/logs/zeta_gate.log"
exit $rc
