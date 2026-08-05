#!/bin/bash
# Off-grid leg-separation diagnostic (NO BSE solve).  Runs the SAME 39-Q path
# the failed run 02 used, dumping BOTH interpolated legs and their symmetry
# covariance.  Source = sources/lorrax_A_exciton_wt (agent/bse-exciton-offgrid).
#
#   usage: JID=<jid> [TAG=diag] [ABANDS=none,21,17] ./run_diag_offgrid.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/09_mos2_exciton_smooth_2026-07-21
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_exciton_smooth_2026-07-21
RD=$R/03_diag_offgrid
SH=$R/run_shifter.sh
TAG="${TAG:-diag}"
ABANDS="${ABANDS:-none,21,17}"
EXTRA="${EXTRA:-}"

echo "=== diag start $(date +%T)  a_bands=${ABANDS}"
JID=$JID NNODES=4 NTASKS=16 GRES=4 \
  LX_SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A_exciton_wt/src \
  EXTRA_ENV="--env=LORRAX_SKIP_VQ_GATES=1" "$SH" "$RD" \
  python3 -u $REP/offgrid_diag.py -i exciton.in --eqp eqp1.dat \
    --px 4 --py 4 --a-bands "$ABANDS" --out "$RD/${TAG}" $EXTRA \
    > "$RD/run_${TAG}.log" 2>&1
rc=$?
echo "=== diag end $(date +%T) rc=$rc"
grep -aE "\[eps|\[V_q|\[dist\]|symmetry|Q path|on-grid|2nd-diff|rel 2nd|Wrote|Traceback|Error" \
     "$RD/run_${TAG}.log" | tail -60
exit $rc
