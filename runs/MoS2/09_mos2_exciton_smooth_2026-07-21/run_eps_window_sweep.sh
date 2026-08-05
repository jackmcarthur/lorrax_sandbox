#!/bin/bash
# fH-window sweep for the off-grid eps_c(k+Q) leg (the fix search).
#   usage: JID=<jid> [WINDOWS="14,14;12,12;10,10;8,10"] [KSTRIDE=24] ./run_eps_window_sweep.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/09_mos2_exciton_smooth_2026-07-21
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_exciton_smooth_2026-07-21
RD=$R/03_diag_offgrid
SH=$R/run_shifter.sh
WINDOWS="${WINDOWS:-14,14;12,12;10,10;8,10}"
KSTRIDE="${KSTRIDE:-24}"
TAG="${TAG:-window_sweep}"

echo "=== window sweep start $(date +%T)  windows=${WINDOWS}"
JID=$JID NNODES=4 NTASKS=16 GRES=4 \
  LX_SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A_exciton_wt/src \
  EXTRA_ENV="--env=LORRAX_SKIP_VQ_GATES=1" "$SH" "$RD" \
  python3 -u $REP/eps_window_sweep.py -i exciton.in --eqp eqp1.dat \
    --px 4 --py 4 --windows "$WINDOWS" --k-stride "$KSTRIDE" \
    --out "$RD/${TAG}" > "$RD/run_${TAG}.log" 2>&1
rc=$?
echo "=== window sweep end $(date +%T) rc=$rc"
grep -aE "^\[window|^    (M-Gamma|Gamma-K)|^Wrote|Traceback|Error" "$RD/run_${TAG}.log" | tail -40
exit $rc
