#!/bin/bash
# LORRAX test suite on agent/bse-exciton-converged (3 source files touched:
# bandstructure/{htransform,bse_setup}.py, bse/exciton_bands.py).
#   usage: JID=<jid> ./run_pytest.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/08_mos2_exciton_converged_2026-07-21
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_exciton_converged_2026-07-21
W=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_bse_exciton_conv
echo "=== pytest start $(date +%T)"
JID=$JID NNODES=1 NTASKS=1 GRES=4 "$R/run_shifter.sh" "$W" \
  python3 -m pytest -q > "$REP/logs/pytest.log" 2>&1
rc=$?
echo "=== pytest end $(date +%T) rc=$rc"
tail -15 "$REP/logs/pytest.log"
exit $rc
