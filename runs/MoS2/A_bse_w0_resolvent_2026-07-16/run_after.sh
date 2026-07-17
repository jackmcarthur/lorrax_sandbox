#!/bin/bash
# After-fix validation sequence (jitted seed/project boundaries).
# Sequential sruns (1-GPU then 4-GPU) to avoid GPU contention.
set -uo pipefail
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16
export JID="${JID:?set JID}"
FIX=$PWD/fixture_run
IN=$FIX/gnppm_test.in

echo "### validate 1x1"
./lxrun_free.sh      "$FIX" python3 -u "$PWD/validate_after.py" "$IN" 1 1 "$PWD/val_1x1.npz" > val_1x1.log 2>&1
echo "### validate 2x2"
./lxrun_free_4gpu.sh "$FIX" python3 -u "$PWD/validate_after.py" "$IN" 2 2 "$PWD/val_2x2.npz" > val_2x2.log 2>&1
echo "### profile 1x1 (after)"
./lxrun_free.sh      "$FIX" python3 -u "$PWD/profile_resolvent.py" "$IN" 1 1 8 > prof_1gpu_after.log 2>&1
echo "### profile 2x2 (after)"
./lxrun_free_4gpu.sh "$FIX" python3 -u "$PWD/profile_resolvent.py" "$IN" 2 2 8 > prof_2gpu_after.log 2>&1
echo "### AFTER-RUNS-COMPLETE"
