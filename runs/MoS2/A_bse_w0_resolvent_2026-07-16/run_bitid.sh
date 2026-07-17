#!/bin/bash
# Measure the BARE (pre-fix) 1x1-vs-2x2 device-invariance baseline.
# Stash ONLY the edited file, run bare probes at both meshes, then restore.
set -uo pipefail
LA=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A
RD=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16
export JID="${JID:?set JID}"
FIX=$RD/fixture_run
IN=$FIX/gnppm_test.in

git -C "$LA" stash push -m bitid-baseline -- src/bse/bse_ring_comm.py
echo "### STASHED (bare builders active)"
git -C "$LA" diff --stat -- src/bse/bse_ring_comm.py | tail -1

cd "$RD"
./lxrun_free.sh      "$FIX" python3 -u "$RD/probe_bitid.py" "$IN" 1 1 "$RD/bare_1x1.npz" > probe_bitid_1x1.log 2>&1
./lxrun_free_4gpu.sh "$FIX" python3 -u "$RD/probe_bitid.py" "$IN" 2 2 "$RD/bare_2x2.npz" > probe_bitid_2x2.log 2>&1

git -C "$LA" stash pop
echo "### RESTORED (edit back)"
git -C "$LA" diff --stat -- src/bse/bse_ring_comm.py | tail -1
echo "### BITID-COMPLETE"
