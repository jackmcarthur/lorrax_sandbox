#!/bin/bash
# 12x12 NSCF -> pw2bgw -> wfn2hdf (skills/execute_workflow QE pattern).
# Lmod is broken in scripted contexts (KNOWN_SANDBOX_ERRORS 2026-07-15):
# source the profile shim explicitly, as 05_lorrax_cohsex_native/run.sh does.
#   usage: JID=<jobid> ./run_nscf.sh
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
cd "$(dirname "$0")/nscf"

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load espresso berkeleygw

echo "=== NSCF start $(date +%s) $(date)"
OMP_NUM_THREADS=16 srun --jobid=$JID --overlap --gres=gpu:4 -N 1 -n 4 -c 16 \
    pw.x -npools 4 -i nscf.in > nscf.out 2>&1
echo "=== NSCF end   $(date +%s) $(date)  rc=$?"
grep -c "bands (ev)" nscf.out || true
tail -3 nscf.out

echo "=== pw2bgw start $(date +%s) $(date)"
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --overlap --gres=gpu:1 -N 1 -n 1 \
    pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
echo "=== pw2bgw end   $(date +%s) $(date)  rc=$?"

echo "=== wfn2hdf start $(date +%s) $(date)"
srun --jobid=$JID --overlap --gres=gpu:1 -N 1 -n 1 wfn2hdf.x BIN WFN WFN.h5
echo "=== wfn2hdf end   $(date +%s) $(date)  rc=$?"
ls -la WFN.h5 vxc.dat kih.dat
