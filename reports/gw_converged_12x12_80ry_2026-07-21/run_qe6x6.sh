#!/bin/bash
# 6x6 NSCF on the SAME converged 80 Ry SCF density, 120 bands -> WFN.h5.
# Exists only because the BSE/htransform machinery cannot be driven from a
# 12x12 GW (see report section 5): the htransform needs ISDF rank > nk*nb,
# and at nk=144 that demands n_mu > 3456, whose replicated fH_R is 102 GiB.
# At nk=36 the same window needs n_mu > 864 -- comfortably satisfied.
# SCF is REUSED unchanged (same cell/pseudos/cutoff/converged density).
#   usage: JID=<jid> ./run_qe6x6.sh
set -uo pipefail
JID="${JID:?set JID}"
Q6=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/_qe6x6
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load espresso berkeleygw
cd "$Q6"

echo "=== NSCF start $(date +%T)"
OMP_NUM_THREADS=16 srun --jobid=$JID --overlap --gres=gpu:4 -N 4 -n 16 -c 16 \
    pw.x -npools 16 -i nscf.in > nscf.out 2>&1
echo "=== NSCF end   $(date +%T) rc=$?"; tail -2 nscf.out

echo "=== pw2bgw start $(date +%T)"
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --overlap --gres=gpu:1 -N 1 -n 1 \
    pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
echo "=== pw2bgw end   $(date +%T) rc=$?"; tail -2 pw2bgw.out

echo "=== wfn2hdf start $(date +%T)"
srun --jobid=$JID --overlap --gres=gpu:1 -N 1 -n 1 wfn2hdf.x BIN WFN WFN.h5
echo "=== wfn2hdf end   $(date +%T) rc=$?"
ls -la WFN.h5 vxc.dat kih.dat 2>&1
