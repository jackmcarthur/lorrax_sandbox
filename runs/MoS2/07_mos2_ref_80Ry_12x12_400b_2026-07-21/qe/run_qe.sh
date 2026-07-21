#!/bin/bash
# Converged MoS2 QE reference: SCF (12x12, 80 Ry) -> NSCF (12x12x1, 400 bands,
# 80 Ry) -> pw2bgw (WFN + vxc.dat + kih.dat) -> pw2bgw (RHO) -> wfn2hdf.
# Follows skills/execute_workflow QE patterns; launch geometry copied from
# runs/MoS2/A_bse_figures_2026-07-20/qe/run_qe.sh (the 95 s / 16 GPU calibration
# point) with -npools = number of GPUs so k-point parallelism is near-linear.
#
# Lmod is broken in scripted contexts (KNOWN_SANDBOX_ERRORS 2026-06-16/2026-07-15):
# source the profile shim explicitly before `module load`.
#   usage: JID=<jobid> ./run_qe.sh
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
QE="$(cd "$(dirname "$0")" && pwd)"

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load espresso berkeleygw

# ---- SCF (1 node, 4 GPUs, -npools 4) ----
cd "$QE/scf"
echo "=== SCF start $(date +%s) $(date)"
OMP_NUM_THREADS=16 srun --jobid=$JID --overlap --gres=gpu:4 -N 1 -n 4 -c 16 \
    pw.x -npools 4 -i scf.in > scf.out 2>&1
echo "=== SCF end   $(date +%s) $(date)  rc=$?"
grep -c "convergence has been achieved" scf.out || echo "WARN: SCF not converged"
tail -2 scf.out

# ---- NSCF (4 nodes, 16 GPUs, -npools 16; 144 k-points -> 9 k/pool) ----
cd "$QE/nscf"
cp -r "$QE/scf/MoS2.save" .
echo "=== NSCF start $(date +%s) $(date)"
OMP_NUM_THREADS=16 srun --jobid=$JID --overlap --gres=gpu:4 -N 4 -n 16 -c 16 \
    pw.x -npools 16 -i nscf.in > nscf.out 2>&1
echo "=== NSCF end   $(date +%s) $(date)  rc=$?"
grep -c "bands (ev)" nscf.out || true
tail -3 nscf.out

# ---- pw2bgw: WFN + vxc.dat + kih.dat (CPU workaround; 1 GPU) ----
echo "=== pw2bgw start $(date +%s) $(date)"
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --overlap --gres=gpu:1 -N 1 -n 1 \
    pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
echo "=== pw2bgw end   $(date +%s) $(date)  rc=$?"
tail -2 pw2bgw.out

# ---- pw2bgw: RHO only (separate call so a rhog failure cannot cost the WFN) ----
echo "=== pw2bgw_rho start $(date +%s) $(date)"
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --overlap --gres=gpu:1 -N 1 -n 1 \
    pw2bgw.x -i pw2bgw_rho.in > pw2bgw_rho.out 2>&1
echo "=== pw2bgw_rho end   $(date +%s) $(date)  rc=$?"
tail -2 pw2bgw_rho.out

# ---- wfn2hdf ----
echo "=== wfn2hdf start $(date +%s) $(date)"
srun --jobid=$JID --overlap --gres=gpu:1 -N 1 -n 1 wfn2hdf.x BIN WFN WFN.h5
echo "=== wfn2hdf end   $(date +%s) $(date)  rc=$?"
ls -la WFN.h5 vxc.dat kih.dat RHO 2>&1
echo "=== ALL DONE $(date +%s) $(date)"
