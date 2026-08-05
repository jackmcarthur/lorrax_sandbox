#!/bin/bash
# FM CrI3 QE pipeline: SCF (magnetic) -> NSCF (180 bands) -> pw2bgw -> WFN.h5
# Requires SLURM_JOBID (JID) of an lx-alloc; run via lxattach'd shell.
set -e
JID="${SLURM_JOBID:?need SLURM_JOBID}"
RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16
module load espresso/7.5-libxc-7.0.0-gpu

echo "===== SCF (magnetic, 3x3) ====="
cd $RUN/qe/scf
OMP_NUM_THREADS=16 srun --jobid=$JID --gres=gpu:4 --overlap -N1 -n4 -c16 \
    pw.x -npools 4 -i scf.in > scf.out 2>&1
grep -E "convergence has been achieved|total magnetization|absolute magnetization" scf.out | tail -5
if ! grep -q "convergence has been achieved" scf.out; then
    echo "SCF DID NOT CONVERGE — aborting"; exit 2
fi
MAG=$(grep "total magnetization" scf.out | tail -1 | awk '{print $4}')
echo "SCF total magnetization = $MAG (expect ~6)"

echo "===== NSCF (6x6, 180 bands) ====="
cd $RUN/qe/nscf
ln -sf ../scf/CrI3.save .
OMP_NUM_THREADS=16 srun --jobid=$JID --gres=gpu:4 --overlap -N1 -n4 -c16 \
    pw.x -npools 4 -i nscf.in > nscf.out 2>&1
grep -E "convergence|Kohn-Sham states|number of k points" nscf.out | tail -3 || true
tail -3 nscf.out

echo "===== pw2bgw + wfn2hdf (CPU workaround) ====="
cd $RUN/qe/nscf
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --gres=gpu:1 --overlap -N1 -n1 \
    pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
tail -4 pw2bgw.out
srun --jobid=$JID --gres=gpu:1 --overlap -N1 -n1 wfn2hdf.x BIN WFN WFN.h5 > wfn2hdf.out 2>&1
ls -la $RUN/qe/nscf/WFN.h5
echo "===== QE PIPELINE DONE ====="
