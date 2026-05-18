#!/bin/bash -l
# Run SCF + NSCF + pw2bgw + wfn2hdf for a given kgrid subdir.
# Usage: _run_qe.sh <NxNxN>
set -eu

KG="${1:?kgrid e.g. 2x2x2}"
RUN_DIR="$(dirname "${BASH_SOURCE[0]}")/${KG}"
cd "$RUN_DIR"

module load espresso berkeleygw
PW=$(which pw.x)
PW2BGW=$(which pw2bgw.x)
WFN2HDF=$(which wfn2hdf.x)
echo "PW=$PW PW2BGW=$PW2BGW WFN2HDF=$WFN2HDF"

JID="${SLURM_JOBID:?need SLURM_JOBID}"

# SCF — 1 node, 4 GPUs
cd qe/scf
echo "=== SCF for $KG ===" | tee scf.out
OMP_NUM_THREADS=16 srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 -c 16 \
    $PW -npools 4 -i scf.in 2>&1 | tee -a scf.out
grep -q "convergence has been achieved" scf.out || { echo "SCF failed for $KG"; exit 1; }

# NSCF — 1 node, 4 GPUs; symlink scf save
cd ../nscf
ln -sf ../scf/silicon.save .
echo "=== NSCF for $KG ===" | tee nscf.out
OMP_NUM_THREADS=16 srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 -c 16 \
    $PW -npools 4 -i nscf.in 2>&1 | tee -a nscf.out
grep -q "JOB DONE" nscf.out || { echo "NSCF failed for $KG"; exit 1; }

# pw2bgw — CPU workaround
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 \
    $PW2BGW -i pw2bgw.in 2>&1 | tee pw2bgw.out
grep -q "JOB DONE" pw2bgw.out || { echo "pw2bgw failed for $KG"; exit 1; }

# wfn2hdf
srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 $WFN2HDF BIN WFN WFN.h5 2>&1
ls -la WFN.h5

echo "=== $KG complete ==="
