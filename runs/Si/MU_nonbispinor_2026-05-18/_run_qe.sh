#!/bin/bash -l
# Run the full QE chain (scf → nscf → pw2bgw → wfn2hdf) for the Si nonbispinor reference.
# Usage: _run_qe.sh
set -eu

cd "$(dirname "${BASH_SOURCE[0]}")"

export SLURM_JOBID="${SLURM_JOBID:?need SLURM_JOBID}"

module load espresso/7.3.1-libxc-6.2.2-gpu
module load berkeleygw

# Sanity check
which pw.x pw2bgw.x wfn2hdf.x

#-----------------------------------------------------------------------
# Step 1: SCF
#-----------------------------------------------------------------------
echo "=== SCF ==="
cd qe/scf
OMP_NUM_THREADS=16 srun --jobid=$SLURM_JOBID --gres=gpu:4 -N 1 -n 4 -c 16 \
    pw.x -npools 4 -i scf.in > scf.out 2>&1
grep -E "convergence has been achieved|JOB DONE|ERROR" scf.out | head -5

#-----------------------------------------------------------------------
# Step 2: NSCF (no shifted-q for LORRAX-only run)
#-----------------------------------------------------------------------
echo "=== NSCF ==="
cd ../nscf
ln -sf ../scf/silicon.save .
ln -sf /pscratch/sd/j/jackm/lorrax_sandbox/assets/pseudopotentials/standard/Si.upf Si.upf
OMP_NUM_THREADS=16 srun --jobid=$SLURM_JOBID --gres=gpu:4 -N 1 -n 4 -c 16 \
    pw.x -npools 4 -i nscf.in > nscf.out 2>&1
grep -E "convergence has been achieved|JOB DONE|ERROR" nscf.out | head -5

#-----------------------------------------------------------------------
# Step 3: pw2bgw + wfn2hdf
#-----------------------------------------------------------------------
echo "=== pw2bgw + wfn2hdf ==="
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$SLURM_JOBID --gres=gpu:1 -N 1 -n 1 \
    pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
grep -E "JOB DONE|ERROR" pw2bgw.out | head -5

srun --jobid=$SLURM_JOBID --gres=gpu:1 -N 1 -n 1 wfn2hdf.x BIN WFN WFN.h5

#-----------------------------------------------------------------------
# Final verification
#-----------------------------------------------------------------------
echo "=== Artifacts ==="
ls -la WFN.h5 vxc.dat kih.dat 2>&1 || echo "MISSING ARTIFACTS"
echo "Done."
