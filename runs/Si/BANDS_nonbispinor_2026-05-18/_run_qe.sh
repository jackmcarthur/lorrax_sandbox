#!/bin/bash -l
# Run SCF + NSCF + pw2bgw + wfn2hdf for a given BANDS subdir.
# Usage: _run_qe.sh <subdir>           # e.g. 3x3x3_nb200
set -eu

SUB="${1:?subdir e.g. 3x3x3_nb200}"
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/${SUB}"
[ -d "$RUN_DIR" ] || { echo "no such dir $RUN_DIR" >&2; exit 1; }
cd "$RUN_DIR"
echo "=== working in $RUN_DIR ==="

JID="${SLURM_JOBID:?need SLURM_JOBID}"

# Step 1: SCF — 1 node, 4 GPUs.  Module load inside srun to get the right PATH.
cd qe/scf
echo "=== SCF for $SUB ===" | tee scf.out
OMP_NUM_THREADS=16 srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 -c 16 bash -lc '
    module load espresso berkeleygw 2>/dev/null
    pw.x -npools 4 -i scf.in
' 2>&1 | tee -a scf.out
grep -q "convergence has been achieved" scf.out || { echo "SCF failed for $SUB"; exit 1; }

# Step 2: NSCF — 1 node, 4 GPUs; symlink scf save
cd ../nscf
ln -sf ../scf/silicon.save .
echo "=== NSCF for $SUB ===" | tee nscf.out
OMP_NUM_THREADS=16 srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 -c 16 bash -lc '
    module load espresso berkeleygw 2>/dev/null
    pw.x -npools 4 -i nscf.in
' 2>&1 | tee -a nscf.out
grep -q "JOB DONE" nscf.out || { echo "NSCF failed for $SUB"; exit 1; }

# Step 3: pw2bgw — CPU workaround (GPU pw2bgw can segfault on kih)
echo "=== pw2bgw for $SUB ===" | tee pw2bgw.out
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 bash -lc '
    module load espresso berkeleygw 2>/dev/null
    pw2bgw.x -i pw2bgw.in
' 2>&1 | tee -a pw2bgw.out
grep -q "JOB DONE" pw2bgw.out || { echo "pw2bgw failed for $SUB"; exit 1; }

# Step 4: wfn2hdf — FORTRAN STOP may give nonzero exit even on success, so guard
echo "=== wfn2hdf for $SUB ===" | tee -a pw2bgw.out
srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 bash -lc '
    module load espresso berkeleygw 2>/dev/null
    wfn2hdf.x BIN WFN WFN.h5 || true
' 2>&1 | tee -a pw2bgw.out
[ -f WFN.h5 ] || { echo "wfn2hdf failed for $SUB (no WFN.h5)"; exit 1; }
ls -la WFN.h5

echo "=== $SUB QE chain complete ==="
