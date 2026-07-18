#!/bin/bash
# 16-GPU gw_jax COHSEX+do_screened, 1000-centroid variant.  Copy of
# ../00_lorrax_cohsex/run_gw.sh WITHOUT the dipole/kin_ion steps (dipole.h5 +
# kin_ion.h5 are centroid-independent — symlinked from 00_lorrax_cohsex,
# recorded in manifest reuse).  Module-free srun+shifter (KSE 2026-07-15);
# mirrors skills/execute_workflow GWJAX step 6 with the exciton-bands
# worktree src.  Requires a 4-node allocation (4x4 device mesh).
#   usage: JID=<jobid> ./run_gw.sh
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
RD="$(cd "$(dirname "$0")" && pwd)"
cd "$RD"

IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_exciton_bands/src
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
SLATE_INSTALL=/global/homes/j/jackm/software/slate/install
MPICH=/opt/udiImage/modules/mpich
DARSHAN=/global/common/software/nersc9/darshan/default/lib
PYPATH="$SRC:$SITE:$DEPS"
LDLIB="$SLATE_INSTALL/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep:$DARSHAN"
SEL=$SRC/ffi/common/cpp/select_gpu.sh
INC=$SRC/ffi/common/cpp/in_container.sh

SHIFTER_ARGS=(--image="$IMAGE" --module=gpu,mpich
  --volume="$NVHPC:/lorrax_nvhpc" --volume="$PHDF5:/lorrax_phdf5"
  --volume="$SLATE:/lorrax_slate"
  --env=PYTHONPATH="$PYPATH" --env=HDF5_USE_FILE_LOCKING=FALSE
  --env=LD_LIBRARY_PATH="$LDLIB"
  --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0
  --env=MPICH_GPU_SUPPORT_ENABLED=1
  --env=JAX_ENABLE_X64=1)

ls -la dipole.h5 kin_ion.h5

echo "=== gw_jax start $(date +%s) $(date)"
srun --jobid=$JID --overlap --gres=gpu:4 -N 4 -n 16 --chdir="$RD" $SEL \
  shifter "${SHIFTER_ARGS[@]}" \
  --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  $INC \
  python3 -u -m gw.gw_jax -i "$RD/cohsex.in" > gw.out 2>&1
echo "=== gw_jax end   $(date +%s) $(date)  rc=$?"
ls -la eqp0.dat tmp/ 2>&1 | head
