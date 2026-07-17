#!/bin/bash
# proto0 srun+shifter wrapper (module-free; mirrors interp_study/mos2_6x6/run_6x6.sh).
# Usage: JID=<jobid> ./proto0_run.sh <script.py> [args...]
set -u
JID="${JID:?set JID}"
BASE=/pscratch/sd/j/jackm/lorrax_sandbox
STUDY=$BASE/runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study
IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC=$BASE/sources/lorrax_A/src
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=$BASE/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
SLATE_INSTALL=/global/homes/j/jackm/software/slate/install
MPICH=/opt/udiImage/modules/mpich
PYPATH="$STUDY:$SRC:$SITE:$DEPS"
LDLIB="$SLATE_INSTALL/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep"
SEL=$SRC/ffi/common/cpp/select_gpu.sh
INC=$SRC/ffi/common/cpp/in_container.sh
JAXCACHE=/pscratch/sd/j/jackm/.jax_cache

exec srun --jobid="$JID" --overlap -N1 -n1 --gres=gpu:1 --cpu-bind=cores --chdir="$STUDY" \
  $SEL shifter --image="$IMAGE" --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" --volume="$PHDF5:/lorrax_phdf5" --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$PYPATH" --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH="$LDLIB" \
    --env=JAX_ENABLE_X64=1 \
    --env=JAX_COMPILATION_CACHE_DIR="$JAXCACHE" \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    $INC python3 -u "$@"
