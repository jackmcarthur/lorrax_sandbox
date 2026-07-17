#!/bin/bash
# proto1 runner — module-free srun+shifter (KNOWN_SANDBOX_ERRORS 2026-07-15:
# Lmod broken in scripts). Mirrors ../lxrun_free.sh but with -c 64 so numpy
# BLAS gets real threads, 1 GPU visible for optional jax hot spots.
#   usage: JID=<jobid> ./proto1_run.sh <python args...>
set -euo pipefail
JID="${JID:?set JID to the salloc job id}"
WD=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study

IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
SLATE_INSTALL=/global/homes/j/jackm/software/slate/install
MPICH=/opt/udiImage/modules/mpich
DARSHAN=/global/common/software/nersc9/darshan/default/lib

PYPATH="$WD:$SRC:$SITE:$DEPS"
LDLIB="$SLATE_INSTALL/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep:$DARSHAN"
JAXCACHE="${SCRATCH:-$HOME}/.jax_cache"

srun --jobid="$JID" --overlap -N1 -n1 -c 64 --gres=gpu:1 --cpu-bind=cores \
  --chdir="$WD" \
  shifter --image="$IMAGE" --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" \
    --volume="$PHDF5:/lorrax_phdf5" \
    --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$PYPATH" \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH="$LDLIB" \
    --env=OMP_NUM_THREADS=32 \
    --env=OPENBLAS_NUM_THREADS=32 \
    --env=MKL_NUM_THREADS=32 \
    --env=JAX_COMPILATION_CACHE_DIR="$JAXCACHE" \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$@"
