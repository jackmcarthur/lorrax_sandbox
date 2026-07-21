#!/bin/bash
# Same as run_shifter.sh but PYTHONPATH -> lorrax_gw_conduction_postfix/src
# (agent/gw-conduction-postfix: xi-floor + rank_truncate charge zeta-solve,
#  zeta_rcond default 1e-6).
#   usage: JID=<jid> [NNODES=1] [NTASKS=1] [GRES=1] [EXTRA_ENV="--env=A=B ..."] \
#          ./run_shifter_postfix.sh <workdir> <python args...>
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
NNODES="${NNODES:-1}"; NTASKS="${NTASKS:-1}"; GRES="${GRES:-1}"
WD="$1"; shift

IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC="${LX_SRC:-/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gw_conduction_postfix/src}"
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
JAXCACHE="${SCRATCH:-$HOME}/.jax_cache"

srun --jobid=$JID --overlap --immediate=120 -N "$NNODES" -n "$NTASKS" \
  --gres=gpu:"$GRES" --cpus-per-task=16 --cpu-bind=cores --chdir="$WD" $SEL \
  shifter --image="$IMAGE" --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" \
    --volume="$PHDF5:/lorrax_phdf5" \
    --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$PYPATH" \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH="$LDLIB" \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=JAX_ENABLE_X64=1 \
    --env=JAX_COMPILATION_CACHE_DIR="$JAXCACHE" \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    --env=MPLBACKEND=Agg \
    --env=OMP_NUM_THREADS=16 \
    ${EXTRA_ENV:-} \
    $INC \
    "$@"
