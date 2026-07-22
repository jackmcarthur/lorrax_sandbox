#!/bin/bash
# Shifter launcher for the htransform distributed-eigh (FFI) initiative.
# PYTHONPATH -> sources/lorrax_A_htffi_wt/src  (branch agent/htransform-distributed-eigh)
#
#   usage: JID=<jid> [NNODES=1] [NTASKS=1] [GRES=1] [LX_SRC=...] [ALLOC=bfc|async]
#          [EXTRA_ENV="--env=A=B ..."] ./run_shifter.sh <workdir> <python args...>
#
# ALLOC:
#   bfc   (default) XLA BFC pool at MEM_FRACTION=0.95 — what every prior
#         htransform measurement used; device.memory_stats() HWM is meaningful.
#   async cuda_malloc_async / platform allocator — REQUIRED by the cuSOLVERMp and
#         SLATE FFI paths (with BFC at 0.95 the NCCL/CAL side starves and
#         cusolverMpSyevd reports "NCCL error 1 unhandled cuda error";
#         see config/modulefiles/lorrax/*.lua).
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
NNODES="${NNODES:-1}"; NTASKS="${NTASKS:-1}"; GRES="${GRES:-1}"
ALLOC="${ALLOC:-bfc}"
WD="$1"; shift

IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC="${LX_SRC:-/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A_htffi_wt/src}"
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

# Per-launch JAX coordinator.  jax.distributed's SLURM auto-detection derives
# the port from SLURM_JOB_ID, so EVERY step of one allocation collides on it —
# fatal in a shared pool (see KNOWN_SANDBOX_ERRORS 2026-07-21).  Pin a unique
# port per launch; runtime.init_jax_distributed takes the explicit path when
# JAX_COORDINATOR_ADDRESS is set.
# The node list is PINNED (not left to srun) so process 0's host is known:
# the coordinator address has to name the node rank 0 lands on.  NODE_OFFSET
# picks which nodes of the allocation to take — default the LAST ones, so a
# co-tenant agent taking the first nodes does not land on top of this run.
NODES_ALL=($(scontrol show hostnames "$(squeue -j "$JID" -h -o %N 2>/dev/null)" 2>/dev/null))
NODE_OFFSET="${NODE_OFFSET:-$(( ${#NODES_ALL[@]} - NNODES ))}"
[ "$NODE_OFFSET" -lt 0 ] && NODE_OFFSET=0
NODES_USE=("${NODES_ALL[@]:$NODE_OFFSET:$NNODES}")
NODELIST=$(IFS=,; echo "${NODES_USE[*]}")
COORD_PORT="${COORD_PORT:-$((20000 + RANDOM % 20000))}"
COORD="${NODES_USE[0]:-localhost}:${COORD_PORT}"
echo "[run_shifter] nodes=$NODELIST coordinator=$COORD alloc=$ALLOC" >&2

case "$ALLOC" in
  async) ALLOC_ENV="--env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=TF_GPU_ALLOCATOR=cuda_malloc_async" ;;
  # bfc60: the BFC pool (so device.memory_stats()['peak_bytes_in_use'] exists
  # and the two backends' high-water marks are comparable) held to 60% of the
  # card, which leaves NCCL/CAL the headroom they starve for at 0.95.  This is
  # the mode to MEASURE in; async is the mode to RUN production in.
  bfc60) ALLOC_ENV="--env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.60" ;;
  *)     ALLOC_ENV="--env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95" ;;
esac

srun --jobid=$JID --overlap --immediate=120 -N "$NNODES" -n "$NTASKS" \
  ${NODELIST:+--nodelist="$NODELIST"} \
  --gres=gpu:"$GRES" --cpus-per-task=16 --cpu-bind=cores --chdir="$WD" ${SEL_OVERRIDE:-$SEL} \
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
    --env=JAX_COORDINATOR_ADDRESS="$COORD" \
    $ALLOC_ENV \
    --env=MPLBACKEND=Agg \
    --env=OMP_NUM_THREADS=16 \
    ${EXTRA_ENV:-} \
    $INC \
    "$@"
