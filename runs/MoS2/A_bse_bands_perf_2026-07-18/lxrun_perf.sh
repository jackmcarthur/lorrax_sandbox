#!/bin/bash
# Module-free srun+shifter runner for the bands-perf worktree (Lmod broken in
# scripts, KSE 2026-07-15).  Pattern copied from
# 04_mos2_12x12_bands_2026-07-18/01_lorrax_exciton_bands/run_wt4_census.sh,
# pointed at sources/worktrees/lorrax_A_bands_perf, GPU count + census cache
# parameterized.
#   usage: JID=<jobid> NGPU=<1|4> [CENSUS=1] [NCPU=32] ./lxrun_perf.sh <workdir> <python args...>
set -euo pipefail

JID="${JID:?set JID to the salloc job id}"
NGPU="${NGPU:-1}"
NCPU="${NCPU:-32}"
WD="$1"; shift

IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC="${SRC_OVERRIDE:-/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_bands_perf/src}"
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

EXTRA_ENV=()
if [[ "${CENSUS:-0}" == "1" ]]; then
  JAXCACHE="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_bands_perf_2026-07-18/.jax_cache_census"
  rm -rf "$JAXCACHE" && mkdir -p "$JAXCACHE"
  EXTRA_ENV+=(--env=JAX_COMPILATION_CACHE_DIR="$JAXCACHE" --env=JAX_LOG_COMPILES=1)
fi

srun --jobid="$JID" --overlap -N1 -n1 --gpus="$NGPU" --cpus-per-task="$NCPU" \
  --cpu-bind=cores --job-name="lx-A-perf-run" \
  --chdir="$WD" \
  shifter --image="$IMAGE" --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" \
    --volume="$PHDF5:/lorrax_phdf5" \
    --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$PYPATH" \
    --env=PYTHONDONTWRITEBYTECODE=1 \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH="$LDLIB" \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=OMP_NUM_THREADS=8 \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
    "${EXTRA_ENV[@]}" \
    "$@"
