#!/bin/bash
# Run the golden-gate pytest suite on ONE GPU (single process) — gate #5.
# Same shifter env as run11.sh but -N1 -n1 --gres=gpu:1 (no select_gpu: one
# process, one visible GPU; jax.distributed is a no-op at proc_count==1).
# Runs from the worktree root so pyproject testpaths=["tests"] resolves.
#   usage: JID=<jid> ./run_tests_1gpu.sh [pytest args / test node ids...]
set -uo pipefail
JID="${JID:?set JID}"
LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_bse_integration
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
IMAGE=nvcr.io/nvidia/jax:25.04-py3
VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1"
COMMON="--image=$IMAGE --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=JAX_ENABLE_X64=1 --env=MPLBACKEND=Agg --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache $LDP"
SHIFTER="shifter $COMMON --env=PYTHONPATH=$LROOT/src:$SITE:$DEPS"

srun --jobid="$JID" --overlap --immediate=120 -N1 -n1 --gres=gpu:1 \
     --cpus-per-task=16 --chdir="$LROOT" \
  $SHIFTER python3 -u -m pytest -q -x "$@"
