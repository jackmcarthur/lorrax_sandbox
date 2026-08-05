#!/bin/bash
# Multi-node (4 nodes / 16 GPU) srun+shifter launcher for the exciton-bands
# driver — ONE PROCESS PER GPU (CUDA_VISIBLE_DEVICES=$SLURM_LOCALID via
# select_gpu.sh) so jax.distributed.initialize() auto-detects the 16-proc /
# 16-device topology from the SLURM env.  Mirrors the PROVEN gw.gw_jax 16-GPU
# launch (runs/VI3/04_gw_6x6_600b_2026-06-17/run_vi3_lorrax.sh line 25) but
# repoints PYTHONPATH/LROOT to the lorrax_A_bse_integration worktree.
#
# NOTE (established LORRAX rule, ffi/common/cpp/run_shifter.sh): use
# `--gres=gpu:4` + select_gpu.sh (CUDA_VISIBLE_DEVICES=$SLURM_LOCALID), NOT
# `--gpus-per-task=1` — the latter breaks JAX's distributed topology sync
# (JAX assumes every rank exposes the same local device ordinals).
#
#   usage: JID=<jobid> ./run11.sh <workdir> <python -u -m ... args...>
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
WD="$1"; shift

LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_bse_integration
SEL=$LROOT/src/ffi/common/cpp/select_gpu.sh
INC=$LROOT/src/ffi/common/cpp/in_container.sh
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
IMAGE=nvcr.io/nvidia/jax:25.04-py3

VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1"
COMMON="--image=$IMAGE --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=TF_GPU_ALLOCATOR=cuda_malloc_async --env=JAX_ENABLE_X64=1 --env=MPLBACKEND=Agg --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache $LDP"
SHIFTER="shifter $COMMON --env=PYTHONPATH=$LROOT/src:$SITE:$DEPS"

# -N4 -n16 --gres=gpu:4 = 16 tasks (1 GPU each) over 4 nodes.  select_gpu.sh
# pins CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; in_container.sh re-exports
# MPICH_GPU_SUPPORT_ENABLED.  No --mpi flag: JAX uses SLURM env, not PMI.
srun --jobid="$JID" --overlap --immediate=120 -N4 -n16 --gres=gpu:4 \
     --cpus-per-task=16 --chdir="$WD" \
  $SEL $SHIFTER $INC "$@"
