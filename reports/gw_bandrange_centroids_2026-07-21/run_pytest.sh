#!/bin/bash
# Golden-gate / full pytest run on the agent/gw-conduction-postfix worktree.
#   usage: JID=<jid> ./run_pytest.sh [pytest args...]
set -u
JID="${JID:?set JID}"
WT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gw_conduction_postfix
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
MPICH=/opt/udiImage/modules/mpich
srun --jobid=$JID --mpi=cray_shasta -N1 -n1 --gres=gpu:1 --cpus-per-task=32 \
  --overlap --immediate=120 --job-name=lx-pytest --chdir="$WT" \
  shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" --volume="$PHDF5:/lorrax_phdf5" \
    --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$WT/src:$SITE:$DEPS" \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 --env=JAX_ENABLE_X64=1 \
    --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=MPLBACKEND=Agg \
    --env=LORRAX_NGPU=1 --env=OMP_NUM_THREADS=16 \
    --env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include \
    --env=LORRAX_MPICH_LIB_DIR=$MPICH \
    $WT/src/ffi/common/cpp/in_container.sh python3 -m pytest "$@"
