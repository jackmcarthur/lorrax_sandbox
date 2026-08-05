#!/bin/bash
# Manual pool-free runner on my own allocation, PYTHONPATH -> ppm_sigma_reg worktree.
# Usage: mrun.sh <NGPU> <workdir> <command...>
set -u
JID=$(squeue -u jackm -h -o "%.12i %j" | awk '/lx-ppmreg-jackm/{print $1; exit}')
[ -z "$JID" ] && { echo "NO_PPMREG_ALLOC"; exit 3; }
NGPU="$1"; shift
WD="$1"; shift
WT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_ppm_sigma_reg
NV=/global/homes/j/jackm/software/lorrax_nvhpc
PH=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SL=/global/homes/j/jackm/software/lorrax_slate_cray/stage
SHIFTER="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich \
--volume=$NV:/lorrax_nvhpc --volume=$PH:/lorrax_phdf5 --volume=$SL:/lorrax_slate \
--env=PYTHONPATH=$WT/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources \
--env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep:/global/common/software/nersc9/darshan/default/lib \
--env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1 \
--env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache --env=MPLBACKEND=Agg --env=LORRAX_NGPU=1 \
--env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include --env=LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich"
[ -n "${LORRAX_SIGDBG:-}" ] && SHIFTER="$SHIFTER --env=LORRAX_SIGDBG=$LORRAX_SIGDBG"
cd "$WD" || exit 4
srun --jobid=$JID --overlap -N1 -n$NGPU --gres=gpu:$NGPU --cpus-per-task=8 --immediate=120 \
  $WT/src/ffi/common/cpp/select_gpu.sh $SHIFTER $WT/src/ffi/common/cpp/in_container.sh "$@"
