#!/bin/bash
set -u
WT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_exciton_bands
B=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/04_mos2_12x12_bands_2026-07-18
SHIFTER="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich \
--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc \
--volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 \
--volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate \
--env=PYTHONPATH=$WT/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site \
--env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
--env=MPLBACKEND=Agg \
--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep \
--env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1 \
--env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache \
--env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include --env=LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich"
run() {
  srun --jobid=$SLURM_JOBID --mpi=cray_shasta -N1 -n1 --gres=gpu:1 --cpus-per-task=16 \
    --overlap --immediate=60 --job-name=lx-A-sp6 --chdir="$1" \
    $SHIFTER $WT/src/ffi/common/cpp/in_container.sh "${@:2}"
}
run $B/07_spbands_6x6_fullband python3 -u -m bandstructure.htransform -i hts_full.in 2>&1 | tail -3
run $B/07_spbands_6x6_fullband python3 plot.py
run $B/06_spbands_3x3_fullband python3 plot.py
