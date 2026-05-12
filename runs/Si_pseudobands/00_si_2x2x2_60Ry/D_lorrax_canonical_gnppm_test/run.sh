#!/bin/bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/D_lorrax_canonical_gnppm_test
SHIFTER='shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich --volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc --volume=/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/pscratch/sd/j/jackm/lorrax_slate_cray/stage:/lorrax_slate --env=PYTHONPATH=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=TF_GPU_ALLOCATOR=cuda_malloc_async --env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep:/global/common/software/nersc9/darshan/default/lib --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1 --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache --env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include --env=LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich'
SELECT_GPU=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/ffi/common/cpp/select_gpu.sh
IN_CONT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/ffi/common/cpp/in_container.sh
LOG=${1:-gw.out}
rm -rf tmp sigma_diag.dat sigma_freq_debug.dat
exec srun --jobid=52541884 --mpi=cray_shasta --gres=gpu:4 -N 1 -n 4 --output=$LOG --error=$LOG \
    $SELECT_GPU $SHIFTER $IN_CONT python3 -u -m gw.gw_jax -i $(pwd)/cohsex.in
