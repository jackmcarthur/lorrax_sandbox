#!/bin/bash
SHIFTER='shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich --volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc --volume=/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/pscratch/sd/j/jackm/lorrax_slate_cray/stage:/lorrax_slate --env=PYTHONPATH=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=TF_GPU_ALLOCATOR=cuda_malloc_async --env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep:/global/common/software/nersc9/darshan/default/lib --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1 --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache --env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include --env=LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich'
SELECT_GPU=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/ffi/common/cpp/select_gpu.sh
IN_CONT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/ffi/common/cpp/in_container.sh
JID=52541884

for CASE in "208 1464" "208 3264" "808 1464" "808 3264"; do
  set -- $CASE
  NB=$1; NC=$2
  D=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/D_lorrax_para_${NB}_gnppm
  cd $D
  rm -rf tmp sigma_diag.dat sigma_freq_debug.dat
  T0=$(date +%s)
  echo "[$(date +%H:%M:%S)] start nb=${NB} N_c=${NC}"
  srun --jobid=$JID --mpi=cray_shasta --gres=gpu:4 -N 1 -n 4 \
    --output=$D/gw_w10_${NC}.out --error=$D/gw_w10_${NC}.out \
    $SELECT_GPU $SHIFTER $IN_CONT python3 -u -m gw.gw_jax -i $D/cohsex_${NC}.in
  [ -f $D/eqp_g0w0.dat ] && mv $D/eqp_g0w0.dat $D/eqp_g0w0_${NC}.dat
  [ -f $D/sigma_freq_debug.dat ] && mv $D/sigma_freq_debug.dat $D/sigma_freq_debug_${NC}.dat
  T1=$(date +%s)
  echo "[$(date +%H:%M:%S)] done  nb=${NB} N_c=${NC} dt=$((T1-T0))s"
done
echo "ALL DONE at $(date +%H:%M:%S)"
