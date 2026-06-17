#!/bin/bash -l
set -u
export SLURM_JOBID=54622523
J="--jobid=$SLURM_JOBID --overlap"
RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/VI3/04_gw_6x6_600b_2026-06-17
LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
SEL=$LROOT/src/ffi/common/cpp/select_gpu.sh; INC=$LROOT/src/ffi/common/cpp/in_container.sh
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
VOL="--volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc --volume=/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/pscratch/sd/j/jackm/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1"
SHIFTER="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=JAX_ENABLE_X64=1 --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache $LDP --env=PYTHONPATH=$LROOT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
cd $RUN
echo "[gw] start $(date)"
srun $J -N4 -n16 --gres=gpu:4 $SEL $SHIFTER $INC python3 -u -m gw.gw_jax -i $RUN/cohsex.in > gw.out 2>&1
[ -s eqp0.dat ] && [ -s sigma_mnk.h5 ] && echo "[gw] DONE $(date)" || echo "[gw] FAILED $(date)"
