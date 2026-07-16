#!/bin/bash -l
# Vectorised dense-BSE degeneracy analysis across band windows (both arms).
# Reuses existing gw_jax restarts; no gw_jax re-run.  Module-free srun+shifter.
set -u
RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16
LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A
SEL=$LROOT/src/ffi/common/cpp/select_gpu.sh
INC=$LROOT/src/ffi/common/cpp/in_container.sh
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1"
COMMON="--image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=JAX_ENABLE_X64=1 --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache $LDP"
SHIFTER="shifter $COMMON --env=PYTHONPATH=$LROOT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
J="--jobid=${SLURM_JOBID} --overlap"

echo "=== fast-analysis START $(date) JOBID=$SLURM_JOBID ==="
srun $J -N1 -n1 --gres=gpu:1 $SEL $SHIFTER $INC \
    python3 -u $RUN/analyze_fast_all.py > $RUN/analyze_fast_all.out 2>&1
rc=$?
echo "--- analyze_fast_all.py exit=$rc ---"
tail -n 60 $RUN/analyze_fast_all.out
echo "=== fast-analysis DONE $(date) ==="
