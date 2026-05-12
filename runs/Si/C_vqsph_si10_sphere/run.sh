#!/bin/bash
set -e
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_vqsph_si10_sphere

export SLURM_JOBID=51812085

if command -v lfs >/dev/null 2>&1; then
    mkdir -p "$PWD/tmp"
    lfs setstripe -c 16 -S 4M "$PWD/tmp" >/dev/null 2>&1 || true
fi

SHIFTER_ARGS="--module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \
  --volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc \
  --volume=/pscratch/sd/j/jackm/lorrax_phdf5_openmpi/stage:/lorrax_phdf5 \
  --env=PYTHONPATH=/global/homes/j/jackm/software/lorrax_C/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources \
  --env=HDF5_USE_FILE_LOCKING=FALSE \
  --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
  --env=LD_LIBRARY_PATH=/lorrax_phdf5/lib:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:/opt/hpcx/ompi/lib"

echo "=== branch: $(git -C /global/homes/j/jackm/software/lorrax_C branch --show-current) ==="
echo "=== gw.gw_jax ==="
srun --jobid=$SLURM_JOBID --mpi=pmix --gres=gpu:4 -N 1 -n 4 \
  shifter $SHIFTER_ARGS \
  python3 -u -m gw.gw_jax -i cohsex.in 2>&1
