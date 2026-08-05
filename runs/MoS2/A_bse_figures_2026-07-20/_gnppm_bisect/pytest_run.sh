#!/bin/bash
# Run the golden gates + bispinor + new mesh-invariance gate in the shifter
# container on the caller's allocation.  env: JID  WT(=worktree root)  KEXPR
set -uo pipefail
JID="${JID:?}"; WT="${WT:?}"; KEXPR="${KEXPR:-gnppm or bispinor or cohsex or si_cohsex_3d}"
SRC="$WT/src"
IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
SLATE_INSTALL=/global/homes/j/jackm/software/slate/install
MPICH=/opt/udiImage/modules/mpich
DARSHAN=/global/common/software/nersc9/darshan/default/lib
PYPATH="$SRC:$SITE:$DEPS"
LDLIB="$SLATE_INSTALL/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep:$DARSHAN"
SEL=$SRC/ffi/common/cpp/select_gpu.sh
INC=$SRC/ffi/common/cpp/in_container.sh
JAXCACHE="${JAXCACHE:-$WT/.pytest_jax_cache}"; mkdir -p "$JAXCACHE"

srun --jobid=$JID --overlap --immediate=120 -N 1 -n 1 \
  --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores --chdir="$WT" $SEL \
  shifter --image="$IMAGE" --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" --volume="$PHDF5:/lorrax_phdf5" --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$PYPATH" --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH="$LDLIB" \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 --env=JAX_ENABLE_X64=1 \
    --env=JAX_COMPILATION_CACHE_DIR="$JAXCACHE" \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    --env=MPLBACKEND=Agg --env=OMP_NUM_THREADS=16 \
    $INC \
    python3 -u -m pytest -q "$WT/tests/test_gw_jax_regression.py" "$WT/tests/test_symmetry_unfold.py" \
      -k "$KEXPR" -p no:cacheprovider 2>&1
echo "=== PYTEST rc=$? ==="
