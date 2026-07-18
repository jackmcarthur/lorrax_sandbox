#!/bin/bash
# 1000-centroid k-means on the 12x12 WFN (owner ISDF-convergence variant).
# Matches the 640 fixture file's generation convention: literal points
# (--no-orbit; the orbit-closure failure keeps downstream zeta storage
# FULL-BZ, which bse.vq_interp requires), seed 42, default oversample 1.5
# + pivoted-Cholesky prune (window v_x_vc, n_val=nelec=26, n_cond=26).
# Module-free srun+shifter (Lmod broken in scripts, KSE 2026-07-15).
#   usage: JID=<jobid> ./run_kmeans.sh
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
RD="$(cd "$(dirname "$0")" && pwd)"

IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_exciton_bands/src
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

echo "=== kmeans start $(date +%s) $(date)"
srun --jobid="$JID" --overlap -N1 -n1 --gres=gpu:2 --cpus-per-task=32 \
  --cpu-bind=cores --chdir="$RD" \
  shifter --image="$IMAGE" --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" \
    --volume="$PHDF5:/lorrax_phdf5" \
    --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$PYPATH" \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH="$LDLIB" \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=OMP_NUM_THREADS=16 \
    --env=JAX_ENABLE_X64=1 \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
  python3 -u -m centroid.kmeans_cli 1000 --seed 42 --no-orbit --force-shard \
    --qe-save "$RD/../qe/nscf/MoS2.save" > "$RD/run_kmeans.log" 2>&1
echo "=== kmeans end $(date +%s) $(date) rc=$?"
head -1 "$RD"/centroids_frac_1000*.txt 2>/dev/null; wc -l "$RD"/centroids_frac_1000*.txt 2>/dev/null
