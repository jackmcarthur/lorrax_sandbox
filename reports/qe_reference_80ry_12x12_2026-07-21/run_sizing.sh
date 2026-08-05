#!/bin/bash
# Stage 1b driver: WFN verification (1 GPU) + GW memory sizing (16 GPU / 4x4 mesh).
# Shifter prefix copied verbatim from skills/execute_workflow/SKILL.md (Cray-MPICH
# variant).  No GW is run — only gflat_memory_model.plan_gflat_chunks.
#   usage: JID=<jobid> ./run_sizing.sh
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WFN="${WFN:-/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/qe/nscf/WFN.h5}"

SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
LORRAX_SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src
VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc \
     --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 \
     --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"
SHIFTER="shifter --module=gpu,mpich --image=nvcr.io/nvidia/jax:25.04-py3 $VOL \
    --env=PYTHONPATH=$LORRAX_SRC:$SITE \
    --env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=JAX_ENABLE_X64=1 \
    --env=LORRAX_SRC=$LORRAX_SRC \
    --env=HDF5_USE_FILE_LOCKING=FALSE"
SEL=$LORRAX_SRC/ffi/common/cpp/select_gpu.sh
INC=$LORRAX_SRC/ffi/common/cpp/in_container.sh

cd "$HERE"

echo "=== verify_wfn start $(date)"
srun --jobid=$JID --overlap --gres=gpu:1 -N 1 -n 1 $SEL $SHIFTER $INC \
    python3 -u "$HERE/verify_wfn.py" "$WFN" 2>&1 | tee verify_wfn.log
echo "=== verify_wfn end   $(date)"

echo "=== size_gw start $(date)  (16 GPU, 4x4 mesh)"
srun --jobid=$JID --overlap --gres=gpu:4 -N 4 -n 16 $SEL $SHIFTER \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    --env=SIZE_JSON=$HERE/gw_sizing.json \
    $INC \
    python3 -u "$HERE/size_gw.py" "$WFN" 1600 2400 3000 4000 2>&1 | tee size_gw.log
echo "=== size_gw end   $(date)"
echo "=== SIZING DONE"
