#!/bin/bash
# htransform_lean.py (reports/scissor_farband_htransform_2026-07-20, env-parameterized)
# run against either the PRE or POST GW, using the SAME source tree
# (lorrax_gw_conduction_postfix) so the interpolation machinery is identical.
#   usage: JID=<jid> ./run_ht.sh {pre|post}
set -u
JID="${JID:?set JID}"
TAG="${1:?pre|post}"
# NOTE: htransform interpolation runs from lorrax_A_figures/src for BOTH pre and
# post — that tree has the arbitrary-q `q_list` kwarg on compute_wfns_fi that
# htransform_lean.py needs (it is not on the gw-conduction-postfix base).  The
# quantity under test (Sigma/eqp) comes from the run dir eqp0/eqp1.dat, so the
# A/B is unaffected; using ONE tree for both sides is what keeps it apples-to-apples.
WT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_figures
DIR=/pscratch/sd/j/jackm/lorrax_sandbox/reports/scissor_farband_htransform_2026-07-20
R=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_conduction_postfix_2026-07-21
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
case "$TAG" in
  pre)  RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu ;;
  post) RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/03_lorrax_gw_postfix_2026-07-21 ;;
  *) echo "usage: $0 {pre|post}"; exit 2 ;;
esac
SHIFTER="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich \
--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc \
--volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 \
--volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate \
--env=PYTHONPATH=$WT/src:$SITE \
--env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
--env=MPLBACKEND=Agg \
--env=SCISSOR_RUN=$RUN --env=SCISSOR_OUT=$R/$TAG \
--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/global/homes/j/jackm/software/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep \
--env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1 \
--env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache \
--env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include --env=LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich"
[ -n "${SCISSOR_WINDOWS:-}" ] && SHIFTER="$SHIFTER --env=SCISSOR_WINDOWS=$SCISSOR_WINDOWS"
srun --jobid=$JID --mpi=cray_shasta -N1 -n1 --gres=gpu:1 --cpus-per-task=16 \
  --overlap --immediate=120 --job-name=lx-ht-$TAG --chdir="$DIR" \
  $SHIFTER $WT/src/ffi/common/cpp/in_container.sh python3 -u htransform_lean.py > "$R/$TAG/ht_lean.log" 2>&1
echo "rc=$? -> $R/$TAG/ht_lean.log"
