#!/bin/bash
# zeta_rcond sensitivity of the BGW-anchored si_cohsex_3d gate.
# Runs the tests/regression/si_cohsex_debug fixture at a range of rcond and
# diffs eqp_si_test.dat against the frozen (rcond=1e-10) eqp_si_ref.dat.
#   usage: JID=<jid> ./si_rcond_sweep.sh
set -u
JID="${JID:?set JID}"
WT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gw_conduction_postfix
FIX=$WT/tests/regression/si_cohsex_debug
WORK=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_conduction_postfix_2026-07-21/_si_rcond
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
MPICH=/opt/udiImage/modules/mpich
rm -rf "$WORK"; mkdir -p "$WORK"; cp -r "$FIX"/* "$WORK"/
for RC in 1e-12 1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4; do
  srun --jobid=$JID --mpi=cray_shasta -N1 -n1 --gres=gpu:1 --cpus-per-task=16 \
    --overlap --immediate=120 --job-name=lx-sirc --chdir="$WORK" \
    shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich \
      --volume="$NVHPC:/lorrax_nvhpc" --volume="$PHDF5:/lorrax_phdf5" \
      --volume="$SLATE:/lorrax_slate" \
      --env=PYTHONPATH="$WT/src:$SITE:$DEPS" --env=HDF5_USE_FILE_LOCKING=FALSE \
      --env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep \
      --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
      --env=MPICH_GPU_SUPPORT_ENABLED=1 --env=JAX_ENABLE_X64=1 \
      --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache \
      --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=MPLBACKEND=Agg \
      --env=LORRAX_ZETA_RCOND=$RC \
      --env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include --env=LORRAX_MPICH_LIB_DIR=$MPICH \
      $WT/src/ffi/common/cpp/in_container.sh \
      python3 -u -m gw.gw_jax -i cohsex_si_test.in > "$WORK/gw_$RC.out" 2>&1
  cp "$WORK/eqp_si_test.dat" "$WORK/eqp_$RC.dat" 2>/dev/null
  echo "rcond=$RC rc=$? -> $(ls -la "$WORK/eqp_$RC.dat" 2>/dev/null | awk '{print $5}')"
done
