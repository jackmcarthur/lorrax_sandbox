# ===========================================================================
# inner_common.sh -- container-side env + per-rank memory instrument for every
# b600/P=64 leg.
#
# SOURCE it from a leg's inner script.  The env block is a verbatim
# transcription of the container-side runner in
# config/frontera/templates/gw_dev.sbatch (the certified launch block): same
# collectives implementation, same transport hygiene, same LD_LIBRARY_PATH
# order, same node-local staging, same compile-cache policy.  The ONLY reason
# it exists separately is that the non-GW drivers (kmeans_cli, kin_ion_io,
# get_dipole_mtxels, htransform) have no vendored template of their own and
# must not each re-invent the block.
#
# Differences from the template, all deliberate:
#   * the python module + argv are supplied by the caller instead of being
#     hard-coded to gw.gw_jax;
#   * LORRAX_ROOT points at the FROZEN config copy and LORRAX_SRC_FALLBACK at
#     the frozen src copy, so no job reads the live checkout;
#   * python is run as a child (not exec'd) so a sampler can read its
#     /proc/<pid>/status VmHWM.
# Nothing else is changed.  In particular ISDF_JAX_CACHE_DIR is NOT set to ""
# (compile cache ON, per GATES/AT-AU) and the taskset window is 28 cores.
# ===========================================================================
set -u
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=""
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=28 OPENBLAS_NUM_THREADS=28 MKL_NUM_THREADS=28
# compile cache ON (default resolution $SCRATCH/lorrax_jax_cache/np{P}).
export ISDF_JAX_CACHE_DIR=${ISDF_JAX_CACHE_DIR-$SCRATCH/lorrax_jax_cache}

. "$LORRAX_ROOT/config/frontera/mpi_transport_env.sh"

export JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi
export MPITRAMPOLINE_LIB="$LORRAX_MPIWRAPPER_SO"
export LORRAX_MPI_FINALIZE_FIX=skip_atexit

export LD_LIBRARY_PATH="$LORRAX_SLATE_HOST_LIB:$LORRAX_MKL_LIB:$LORRAX_HDF5_ROOT/lib:$LORRAX_IMPI_ROOT/lib/release:$LORRAX_IMPI_ROOT/lib:$LORRAX_IMPI_ROOT/libfabric/lib:$LORRAX_ICC_RUNTIME:${LD_LIBRARY_PATH:-}"

if [ -d /hostlibs ]; then
  RD=/tmp/rdma_stage.$$; mkdir -p "$RD"
  for l in libibverbs.so.1 librdmacm.so.1 libnl-3.so.200 libnl-route-3.so.200 \
           libucp.so.0 libucs.so.0 libuct.so.0 libucm.so.0 libnuma.so.1; do
    [ -e /hostlibs/$l ] && ln -sf /hostlibs/$l "$RD/$l"
  done
  ln -sfn /hostlibs/ucx "$RD/ucx"
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RD:/hostlibs/libibverbs
fi

. "$LORRAX_ROOT/config/frontera/stage_runtime.sh"
# SRC OVERRIDE, stated loudly.  stage_runtime.sh sets LORRAX_SRC_DIR to the
# src INSIDE the staged bundle, and the bundle available here predates this
# change set.  The bundle is used for the VENV (byte-identical deps to every
# certified run); the LORRAX sources come from the frozen snapshot under
# $R/lorrax_frozen/src.  The live checkout is never on the path.
export PYTHONPATH=$LORRAX_OVERLAY_DIR:$LORRAX_FROZEN_SRC
if [ "${SLURM_PROCID:-0}" = "0" ]; then
  echo "[src] PYTHONPATH lorrax src = $LORRAX_FROZEN_SRC (bundle src $LORRAX_SRC_DIR NOT used)"
  echo "[src] gw_init.py sha256 = $(sha256sum $LORRAX_FROZEN_SRC/gw/gw_init.py | cut -c1-32)"
fi

# --------------------------------------------------------------------------
# lorrax_run  <module> [args...]
#
# Runs the driver on this rank's 28-core taskset window (the certified
# layout: contiguous, deliberately NOT socket-local) and samples the child's
# /proc/<pid>/status VmHWM every 5 s into $LORRAX_HWM_DIR/rank_<procid>.hwm.
#
# INSTRUMENT CAVEAT, stated because it bounds every memory claim made from
# these files: VmHWM is monotone, so the LAST sample is the high-water mark
# up to that sample.  A peak reached in the final <5 s window is missed.
# sacct MaxRSS is collected independently as the cross-check; where the two
# disagree the larger is reported.
# --------------------------------------------------------------------------
lorrax_run () {
  local S=${SLURM_LOCALID:-0}
  local lo=$((S*28)) hi=$((S*28+27))
  local hwm_file=""
  if [ -n "${LORRAX_HWM_DIR:-}" ]; then
    mkdir -p "$LORRAX_HWM_DIR" 2>/dev/null
    hwm_file="$LORRAX_HWM_DIR/rank_${SLURM_PROCID:-0}.hwm"
    : > "$hwm_file"
  fi
  taskset -c ${lo}-${hi} "$LORRAX_PY" -u -m "$@" &
  local pypid=$! sampler=""
  if [ -n "$hwm_file" ]; then
    # APPEND one line per sample (no tmp+rename): a torn rename at teardown
    # cost one rank's sample in the P=4 smoke, job 7885313.  VmHWM is
    # monotone, so the reader takes the max over lines.
    ( while kill -0 "$pypid" 2>/dev/null; do
        awk '/^VmHWM|^VmRSS/{printf "%s %s ", $1, $2} END{printf "\n"}' \
            /proc/$pypid/status >> "$hwm_file" 2>/dev/null
        sleep 5
      done ) &
    sampler=$!
  fi
  wait "$pypid"
  local rc=$?
  # Reap the sampler at once.  Left to notice the dead pid on its own it can
  # sit in `sleep` for up to 5 s past the driver, which is long enough for
  # apptainer to log "Terminating fuse-overlayfs after timeout" (job 7885313)
  # and to add that delay to every leg's wall.
  [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
  return $rc
}
