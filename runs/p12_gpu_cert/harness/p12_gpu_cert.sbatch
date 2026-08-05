#!/bin/bash
# =============================================================================
# P1.2 pre-push GPU certification (2026-08-01) — rtx-dev, 1 node x 4 GPUs.
# Source: FROZEN snapshot of /work2/08271/jackmc/frontera/lorrax @ 73a9013
# (clean tree) at $SNAP.  NEW defaults under test (decisions.md 2026-08-01):
# LORRAX_FFT_FFI / LORRAX_FFT_FFI_FUSED / LORRAX_BANDS_GEMM_FFI all default
# ON, FFI REQUIRED — on CUDA the cuFFT FFI library is a hard requirement.
#
#   cell build  : build_ffi.sh CUDA leg (phdf5 ON) from the snapshot into a
#                 FRESH build dir ($SP12/build_phdf5).  First-ever object
#                 compile of the COMMITTED unified-tree cuFFT TU
#                 (src/ffi/cpp/cufft/fft_flat_k_cuda_ffi.cc) — a failure here
#                 is a P0 finding, log kept verbatim.  nm -D symbol gate +
#                 readelf DT_NEEDED (cufft/nvrtc).
#   cell smoke  : g108 bare-input GPU GW end-to-end, PURE DEFAULTS — no
#                 LORRAX_FFT_FFI / _FUSED / BANDS_GEMM_FFI exports.  Gates:
#                 rc=0; eqp0 parity vs pinned CPU baseline run_am_p1
#                 (2e-5 eV, multidev_compare); startup report shows the
#                 intended CUDA gate resolutions ([fft_ffi] cuFFT banner,
#                 dial sentences, silent GEMM skip).
#   cell absent : same deck, LORRAX_FFI_SO/_SO_PHDF5 UNSET — the required-
#                 layer contract says startup must REFUSE naming
#                 liblorrax_ffi.so / LORRAX_FFI_SO, never silently fall
#                 back.  Expected rc != 0 + the Gate.require refusal.
# =============================================================================
#SBATCH -J lrxP12gpu
#SBATCH -p rtx-dev
#SBATCH -A PHY25006
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o /scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/p12.%j.out
#SBATCH -e /scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/p12.%j.out

set -u
set +u; module load tacc-apptainer 2>/dev/null; set -u

WKP=/scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert
SNAP=$WKP/srcsnap_73a9013
SP12=/work2/08271/jackmc/frontera/lorrax_ffi_p12_cuda
SO=$SP12/build_phdf5/liblorrax_ffi.so
D=/scratch2/08271/jackmc/mos2_4x4_test
ARC=$D/_archive/2026-07-30/runs
SIF=/scratch2/08271/jackmc/lorrax_setup/py312.sif
VENV=/work2/08271/jackmc/frontera/lorrax_env/.venv
BIND=/home1,/work2,/scratch1,/scratch2,/opt/intel,/tmp
CMP=/scratch2/08271/jackmc/lorrax_setup/multidev_compare.py

echo "########### P1.2 GPU cert  jid=$SLURM_JOB_ID  node=$(hostname)  $(date) ###########"
echo "snapshot: $(cat $SNAP/SNAPSHOT_STAMP.txt | head -1)  (dirty-file count: $(sed -n 2p $SNAP/SNAPSHOT_STAMP.txt))"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

# ------------------------------------------------------------- cell build
echo
echo "################ CELL build (snapshot -> $SP12/build_phdf5) $(date +%T) ################"
cat > $WKP/harness/build_inner.sh <<EOS
export LORRAX_ROOT=$SNAP
export LORRAX_VENV=$VENV
export LORRAX_FFI_STAGE=$SP12
export LORRAX_FFI_PHDF5=1
bash $SNAP/config/frontera/build_ffi.sh --fresh
EOS
timeout 1800 srun -N1 -n1 apptainer exec --nv --bind $BIND $SIF \
    bash $WKP/harness/build_inner.sh > $WKP/logs/build.log 2>&1
BRC=$?
grep -aE "cufft:|configuring|error|Error|FAILED|artifact" $WKP/logs/build.log | head -12 | sed 's/^/    /'
tail -4 $WKP/logs/build.log | sed 's/^/    /'
BUILD_VERDICT=FAIL
if [ $BRC -eq 0 ] && [ -f $SO ]; then
  BUILD_VERDICT=PASS
  echo "  --- symbol gate (nm -D) ---"
  for sym in CufftFlatKCudaFfi CufftGwConvCudaFfi PhdfWriteFfi PhdfReadKchunkUnionFfi EighMpFfi; do
    n=$(nm -D $SO 2>/dev/null | grep -c " T $sym\$")
    [ "$n" -ge 1 ] || BUILD_VERDICT=FAIL
    echo "    $sym: $([ "$n" -ge 1 ] && echo OK || echo MISSING)"
  done
  echo "  --- cufft/nvrtc DT_NEEDED ---"
  readelf -d $SO | grep -E "cufft|nvrtc" | sed 's/^/    /'
else
  echo "  [build] rc=$BRC — P0: the committed cuFFT TU did NOT build with the production toolkit."
  echo "  [build] full log kept at $WKP/logs/build.log — smoke SKIPPED, absent-lib cell still runs."
fi
echo "  BUILD_VERDICT=$BUILD_VERDICT"

# ------------------------------------------------------------- run-dir factory
mk_rundir () {  # $1 dir
  local R=$1
  rm -rf $R; mkdir -p $R
  for f in WFN.h5 dipole.h5 kin_ion.h5 kih.dat vxc.dat RHO out Mo.upf S.upf; do
    ln -sf $D/$f $R/$f
  done
  cp $D/centroids_c100.txt $R/centroids_c100.txt
  sed 's|\.\./centroids_c100\.txt|centroids_c100.txt|' \
      $ARC/run_gpu_rel_g108/gw.in > $R/gw.in
}

# ------------------------------------------------------------- cell smoke
SMOKE_RC=97   # 97 = skipped
if [ "$BUILD_VERDICT" = "PASS" ]; then
  echo
  echo "################ CELL smoke g108, PURE DEFAULTS $(date +%T) ################"
  R=$WKP/run_g108_defaults
  mk_rundir $R
  cat > $R/inner.sh <<EOS
export LORRAX_FFI_PHDF5=1
export LORRAX_FFI_STAGE=$SP12
export LORRAX_FFI_SO=$SO
export LORRAX_FFI_SO_PHDF5=$SO
source $SNAP/config/frontera/ffi_env.sh
export LORRAX_FFI_SO=$SO
export PYTHONPATH=$SNAP/src
export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
# PURE DEFAULTS: no LORRAX_FFT_FFI / LORRAX_FFT_FFI_FUSED / LORRAX_BANDS_GEMM_FFI
cd $R
exec $VENV/bin/python -u -m gw.gw_jax -i gw.in > $R/rank\$SLURM_PROCID.log 2>&1
EOS
  T0=$SECONDS
  timeout 1800 srun --mpi=pmi2 -N1 -n4 \
      apptainer exec --nv --bind $BIND $SIF bash $R/inner.sh > $R/srun.log 2>&1
  SMOKE_RC=$?
  echo "  [smoke] rc=$SMOKE_RC wall=$((SECONDS-T0))s"
  echo "  --- startup report (rank0) ---"
  awk '/LORRAX runtime/{on=1} on{print; if(/^=+$/){r++; if(r==2) exit}}' $R/rank0.log \
      | head -100 | sed 's/^/    /'
  echo "  --- gate/dial/router banners (rank0) ---"
  grep -aE "\[fft_ffi\]|\[bands_gemm\]|LORRAX_FFT_FFI|LORRAX_BANDS_GEMM_FFI|slab_io|read backend|Gloo" \
      $R/rank0.log | head -15 | sed 's/^/    /'
  echo "  --- gloo-line count (expect 0): $(grep -ac '\[runtime\] Gloo' $R/rank0.log) ---"
  echo "  --- tail (rank0) ---"
  grep -av "gocryptfs\|Assume version\|cuInit\|jax_plugins" $R/rank0.log | tail -6 | sed 's/^/    /'

  echo
  echo "######## eqp parity vs pinned CPU baseline (run_am_p1) ########"
  for pair in "eqp0.dat 2e-5" "eqp1.dat 2e-5"; do
    set -- $pair
    if [ -f $ARC/run_am_p1/$1 ] && [ -f $R/$1 ]; then
      echo "-- $1 (tol $2 eV) --"
      apptainer exec --bind $BIND $SIF $VENV/bin/python -u $CMP \
          $ARC/run_am_p1/$1 $R/$1 $2 2>&1 | tail -4 | sed 's/^/    /'
    else
      echo "-- $1: MISSING ($ARC/run_am_p1/$1 / $R/$1) --"
    fi
  done
else
  echo
  echo "################ CELL smoke SKIPPED (build failed) ################"
fi

# ------------------------------------------------------------- cell absent
echo
echo "################ CELL absent-lib (LORRAX_FFI_SO unset) $(date +%T) ################"
RA=$WKP/run_g108_absent
mk_rundir $RA
cat > $RA/inner.sh <<EOS
export LORRAX_FFI_PHDF5=1
export LORRAX_FFI_STAGE=$SP12
source $SNAP/config/frontera/ffi_env.sh
unset LORRAX_FFI_SO LORRAX_FFI_SO_PHDF5
export PYTHONPATH=$SNAP/src
export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
cd $RA
exec $VENV/bin/python -u -m gw.gw_jax -i gw.in > $RA/rank\$SLURM_PROCID.log 2>&1
EOS
T0=$SECONDS
timeout 900 srun --mpi=pmi2 -N1 -n4 \
    apptainer exec --nv --bind $BIND $SIF bash $RA/inner.sh > $RA/srun.log 2>&1
ARC_RC=$?
echo "  [absent] rc=$ARC_RC wall=$((SECONDS-T0))s (nonzero EXPECTED — required-layer refusal)"
echo "  --- refusal evidence (rank0) ---"
grep -aE "unusable on platform|liblorrax_ffi|LORRAX_FFI_SO|REQUIRED|RuntimeError|FileNotFoundError" \
    $RA/rank0.log | head -12 | sed 's/^/    /'
echo "  --- did it fall back / produce output anyway? ---"
ls -l $RA/eqp0.dat $RA/sigma_diag.dat 2>/dev/null | sed 's/^/    /' || echo "    (no outputs — good)"
echo "  --- tail (rank0) ---"
grep -av "gocryptfs\|Assume version\|cuInit\|jax_plugins" $RA/rank0.log | tail -8 | sed 's/^/    /'

echo
echo "=== SUMMARY build=$BUILD_VERDICT smoke_rc=$SMOKE_RC absent_rc=$ARC_RC $(date) ==="
