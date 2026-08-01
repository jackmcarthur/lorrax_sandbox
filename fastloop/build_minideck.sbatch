#!/bin/bash
#SBATCH -J fl_minideck
#SBATCH -p development
#SBATCH -A PHY25006
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -t 01:30:00
#SBATCH -o /scratch2/08271/jackmc/lorrax_sandbox/fastloop/work/build_minideck.%j.out
#SBATCH -e /scratch2/08271/jackmc/lorrax_sandbox/fastloop/work/build_minideck.%j.out
# ---------------------------------------------------------------------------
# ONE-TIME mini-deck build + first certification, in one allocation:
#
#   1  QE NSCF nbnd=40 / 2x2x1 from the ONE converged SCF density
#      (/scratch2/.../mos2_4x4_test/out/MoS2.save), staged into a fresh
#      build dir — the certified job-7884642 recipe (qe_deck_b300.sbatch),
#      shrunk: nbnd 300->40, k-grid 4x4->2x2. Cell/cutoff (and therefore
#      the 24x24x80 FFT grid) are pinned by that SCF density and reused.
#   2  pw2bgw (WFN_mini + vxc_mini + kih_mini; RHO_mini) -> wfn2hdf
#   3  assemble fastloop/deck_mini/ (WFN_mini.h5, RHO/vxc/kih, the QE save
#      dir WITHOUT wfc*.dat — nothing reads them after pw2bgw)
#   4  run_fastloop.py --mode pin   (P=1 chain -> reference/ + pins)
#   5  run_fastloop.py --mode check (P=1 AND shard4 legs vs the pins —
#      certifies the instrument itself: rerun determinism + device
#      invariance in the same job that pinned)
#
# -n 56 is for ibrun; every LORRAX driver runs with JAX_PROCESS_COUNT=1
# pinned by the runner (the job-7884642 SLURM_NTASKS lesson).
# Refuses to clobber an existing deck unless FASTLOOP_REBUILD=1.
# ---------------------------------------------------------------------------
set -u
set +u
module load intel/19.1.1 impi/19.0.9
module load phdf5 2>/dev/null
module load tacc-apptainer 2>/dev/null
set -u
export OMP_NUM_THREADS=1

SB=/scratch2/08271/jackmc/lorrax_sandbox
FL=$SB/fastloop
DECK=$FL/deck_mini
BUILD=$FL/work/deck_build.${SLURM_JOB_ID:-manual}
SRC_SAVE=/scratch2/08271/jackmc/mos2_4x4_test/out/MoS2.save
QE=$HOME/software/qe-7.2/bin
BGW=$HOME/software/BerkeleyGW-master/bin
SIF=${LORRAX_SIF:-/scratch2/08271/jackmc/lorrax_setup/py312.sif}
VENVPY=/work2/08271/jackmc/frontera/lorrax_env/.venv/bin/python
BIND=/home1,/work2,/scratch1,/scratch2,/opt/intel,/tmp
export LORRAX_SRC=${LORRAX_SRC:-/work2/08271/jackmc/frontera/lorrax/src}
export LORRAX_SRC_COMMIT=$(git -C "$LORRAX_SRC/.." rev-parse --short HEAD 2>/dev/null || echo unknown)

echo "=== minideck build $(date) host=$(hostname) job=${SLURM_JOB_ID:-manual} src@$LORRAX_SRC_COMMIT ==="
if [ -e "$DECK/WFN_mini.h5" ] && [ "${FASTLOOP_REBUILD:-0}" != 1 ]; then
  echo "[minideck] deck exists — SKIPPING the QE leg (steps 1-3); going"
  echo "[minideck] straight to pin+check (FASTLOOP_REBUILD=1 rebuilds the deck)"
  SKIP_QE=1
else
  SKIP_QE=0
fi
if [ "$SKIP_QE" = 0 ]; then

# ---------- step 1: stage SCF density + NSCF nbnd=40 / 2x2x1 ---------------
mkdir -p "$BUILD/mini_out/MoS2.save"
cd "$BUILD"
cp -p $SRC_SAVE/charge-density.dat   mini_out/MoS2.save/
cp -p $SRC_SAVE/data-file-schema.xml mini_out/MoS2.save/
cp -p $SRC_SAVE/Mo.upf $SRC_SAVE/S.upf mini_out/MoS2.save/
cp -p $SRC_SAVE/Mo.upf $SRC_SAVE/S.upf .
md5sum $SRC_SAVE/charge-density.dat mini_out/MoS2.save/charge-density.dat
cp $DECK/nscf_mini.in $DECK/pw2bgw_mini.in $DECK/pw2bgw_mini_rho.in .

echo "##### STEP 1  NSCF nbnd=40 2x2x1  $(date) #####"
t0=$(date +%s)
ibrun -n 8 $QE/pw.x -nk 2 -ndiag 1 -in nscf_mini.in > nscf_mini.out 2>&1
echo "[nscf rc=$? wall=$(( $(date +%s) - t0 ))s]"
grep -aE "convergence NOT|%%%%|JOB DONE" nscf_mini.out | tail -3
grep -aE "highest occupied|number of k points|Kohn-Sham states|Sym. Ops|FFT dimensions|number of electrons" nscf_mini.out | head -8
grep -qa "JOB DONE" nscf_mini.out || { echo "FATAL: NSCF did not finish"; exit 8; }

# ---------- step 2: pw2bgw + wfn2hdf ---------------------------------------
echo "##### STEP 2  pw2bgw + wfn2hdf  $(date) #####"
t0=$(date +%s)
ibrun -n 8 $QE/pw2bgw.x -in pw2bgw_mini.in > pw2bgw_mini.out 2>&1
echo "[pw2bgw a rc=$? wall=$(( $(date +%s) - t0 ))s]"; tail -3 pw2bgw_mini.out
t0=$(date +%s)
ibrun -n 8 $QE/pw2bgw.x -in pw2bgw_mini_rho.in > pw2bgw_mini_rho.out 2>&1
echo "[pw2bgw b rc=$? wall=$(( $(date +%s) - t0 ))s]"; tail -3 pw2bgw_mini_rho.out
t0=$(date +%s)
ibrun -n 1 $BGW/wfn2hdf.x BIN mini_out/WFN_mini WFN_mini.h5 > wfn2hdf_mini.out 2>&1
echo "[wfn2hdf rc=$? wall=$(( $(date +%s) - t0 ))s]"; tail -3 wfn2hdf_mini.out
[ -s WFN_mini.h5 ] || { echo "FATAL: WFN_mini.h5 missing/empty"; exit 8; }

echo "##### STEP 2b  WFN_mini.h5 metadata (in-container)  $(date) #####"
apptainer exec --bind $BIND "$SIF" \
  "$VENVPY" -u $DECK/probe_wfn_bands.py "$BUILD/WFN_mini.h5"

# ---------- step 3: assemble the checked-in deck ---------------------------
echo "##### STEP 3  assemble $DECK  $(date) #####"
rm -f mini_out/MoS2.save/wfc*.dat   # nothing reads them after pw2bgw
mkdir -p "$DECK/mini_out/MoS2.save"
cp -p WFN_mini.h5 "$DECK/"
cp -p mini_out/vxc_mini.dat mini_out/kih_mini.dat mini_out/RHO_mini "$DECK/"
cp -p mini_out/MoS2.save/charge-density.dat \
      mini_out/MoS2.save/data-file-schema.xml \
      mini_out/MoS2.save/Mo.upf mini_out/MoS2.save/S.upf \
      "$DECK/mini_out/MoS2.save/"
cp -p Mo.upf S.upf "$DECK/"
ls -la "$DECK" "$DECK/mini_out/MoS2.save"
du -sh "$DECK"
fi  # SKIP_QE

# ---------- steps 4-5: pin, then certify the gate --------------------------
echo "##### STEP 4  run_fastloop --mode pin (P=1 chain)  $(date) #####"
apptainer exec --bind $BIND "$SIF" \
  "$VENVPY" -u $FL/run_fastloop.py --mode pin ${FASTLOOP_PIN_ARGS:-}
PIN_RC=$?
echo "[pin rc=$PIN_RC]"
if [ $PIN_RC -ne 0 ]; then
  echo "[minideck SUMMARY] job=${SLURM_JOB_ID:-manual} pin_rc=$PIN_RC check_rc=- (build kept at $BUILD)"
  exit $PIN_RC
fi

echo "##### STEP 5  run_fastloop --mode check (p1 + shard4)  $(date) #####"
apptainer exec --bind $BIND "$SIF" \
  "$VENVPY" -u $FL/run_fastloop.py --mode check ${FASTLOOP_CHECK_ARGS:-}
CHECK_RC=$?
echo "[check rc=$CHECK_RC]"

echo "[minideck SUMMARY] job=${SLURM_JOB_ID:-manual} pin_rc=$PIN_RC check_rc=$CHECK_RC"
echo "=== minideck build done $(date) ==="
exit $CHECK_RC
