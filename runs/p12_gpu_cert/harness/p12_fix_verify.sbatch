#!/bin/bash
# =============================================================================
# P1.2 follow-up (2026-08-01) — verify the startup-report dial-sentence fix
# found by job 7885151 (platform-blind "routed through the FFI handler" for
# LORRAX_BANDS_GEMM_FFI on CUDA).  Source: srcsnap_p12fix = 73a9013 + the
# two-file fix (src/runtime/__init__.py, tests/test_runtime_startup_report.py).
#
#   cell pytest : tests/test_runtime_startup_report.py in the container
#                 (CPU platform — the formatter is pure).
#   cell smoke  : g108 bare-input GPU GW re-run, PURE DEFAULTS, patched
#                 snapshot: rc=0, eqp0 parity vs run_am_p1 (2e-5 eV), and
#                 the CORRECTED dial sentence in the live startup report.
# =============================================================================
#SBATCH -J lrxP12fix
#SBATCH -p rtx-dev
#SBATCH -A PHY25006
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:45:00
#SBATCH -o /scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/p12fix.%j.out
#SBATCH -e /scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/p12fix.%j.out

set -u
set +u; module load tacc-apptainer 2>/dev/null; set -u

WKP=/scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert
SNAP=$WKP/srcsnap_p12fix
SP12=/work2/08271/jackmc/frontera/lorrax_ffi_p12_cuda
SO=$SP12/build_phdf5/liblorrax_ffi.so
D=/scratch2/08271/jackmc/mos2_4x4_test
ARC=$D/_archive/2026-07-30/runs
SIF=/scratch2/08271/jackmc/lorrax_setup/py312.sif
VENV=/work2/08271/jackmc/frontera/lorrax_env/.venv
BIND=/home1,/work2,/scratch1,/scratch2,/opt/intel,/tmp
CMP=/scratch2/08271/jackmc/lorrax_setup/multidev_compare.py

echo "########### P1.2 fix verify  jid=$SLURM_JOB_ID  node=$(hostname)  $(date) ###########"
head -1 $SNAP/SNAPSHOT_STAMP.txt

# ------------------------------------------------------------- cell pytest
echo
echo "################ CELL pytest (startup-report suite) $(date +%T) ################"
cat > $WKP/harness/fix_pytest_inner.sh <<EOS
export PYTHONPATH=$SNAP/src
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
cd $SNAP
exec $VENV/bin/python -m pytest tests/test_runtime_startup_report.py -q
EOS
timeout 900 srun -N1 -n1 apptainer exec --bind $BIND $SIF \
    bash $WKP/harness/fix_pytest_inner.sh > $WKP/logs/fix_pytest.log 2>&1
PRC=$?
echo "  [pytest] rc=$PRC"
tail -6 $WKP/logs/fix_pytest.log | sed 's/^/    /'

# ------------------------------------------------------------- cell smoke
echo
echo "################ CELL smoke g108, PURE DEFAULTS, patched snapshot $(date +%T) ################"
R=$WKP/run_g108_defaults_fix
rm -rf $R; mkdir -p $R
for f in WFN.h5 dipole.h5 kin_ion.h5 kih.dat vxc.dat RHO out Mo.upf S.upf; do
  ln -sf $D/$f $R/$f
done
cp $D/centroids_c100.txt $R/centroids_c100.txt
sed 's|\.\./centroids_c100\.txt|centroids_c100.txt|' \
    $ARC/run_gpu_rel_g108/gw.in > $R/gw.in
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
cd $R
exec $VENV/bin/python -u -m gw.gw_jax -i gw.in > $R/rank\$SLURM_PROCID.log 2>&1
EOS
T0=$SECONDS
timeout 1800 srun --mpi=pmi2 -N1 -n4 \
    apptainer exec --nv --bind $BIND $SIF bash $R/inner.sh > $R/srun.log 2>&1
SMOKE_RC=$?
echo "  [smoke] rc=$SMOKE_RC wall=$((SECONDS-T0))s"
echo "  --- corrected dial sentences (rank0) ---"
grep -a 'dial is' $R/rank0.log | sed 's/^/    /'
echo "  --- fft banners ---"
grep -a '\[fft_ffi\]' $R/rank0.log | cut -c1-120 | sed 's/^/    /'
echo "  --- eqp0 parity vs run_am_p1 (2e-5 eV) ---"
apptainer exec --bind $BIND $SIF $VENV/bin/python -u $CMP \
    $ARC/run_am_p1/eqp0.dat $R/eqp0.dat 2e-5 2>&1 | tail -3 | sed 's/^/    /'

echo
echo "=== SUMMARY pytest_rc=$PRC smoke_rc=$SMOKE_RC $(date) ==="
