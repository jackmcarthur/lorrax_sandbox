#!/bin/bash
# MoS2 6x6x1 pipeline for the ingredient-interpolation study.  Module-free
# (Lmod broken in scripts): absolute QE/BGW bins + manual PATH; LORRAX via the
# lxrun shifter pattern with LORRAX_FORCE_FULL_BZ=1 so zeta is written full-BZ.
set -u
JID="${JID:?set JID}"
BASE=/pscratch/sd/j/jackm/lorrax_sandbox
RUN=$BASE/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6
NSCF=$RUN/qe/nscf
LOR=$RUN/lorrax
QEBIN=/global/common/software/nersc9/espresso/7.5-libxc-7.0.0-gpu/bin
BGWBIN=/global/common/software/nersc9/berkeleygw/zen3-ampere80/nvidia23.9/mpich/berkeleygw/BerkeleyGW-4.0/bin

# LORRAX shifter pieces (mirror lxrun_free.sh)
IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC=$BASE/sources/lorrax_A/src
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=$BASE/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
SLATE_INSTALL=/global/homes/j/jackm/software/slate/install
MPICH=/opt/udiImage/modules/mpich
PYPATH="$SRC:$SITE:$DEPS"
LDLIB="$SLATE_INSTALL/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep"
SEL=$SRC/ffi/common/cpp/select_gpu.sh
INC=$SRC/ffi/common/cpp/in_container.sh
JAXCACHE=/pscratch/sd/j/jackm/.jax_cache

lorrax_srun() {  # <ngpu> <args...>
  local NG=$1; shift
  srun --jobid="$JID" --overlap -N1 -n$NG --gres=gpu:$NG --cpu-bind=cores --chdir="$LOR" \
    $SEL shifter --image="$IMAGE" --module=gpu,mpich \
      --volume="$NVHPC:/lorrax_nvhpc" --volume="$PHDF5:/lorrax_phdf5" --volume="$SLATE:/lorrax_slate" \
      --env=PYTHONPATH="$PYPATH" --env=HDF5_USE_FILE_LOCKING=FALSE \
      --env=LD_LIBRARY_PATH="$LDLIB" --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
      --env=MPICH_GPU_SUPPORT_ENABLED=1 --env=JAX_ENABLE_X64=1 \
      --env=JAX_COMPILATION_CACHE_DIR="$JAXCACHE" \
      --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
      --env=LORRAX_FORCE_FULL_BZ=1 \
      $INC "$@"
}

echo "=== [1] NSCF 6x6 $(date) ==="
cd $NSCF
OMP_NUM_THREADS=16 srun --jobid=$JID --overlap --gres=gpu:4 -N1 -n4 -c16 \
  bash -lc "export PATH=$QEBIN:\$PATH; $QEBIN/pw.x -npools 4 -i nscf.in" > nscf.out 2>&1
grep -q "JOB DONE" nscf.out && echo "  NSCF ok" || { echo "  NSCF FAIL"; tail -20 nscf.out; exit 1; }

echo "=== [2] pw2bgw $(date) ==="
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --overlap --gres=gpu:1 -N1 -n1 \
  bash -lc "export PATH=$QEBIN:\$PATH; $QEBIN/pw2bgw.x -i pw2bgw.in" > pw2bgw.out 2>&1
grep -qiE "JOB DONE|pw2bgw" pw2bgw.out && ls -la WFN vxc.dat kih.dat 2>&1 | tail -3

echo "=== [3] wfn2hdf $(date) ==="
srun --jobid=$JID --overlap --gres=gpu:1 -N1 -n1 \
  bash -lc "export PATH=$BGWBIN:\$PATH; $BGWBIN/wfn2hdf.x BIN WFN WFN.h5" > wfn2hdf.out 2>&1
ls -la WFN.h5 2>&1 | tail -1 || { echo "  wfn2hdf FAIL"; tail -20 wfn2hdf.out; exit 1; }

echo "=== [4] stage LORRAX inputs $(date) ==="
cd $LOR
ln -sf $NSCF/WFN.h5 WFN.h5
ln -sf $NSCF/kih.dat kih.dat
ln -sf $NSCF/vxc.dat vxc.dat
ln -sf $BASE/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex/centroids_frac_640.txt centroids_frac_640.txt
ls -la WFN.h5 centroids_frac_640.txt kih.dat

echo "=== [5] kin_ion + dipole $(date) ==="
lorrax_srun 1 python3 -u -m gw.kin_ion_io -i $LOR/cohsex.in > kin_ion.out 2>&1
ls -la kin_ion.h5 2>&1 | tail -1
lorrax_srun 1 python3 -u -m psp.get_dipole_mtxels -i $LOR/cohsex.in > dipole.out 2>&1
ls -la dipole.h5 2>&1 | tail -1

echo "=== [6] gw_jax (4 GPU, force full BZ) $(date) ==="
lorrax_srun 4 python3 -u -m gw.gw_jax -i $LOR/cohsex.in > gw.out 2>&1
echo "  gw.out tail:"; tail -8 gw.out
ls -la tmp/isdf_tensors_*.h5 tmp/zeta_q.h5 2>&1 | tail -5
echo "=== DONE $(date) ==="
