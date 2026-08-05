#!/bin/bash -l
# CrI3 live: FM 80Ry SCF -> NSCF(600b) -> WFN.h5 (dftU-strip) -> orbmag sweep.
# (GW q=0 head is blocked by the dipole load-all-psi OOM, same as VI3 -> skipped.)
set -u
export SLURM_JOBID=54622523
J="--jobid=$SLURM_JOBID --overlap"
RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/04_gw_6x6_600b_2026-06-17
OMWT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B_orbmag_wt
LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
BGWBIN=/global/common/software/nersc9/berkeleygw/zen3/gcc-12/mpich/berkeleygw/BerkeleyGW-4.0/bin
SEL=$LROOT/src/ffi/common/cpp/select_gpu.sh; INC=$LROOT/src/ffi/common/cpp/in_container.sh
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=MPICH_GPU_SUPPORT_ENABLED=1"
SHIFTER_OM="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=JAX_ENABLE_X64=1 $LDP --env=PYTHONPATH=$OMWT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
die(){ echo "[cri3] FAILED: $1" >&2; exit 1; }
module load espresso/7.5-libxc-7.0.0-gpu
cd $RUN/qe/scf || die cd
echo "[cri3] SCF $(date '+%H:%M')"
OMP_NUM_THREADS=16 srun $J -N4 -n16 -c16 --gres=gpu:4 pw.x -npools 8 -i scf.in > scf.out 2>&1
grep -q "convergence has been achieved" scf.out || die SCF
cp -r $RUN/qe/scf/CrI3.save $RUN/qe/nscf/CrI3.save || die copysave
cd $RUN/qe/nscf
echo "[cri3] NSCF $(date '+%H:%M')"
OMP_NUM_THREADS=16 srun $J -N4 -n16 -c16 --gres=gpu:4 pw.x -npools 16 -i nscf.in > nscf.out 2>&1
grep -q "JOB DONE" nscf.out || die NSCF
sed -i '/<dftU/,/<\/dftU>/d' CrI3.save/data-file-schema.xml
echo "[cri3] pw2bgw $(date '+%H:%M')"
MPICH_GPU_SUPPORT_ENABLED=0 srun $J -N1 -n1 --gres=gpu:1 pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
[ -f WFN ] || die pw2bgw
srun $J -N1 -n1 --gres=gpu:1 $BGWBIN/wfn2hdf.x BIN WFN WFN.h5 > wfn2hdf.out 2>&1
[ -s WFN.h5 ] || die wfn2hdf
echo "[cri3] orbmag sweep $(date '+%H:%M')"
for NB in 100 200 300 400 500 600; do
  srun $J -N1 -n1 --gres=gpu:1 $SEL $SHIFTER_OM $INC python3 -u -m psp.orbital_magnetization --wfn WFN.h5 --nbnd $NB --mu-scan > $RUN/orbmag_nb${NB}.out 2>&1 || echo "  nb=$NB failed"
done
echo "[cri3] DONE $(date '+%H:%M')"
grep -h "m_z" $RUN/orbmag_nb*.out 2>/dev/null
