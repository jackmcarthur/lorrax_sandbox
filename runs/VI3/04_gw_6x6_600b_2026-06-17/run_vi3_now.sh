#!/bin/bash -l
# Interactive driver: VI3 GW pipeline on the live allocation (races 06:00 maint).
set -u
export SLURM_JOBID=54622523
J="--jobid=$SLURM_JOBID --overlap"
RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/VI3/04_gw_6x6_600b_2026-06-17
LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
OMWT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B_orbmag_wt
SEL=$LROOT/src/ffi/common/cpp/select_gpu.sh; INC=$LROOT/src/ffi/common/cpp/in_container.sh
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
VOL="--volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc --volume=/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/pscratch/sd/j/jackm/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1"
COMMON="--image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=JAX_ENABLE_X64=1 --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache $LDP"
SHIFTER="shifter $COMMON --env=PYTHONPATH=$LROOT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
die(){ echo "[now] FAILED: $1" >&2; exit 1; }

cd $RUN/qe/nscf || die cd
module load espresso/7.5-libxc-7.0.0-gpu
echo "[now] NSCF $(date)"
OMP_NUM_THREADS=16 srun $J -N 4 -n 16 -c 16 --gres=gpu:4 pw.x -npools 16 -i nscf.in > nscf.out 2>&1
grep -q "JOB DONE" nscf.out || die NSCF
echo "[now] pw2bgw $(date)"
MPICH_GPU_SUPPORT_ENABLED=0 srun $J -N 1 -n 1 --gres=gpu:1 pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
[ -f WFN ] || die pw2bgw
srun $J -N 1 -n 1 --gres=gpu:1 wfn2hdf.x BIN WFN WFN.h5 > wfn2hdf.out 2>&1
[ -s WFN.h5 ] || die wfn2hdf
cd $RUN; ln -sf qe/nscf/WFN.h5 WFN.h5
echo "[now] centroids $(date)"
srun $J -N 1 -n 1 --gres=gpu:1 $SEL $SHIFTER $INC python3 -u -m centroid.kmeans_cli 4000 --seed 42 > cent.out 2>&1
[ -s centroids_frac_4000.txt ] || die centroids
echo "[now] dipole $(date)"
srun $J -N 1 -n 1 --gres=gpu:1 $SEL $SHIFTER $INC python3 -u -m psp.get_dipole_mtxels -i $RUN/cohsex.in > dipole.out 2>&1
echo "[now] kin_ion $(date)"
srun $J -N 1 -n 1 --gres=gpu:1 $SEL $SHIFTER $INC python3 -u -m gw.kin_ion_io -i $RUN/cohsex.in > kinion.out 2>&1
[ -s kin_ion.h5 ] || die kin_ion
echo "[now] GW $(date)"
srun $J -N 4 -n 16 --gres=gpu:4 $SEL $SHIFTER $INC python3 -u -m gw.gw_jax -i $RUN/cohsex.in > gw.out 2>&1
[ -s eqp0.dat ] && [ -s sigma_mnk.h5 ] && echo "[now] CORE DONE $(date)" || die gw_jax
