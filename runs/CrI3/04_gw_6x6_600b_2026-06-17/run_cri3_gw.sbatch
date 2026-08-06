#!/bin/bash -l
#SBATCH -J lx-D-cri3gw
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out
# Full CrI3 GW pipeline (queued; runs post-maintenance ~Jun 24).
# CORE: FM 80Ry SCF -> NSCF(600b) -> WFN.h5 -> centroids(4000) -> dipole ->
#   kin_ion -> gw_jax (GN-PPM, sigma_mnk(omega), 120-band Sigma).
# DOWNSTREAM (|| true): DFT+GW bandstructures, orbmag sweep, figures.
set -u
RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/04_gw_6x6_600b_2026-06-17
LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
OMWT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B_orbmag_wt
SEL=$LROOT/src/ffi/common/cpp/select_gpu.sh
INC=$LROOT/src/ffi/common/cpp/in_container.sh
BGWBIN=/global/common/software/nersc9/berkeleygw/zen3/gcc-12/mpich/berkeleygw/BerkeleyGW-4.0/bin
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep:/global/common/software/nersc9/darshan/default/lib --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1"
COMMON="--image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=TF_GPU_ALLOCATOR=cuda_malloc_async --env=JAX_ENABLE_X64=1 --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache $LDP"
SHIFTER="shifter $COMMON --env=PYTHONPATH=$LROOT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
SHIFTER_OM="shifter $COMMON --env=PYTHONPATH=$OMWT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
die(){ echo "[sbatch] FAILED: $1" >&2; exit 1; }

# ---------- 1. QE FM SCF (80Ry) ----------
cd $RUN/qe/scf || die "cd scf"
module load espresso/7.5-libxc-7.0.0-gpu
echo "[sbatch] SCF $(date)"
OMP_NUM_THREADS=16 srun -N 4 -n 16 -c 16 --gres=gpu:4 pw.x -npools 8 -i scf.in > scf.out 2>&1
grep -q "convergence has been achieved" scf.out || die "SCF"
# ---------- 2. NSCF (600b) ----------
cp -r $RUN/qe/scf/CrI3.save $RUN/qe/nscf/CrI3.save || die "copy save"
cd $RUN/qe/nscf || die "cd nscf"
echo "[sbatch] NSCF $(date)"
OMP_NUM_THREADS=16 srun -N 4 -n 16 -c 16 --gres=gpu:4 pw.x -npools 16 -i nscf.in > nscf.out 2>&1
grep -q "JOB DONE" nscf.out || die "NSCF"
# DFT+U workaround: strip <dftU> so pw2bgw skips orthoUwfc()/.hub1 (davcio bug).
sed -i '/<dftU/,/<\/dftU>/d' CrI3.save/data-file-schema.xml
MPICH_GPU_SUPPORT_ENABLED=0 srun -N 1 -n 1 --gres=gpu:1 pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
[ -f WFN ] || die "pw2bgw"
srun -N 1 -n 1 --gres=gpu:1 $BGWBIN/wfn2hdf.x BIN WFN WFN.h5 > wfn2hdf.out 2>&1
[ -s WFN.h5 ] || die "wfn2hdf"

# ---------- 3. LORRAX preprocessing + GW ----------
cd $RUN || die "cd RUN"
ln -sf qe/nscf/WFN.h5 WFN.h5
echo "[sbatch] centroids $(date)"
srun -N 1 -n 1 --gres=gpu:1 $SEL $SHIFTER $INC python3 -u -m centroid.kmeans_cli 4000 --seed 42 --oversample 1.0 > cent.out 2>&1
[ -s centroids_frac_4000.txt ] || die "centroids"
echo "[sbatch] dipole $(date)"
echo "[sbatch] dipole skipped (wcoul0_source=epshead avoids 724GiB OOM)"
echo "[sbatch] kin_ion $(date)"
srun -N 1 -n 1 --gres=gpu:1 $SEL $SHIFTER $INC python3 -u -m gw.kin_ion_io -i $RUN/cohsex.in --nb 120 > kinion.out 2>&1
[ -s kin_ion.h5 ] || die "kin_ion"
echo "[sbatch] GW $(date)"
srun -N 4 -n 16 --gres=gpu:4 $SEL $SHIFTER $INC python3 -u -m gw.gw_jax -i $RUN/cohsex.in > gw.out 2>&1
[ -s eqp0.dat ] && [ -s sigma_mnk.h5 ] || die "gw_jax"
echo "[sbatch] CORE DONE $(date)"

# ---------- 4. DOWNSTREAM (best-effort) ----------
srun -N 1 -n 1 --gres=gpu:1 $SEL $SHIFTER $INC python3 -u -m bandstructure.htransform -i $RUN/cohsex_bands.in > bands_dft.out 2>&1 && mv -f bandstructure.dat bandstructure_dft.dat || true
srun -N 1 -n 1 --gres=gpu:1 $SEL $SHIFTER $INC python3 -u -m bandstructure.htransform -i $RUN/cohsex_bands.in --eqp-file eqp0.dat > bands_gw.out 2>&1 && mv -f bandstructure.dat bandstructure_gw.dat || true
cd $RUN/qe/nscf
for NB in 100 200 300 400 500 600; do
  srun -N 1 -n 1 --gres=gpu:1 $SEL $SHIFTER_OM $INC python3 -u -m psp.orbital_magnetization --wfn WFN.h5 --nbnd $NB --mu-scan > $RUN/orbmag_nb${NB}.out 2>&1 || true
done
cd $RUN
srun -N 1 -n 1 $SEL $SHIFTER $INC python3 -u plot_bands.py CrI3 bandstructure_dft.dat bandstructure_gw.dat > plot.out 2>&1 || true
grep -h "m_z" orbmag_nb*.out 2>/dev/null > orbmag_sweep_summary.txt || true
echo "[sbatch] ALL DONE $(date)"
