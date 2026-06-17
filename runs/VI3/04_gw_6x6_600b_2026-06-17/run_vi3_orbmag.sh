#!/bin/bash -l
set -u
export SLURM_JOBID=54622523
J="--jobid=$SLURM_JOBID --overlap"
RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/VI3/04_gw_6x6_600b_2026-06-17
OMWT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B_orbmag_wt
LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
SEL=$LROOT/src/ffi/common/cpp/select_gpu.sh; INC=$LROOT/src/ffi/common/cpp/in_container.sh
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
VOL="--volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc --volume=/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/pscratch/sd/j/jackm/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=MPICH_GPU_SUPPORT_ENABLED=1"
SHIFTER_OM="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=JAX_ENABLE_X64=1 $LDP --env=PYTHONPATH=$OMWT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
cd $RUN/qe/nscf
for NB in 100 200 300 400 500 600; do
  echo "[orbmag] nbnd=$NB $(date '+%H:%M')"
  srun $J -N1 -n1 --gres=gpu:1 $SEL $SHIFTER_OM $INC python3 -u -m psp.orbital_magnetization --wfn WFN.h5 --nbnd $NB --mu-scan > $RUN/orbmag_nb${NB}.out 2>&1 || echo "  nb=$NB failed"
done
echo "[orbmag] sweep done $(date '+%H:%M')"
grep -h "m_z" $RUN/orbmag_nb*.out 2>/dev/null
