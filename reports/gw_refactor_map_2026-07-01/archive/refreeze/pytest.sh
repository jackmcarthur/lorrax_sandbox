#!/bin/bash
export LORRAX_FFI_NVHPC_DIR=/global/homes/j/jackm/software/lorrax_nvhpc
export LORRAX_FFI_SLATE_DIR=/global/homes/j/jackm/software/lorrax_slate_cray/stage
export LORRAX_FFI_PHDF5_DIR=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55401475
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
echo "=== regression gate vs re-frozen eqp_ref.dat ==="
LORRAX_NGPU=1 lxrun python3 -m pytest -q tests/test_gw_jax_regression.py -o addopts="" 2>&1 | tail -15
