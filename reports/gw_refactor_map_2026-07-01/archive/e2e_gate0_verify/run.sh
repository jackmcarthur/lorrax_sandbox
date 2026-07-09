#!/bin/bash
set -e
export LORRAX_FFI_NVHPC_DIR=/global/homes/j/jackm/software/lorrax_nvhpc
export LORRAX_FFI_SLATE_DIR=/global/homes/j/jackm/software/lorrax_slate_cray/stage
export LORRAX_FFI_PHDF5_DIR=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55385913
cd /pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_refactor_map_2026-07-01/e2e_gate0_verify
echo "=== host $(hostname) jobid $SLURM_JOBID src $LORRAX_SRC ==="
LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i cohsex_test.in
echo "=== EXIT $? ==="
