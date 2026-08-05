#!/bin/bash
export LORRAX_FFI_NVHPC_DIR=/global/homes/j/jackm/software/lorrax_nvhpc
export LORRAX_FFI_SLATE_DIR=/global/homes/j/jackm/software/lorrax_slate_cray/stage
export LORRAX_FFI_PHDF5_DIR=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55399936
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/04_lorrax_cohsex_vhead
echo "=== vhead run, jid $SLURM_JOBID, $(date) ==="
lxrun python3 -u -m gw.gw_jax -i cohsex.in
echo "=== EXIT $? ==="
