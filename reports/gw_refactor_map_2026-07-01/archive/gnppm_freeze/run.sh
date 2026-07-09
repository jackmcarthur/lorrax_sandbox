#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55404113
cd /pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_refactor_map_2026-07-01/gnppm_freeze
echo "=== GN-PPM freeze run (1 GPU), $(date) ==="
LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i gnppm_test.in
echo "=== EXIT $? ==="
