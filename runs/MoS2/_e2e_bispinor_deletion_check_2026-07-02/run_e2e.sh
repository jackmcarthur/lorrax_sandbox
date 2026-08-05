#!/bin/bash
source /etc/profile.d/zzz-lmod.sh
module use /global/homes/j/jackm/modulefiles
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D
module load lorrax_agent
export SLURM_JOBID=55417809
WORK=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/_e2e_bispinor_deletion_check_2026-07-02
cd $WORK
echo "=== HOST $(hostname) START $(date) ==="
LORRAX_NGPU=4 lxrun python3 -u -m gw.gw_jax -i cohsex.in
echo "=== EXIT ${PIPESTATUS[0]} END $(date) ==="
