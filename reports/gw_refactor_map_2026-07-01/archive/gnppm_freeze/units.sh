#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55404113
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
echo "=== unit suite (excl slow e2e regression) on the mask-fix branch ==="
LORRAX_NGPU=1 lxrun python3 -m pytest -q tests -m "not regression" -p no:cacheprovider 2>&1 | tail -8
