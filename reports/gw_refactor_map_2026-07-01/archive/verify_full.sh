#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55415890
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
LORRAX_NGPU=1 lxrun python3 -m pytest -q tests -o addopts="" 2>&1 | tail -12
