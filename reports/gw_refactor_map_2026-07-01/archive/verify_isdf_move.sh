#!/bin/bash
source /etc/profile.d/zzz-lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55421620
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
echo "=== import smoke: gw.isdf_fitting resolves? ==="
LORRAX_NGPU=1 lxrun python3 -c "import gw.isdf_fitting; print('import gw.isdf_fitting OK:', gw.isdf_fitting.__file__)" 2>&1 | tail -3
echo "=== full suite + gates (ζ-fit feeds every gate) ==="
LORRAX_NGPU=1 lxrun python3 -m pytest -q tests -o addopts="" 2>&1 | tail -8
