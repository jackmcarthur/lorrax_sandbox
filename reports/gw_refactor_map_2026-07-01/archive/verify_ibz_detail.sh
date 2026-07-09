#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55413970
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
LORRAX_NGPU=1 lxrun python3 -m pytest -q tests/test_gw_jax_regression.py::test_ibz_full_bz_equivalence -o addopts="" 2>&1 | grep -iE "Mismatch|Max absolute|Max relative|abs diff|not close|elements" | head -8
