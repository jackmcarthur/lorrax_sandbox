#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55406527
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
echo "=== planner unit tests + both gates after Phase 2 gflat simplification ==="
LORRAX_NGPU=1 lxrun python3 -m pytest -q \
  tests/test_planner_refit_2026-05-17.py tests/test_band_chunk_size_floor.py \
  tests/test_gw_jax_regression.py -o addopts="" -rA 2>&1 | tail -16
