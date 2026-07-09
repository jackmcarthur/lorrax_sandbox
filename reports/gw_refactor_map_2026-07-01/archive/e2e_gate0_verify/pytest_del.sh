#!/bin/bash
export LORRAX_FFI_NVHPC_DIR=/global/homes/j/jackm/software/lorrax_nvhpc
export LORRAX_FFI_SLATE_DIR=/global/homes/j/jackm/software/lorrax_slate_cray/stage
export LORRAX_FFI_PHDF5_DIR=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
module use /global/homes/j/jackm/modulefiles
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55385913
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
git checkout -q agent/gw-delete-pass
echo "=== branch $(git branch --show-current) — unit suite (excl. slow e2e) ==="
LORRAX_NGPU=1 lxrun python3 -m pytest -q tests -m "not regression" -p no:cacheprovider 2>&1 | tail -8
git checkout -q agent/gate-0-qpwfn
