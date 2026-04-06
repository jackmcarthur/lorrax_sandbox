#!/bin/bash
# Environment setup for Si 4x4x4 calculations on Perlmutter
source /opt/cray/pe/lmod/default/init/bash
module load espresso berkeleygw
export JOBID=51036722
export OMP_NUM_THREADS=16
export HDF5_USE_FILE_LOCKING=FALSE
export RUNDIR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band
export NODES=4
export NGPUS=16

# Shifter prefix for LORRAX
export SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
export SHIFTER="shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \
    --env=PYTHONPATH=/global/u2/j/jackm/software/lorrax/src:$SITE \
    --env=JAX_ENABLE_X64=1 \
    --env=HDF5_USE_FILE_LOCKING=FALSE"
