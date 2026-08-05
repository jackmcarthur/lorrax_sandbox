#!/bin/bash
# Source this to set up modules and lxrun shell function for the sweep.
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles 2>/dev/null
module load lorrax_B lorrax_agent 2>/dev/null
export SLURM_JOBID=53058291
export LORRAX_NNODES=4
export LORRAX_NGPU=4
# Override LORRAX_SHIFTER to enable full-BZ flag
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=LORRAX_FORCE_FULL_BZ=1"
