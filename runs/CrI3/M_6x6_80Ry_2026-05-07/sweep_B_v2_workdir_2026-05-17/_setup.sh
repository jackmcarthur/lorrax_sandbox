#!/bin/bash
# Source this to set up modules + lxrun for sweep B v2 on JID $JID_B.
# $JID_B must be set in the environment before sourcing.
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles 2>/dev/null
module load lorrax_B lorrax_agent 2>/dev/null
export SLURM_JOBID=${JID_B:?must export JID_B before sourcing _setup.sh}
export LORRAX_NNODES=4
export LORRAX_NGPU=4
# Override LORRAX_SHIFTER to enable full-BZ flag
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=LORRAX_FORCE_FULL_BZ=1"
