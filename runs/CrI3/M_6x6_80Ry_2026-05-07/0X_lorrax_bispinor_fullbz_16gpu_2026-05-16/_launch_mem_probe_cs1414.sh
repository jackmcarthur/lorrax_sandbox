#!/bin/bash
# jax.live_arrays() probe at r-chunk boundaries — cs=1414 (OOM regime).
# Uses lorrax_B (agent/bispinor-ibz with mem_probe instrumentation).
# Expected to OOM in accumulate, but probe at after_fit_one_rchunk should fire first.

cd "$(dirname "$0")"

module purge || true
module load lorrax_B
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent

set -u

# Use the other 4-node hbm80g allocation.
export SLURM_JOBID=53075110

# 16-GPU mandatory for CrI3 production.
export LORRAX_NNODES=4
export LORRAX_NGPU=4   # per-node; total = 16 ranks

export LORRAX_SHIFTER_OVERRIDE="$LORRAX_SHIFTER \
  --env=LORRAX_MEM_DEBUG=1 \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_EXIT_AFTER_ZETA=1 \
  --env=LORRAX_MAX_RCHUNKS=2 \
  --env=LORRAX_RCHUNK_DEBUG=1"

LORRAX_SHIFTER="$LORRAX_SHIFTER_OVERRIDE" \
  lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex_mem_probe_cs1414.in" 2>&1 \
  | tee mem_probe_cs1414.out
