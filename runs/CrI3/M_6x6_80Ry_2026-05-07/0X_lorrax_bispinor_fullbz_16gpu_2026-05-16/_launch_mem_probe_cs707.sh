#!/bin/bash
# jax.live_arrays() probe at r-chunk boundaries — cs=707 (safe regime).
# Uses lorrax_B (agent/bispinor-ibz with mem_probe instrumentation).

cd "$(dirname "$0")"

module purge || true
module load lorrax_B
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent

set -u

# Use one of the two alive 4-node hbm80g allocations.
export SLURM_JOBID=53075115

# 16-GPU mandatory for CrI3 production [[feedback-cri3-always-16-gpus]].
export LORRAX_NNODES=4
export LORRAX_NGPU=4   # per-node; total = 16 ranks

# Shifter env passthrough per KNOWN_SANDBOX_ERRORS 2026-05-16 entries.
export LORRAX_SHIFTER_OVERRIDE="$LORRAX_SHIFTER \
  --env=LORRAX_MEM_DEBUG=1 \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_EXIT_AFTER_ZETA=1 \
  --env=LORRAX_MAX_RCHUNKS=2 \
  --env=LORRAX_RCHUNK_DEBUG=1"

LORRAX_SHIFTER="$LORRAX_SHIFTER_OVERRIDE" \
  lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex_mem_probe_cs707.in" 2>&1 \
  | tee mem_probe_cs707.out
