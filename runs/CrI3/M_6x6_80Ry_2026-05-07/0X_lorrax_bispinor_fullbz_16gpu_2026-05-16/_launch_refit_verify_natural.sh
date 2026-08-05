#!/bin/bash
# Round-2 memory-model refit verification — run with all chunk knobs
# cleared so the NEW planner picks naturally.  LORRAX_MAX_RCHUNKS=3
# limits to 3 r-chunks per channel so we get probe output quickly
# without committing a full run.  LORRAX_EXIT_AFTER_ZETA=1 short-
# circuits after fit_zeta (V_q not needed for HWM comparison).

cd "$(dirname "$0")"

module purge || true
module load lorrax_B
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent

set -u

# Use one of the two alive 4-node hbm80g allocations (both alive ~1h+).
export SLURM_JOBID=53075115

# 16-GPU mandatory for CrI3 production [[feedback-cri3-always-16-gpus]].
export LORRAX_NNODES=4
export LORRAX_NGPU=4   # per-node; total = 16 ranks

export LORRAX_SHIFTER_OVERRIDE="$LORRAX_SHIFTER \
  --env=LORRAX_MEM_DEBUG=1 \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_MAX_RCHUNKS=3 \
  --env=LORRAX_RCHUNK_DEBUG=1 \
  --env=LORRAX_EXIT_AFTER_ZETA=1"

LORRAX_SHIFTER="$LORRAX_SHIFTER_OVERRIDE" \
  lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex_refit_verify_natural.in" 2>&1 \
  | tee refit_verify_natural.out
