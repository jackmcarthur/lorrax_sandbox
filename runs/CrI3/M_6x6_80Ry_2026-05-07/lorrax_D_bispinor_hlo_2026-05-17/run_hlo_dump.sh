#!/usr/bin/env bash
# Launch bispinor 80Ry HLO dump for memory-model calibration (M1, M2).
# Kills the run early via LORRAX_EXIT_AFTER_ZETA + LORRAX_MAX_RCHUNKS=2.
#
# All HLO artifacts land in $PWD/xla_dump/.
set -euo pipefail

JID=${SLURM_JOBID:-53075115}
SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
RUN=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_2026-05-17

# Modules (use lorrax_B because it has LORRAX_EXIT_AFTER_ZETA + MAX_RCHUNKS).
module use $SBOX/modulefiles
module use $HOME/modulefiles
module load lorrax_B
module load lorrax_agent

export SLURM_JOBID=$JID
# Do NOT call lxattach — multiple lx-alloc-$USER allocs are present and the
# pool helper refuses to pick. SLURM_JOBID is already set; lxrun reads it
# directly.

cd "$RUN"

# Inject XLA_FLAGS + early-exit env via shifter passthrough.
export LORRAX_SHIFTER="$LORRAX_SHIFTER \
  --env=XLA_FLAGS=--xla_dump_to=$RUN/xla_dump \
  --env=XLA_DUMP_HLO_AS_TEXT=true \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_EXIT_AFTER_ZETA=1 \
  --env=LORRAX_MAX_RCHUNKS=2 \
  --env=LORRAX_RCHUNK_DEBUG=1"

echo "[run] JID=$SLURM_JOBID  cwd=$(pwd)"
echo "[run] HLO -> $RUN/xla_dump"
echo "[run] launch $(date)"

LORRAX_NNODES=4 lxrun \
    python3 -u -m gw.gw_jax -i $RUN/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished $(date)"
