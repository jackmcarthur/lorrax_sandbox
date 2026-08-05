#!/usr/bin/env bash
# Round 8 validation: CrI3 6x6 80 Ry on lorrax_B commit c796420
# (Round-7 symmetric back-pad fix on top of f567aa0 scan-inside-shard_map kernel).
# One run captures both gates:
#   G2 — HLO acceptance vs Round-8 §4 predictions (≤ 15 GiB preallocated-temp).
#   G3 — fit_zeta completes 16 r-chunks + remainder, 0 remat, 0 RESOURCE_EXHAUSTED.
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
DUMP_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round8_validation_2026-05-14

module use $SBOX/modulefiles
module load lorrax_B
module load lorrax_agent
lxattach

cd $DUMP_DIR
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=XLA_FLAGS=--xla_dump_to=$DUMP_DIR/xla_dump --env=XLA_DUMP_HLO_AS_TEXT=true"

echo "[run] SLURM_JOBID=${SLURM_JOBID:-unset}  cwd=$(pwd)"
echo "[run] commit: $(cd $SBOX/sources/lorrax_B && git rev-parse --short HEAD)"
echo "[run] launching gw.gw_jax on lorrax_B (Round 8 validation) at $(date)"
LORRAX_NNODES=4 LORRAX_IMMEDIATE=60 lxrun \
    python3 -u -m gw.gw_jax -i $DUMP_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
