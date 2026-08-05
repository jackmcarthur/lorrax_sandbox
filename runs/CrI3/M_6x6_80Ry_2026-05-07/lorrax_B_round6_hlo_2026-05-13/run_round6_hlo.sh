#!/usr/bin/env bash
# CrI3 6x6 80 Ry HLO dump on lorrax_B G2 gate.
# Exercises commit c796420 (Round-7 back-pad fix on top of Round-6 scan-inside-shard_map rewrite f567aa0).
# Compare against:
#   - lorrax_A_hlo_dump_2026-05-13   (the 200 GiB OOMing pre-Path-D baseline)
#   - lorrax_B_path_d_hlo_2026-05-13 (the 48.63 GiB Round 4 / pre-Round-6 baseline)
# Target (per round5_unified_plan.md §9.1 amended): ≤ 17 GiB total preallocated-temp,
# 0 "Involuntary full rematerialization" warnings, 1 while_loop with 1 all-gather inside body.
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
DUMP_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round6_hlo_2026-05-13

module use $SBOX/modulefiles
module load lorrax_B
module load lorrax_agent
lxattach

cd $DUMP_DIR
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=XLA_FLAGS=--xla_dump_to=$DUMP_DIR/xla_dump --env=XLA_DUMP_HLO_AS_TEXT=true"

echo "[run] SLURM_JOBID=$SLURM_JOBID  cwd=$(pwd)"
echo "[run] launching gw.gw_jax on lorrax_B (Round 6 kernel) at $(date)"
LORRAX_NNODES=4 LORRAX_IMMEDIATE=60 lxrun \
    python3 -u -m gw.gw_jax -i $DUMP_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
