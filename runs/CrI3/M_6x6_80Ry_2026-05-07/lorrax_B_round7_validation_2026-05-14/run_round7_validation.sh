#!/usr/bin/env bash
# Round 7 G2 + G3 validation: CrI3 6x6 80 Ry HLO dump + end-to-end on lorrax_B.
# Exercises commit c796420 (Round-7 symmetric back-pad fix) on top of f567aa0
# (Round-6 scan-inside-shard_map rewrite).  One run serves two gates:
#   - G2 (Agent 3) — HLO acceptance vs round5_unified_plan.md §4 + §9.1 amended.
#   - G3 (Agent 4) — fit_zeta completes 16 r-chunks + remainder, 0 remat,
#                    0 RESOURCE_EXHAUSTED, total preallocated-temp ≤ 17 GiB.
#
# Baselines for comparison:
#   - lorrax_A_hlo_dump_2026-05-13   — 200 GiB OOM pre-Path-D
#   - lorrax_B_path_d_hlo_2026-05-13 — 48.63 GiB Round-4 (pre-Round-6)
#   - lorrax_B_round6_hlo_2026-05-13 — empty; superseded by this run
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
DUMP_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round7_validation_2026-05-14

module use $SBOX/modulefiles
module load lorrax_B
module load lorrax_agent
lxattach

cd $DUMP_DIR
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=XLA_FLAGS=--xla_dump_to=$DUMP_DIR/xla_dump --env=XLA_DUMP_HLO_AS_TEXT=true"

echo "[run] SLURM_JOBID=${SLURM_JOBID:-unset}  cwd=$(pwd)"
echo "[run] commit: $(cd $SBOX/sources/lorrax_B && git rev-parse --short HEAD)"
echo "[run] launching gw.gw_jax on lorrax_B (Round 7 kernel) at $(date)"
LORRAX_NNODES=4 LORRAX_IMMEDIATE=60 lxrun \
    python3 -u -m gw.gw_jax -i $DUMP_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
