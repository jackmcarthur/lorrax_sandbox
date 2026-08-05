#!/usr/bin/env bash
# CrI3 6x6 80 Ry HLO dump on lorrax_B (commit 5cadd4b — Path D structural fix).
# Compare against lorrax_A_hlo_dump_2026-05-13 (the 200 GiB OOMing baseline).
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
DUMP_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_path_d_hlo_2026-05-13

module use $SBOX/modulefiles
module load lorrax_B
module load lorrax_agent
lxattach

cd $DUMP_DIR
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=XLA_FLAGS=--xla_dump_to=$DUMP_DIR/xla_dump --env=XLA_DUMP_HLO_AS_TEXT=true"

echo "[run] SLURM_JOBID=$SLURM_JOBID  cwd=$(pwd)"
echo "[run] launching gw.gw_jax on lorrax_B @ 5cadd4b at $(date)"
LORRAX_NNODES=4 LORRAX_IMMEDIATE=60 lxrun \
    python3 -u -m gw.gw_jax -i $DUMP_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
