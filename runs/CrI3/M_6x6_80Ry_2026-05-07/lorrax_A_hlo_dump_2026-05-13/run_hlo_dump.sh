#!/usr/bin/env bash
# Run gw.gw_jax under HLO dump at CrI3 6x6 80 Ry, planner-free at the
# report.md §7 60GB / band_chunk=16 config. Two outputs:
#   (a) gw.out — captures gflat_plan.format() planner pick (step 1)
#   (b) xla_dump/ — captures memory-usage-report.txt for fit_one_rchunk
#                   (resolves consensus.md B-1, B-3, B-4)
#
# This is intended to be killed once both artifacts land; the full fit
# is not needed.
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
DUMP_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13

module use $SBOX/modulefiles
module load lorrax_A
module load lorrax_agent
lxattach

cd $DUMP_DIR

# Inject XLA_FLAGS into the shifter env. Use only buffer-assignment +
# memory-usage-report dumps to keep the volume manageable; HLO text per
# module is still ~MB-class but we don't need *every* pipeline pass.
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=XLA_FLAGS=--xla_dump_to=$DUMP_DIR/xla_dump --env=XLA_DUMP_HLO_AS_TEXT=true"

echo "[run] SLURM_JOBID=$SLURM_JOBID  cwd=$(pwd)"
echo "[run] launching gw.gw_jax with HLO dump at $(date)"
LORRAX_NNODES=4 LORRAX_IMMEDIATE=60 lxrun \
    python3 -u -m gw.gw_jax -i $DUMP_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
