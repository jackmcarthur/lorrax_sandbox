#!/usr/bin/env bash
# Phase 2: end-to-end CrI3 6x6 80 Ry on lorrax_B with orbit-aware centroids,
# bigger chunks, HLO dump on. Predicted IBZ cascade activation (n_q_ibz~6-9
# instead of 36) + structural fix savings → ~5-10× speedup vs the round-8
# full-BZ baseline.
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
RUN_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_2026-05-14

module use $SBOX/modulefiles
module load lorrax_B
module load lorrax_agent
lxattach

cd $RUN_DIR
mkdir -p xla_dump

export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=XLA_FLAGS=--xla_dump_to=$RUN_DIR/xla_dump --env=XLA_DUMP_HLO_AS_TEXT=true"

echo "[run] SLURM_JOBID=${SLURM_JOBID:-unset}  cwd=$(pwd)"
echo "[run] commit: $(cd $SBOX/sources/lorrax_B && git rev-parse --short HEAD)"
echo "[run] launching gw.gw_jax (IBZ cascade validation) at $(date)"
LORRAX_NNODES=4 lxrun \
    python3 -u -m gw.gw_jax -i $RUN_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
