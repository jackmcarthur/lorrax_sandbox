#!/usr/bin/env bash
# 2nd HLO dump: psig_k_chunk_size=1, validates band-FFT linearity
# Same allocation as the 1st dump (lxattach picks up lx-alloc-jackm).
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
DUMP_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_k1_2026-05-13

module use $SBOX/modulefiles
module load lorrax_A
module load lorrax_agent
lxattach

cd $DUMP_DIR
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=XLA_FLAGS=--xla_dump_to=$DUMP_DIR/xla_dump --env=XLA_DUMP_HLO_AS_TEXT=true"

echo "[run] SLURM_JOBID=$SLURM_JOBID  cwd=$(pwd)"
echo "[run] launching gw.gw_jax (psig_k_chunk=1) at $(date)"
LORRAX_NNODES=4 LORRAX_IMMEDIATE=60 lxrun \
    python3 -u -m gw.gw_jax -i $DUMP_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
