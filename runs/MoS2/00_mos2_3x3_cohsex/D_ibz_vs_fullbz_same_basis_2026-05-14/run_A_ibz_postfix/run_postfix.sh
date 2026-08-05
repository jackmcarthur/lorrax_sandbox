#!/usr/bin/env bash
# R1 — Post-fix verification: same-basis IBZ cascade-ACTIVE run.
# Expected: bit-equal to ../run_B_fullbz (per Agent 4's verification plan §3.1).
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
RUN_DIR=$SBOX/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_postfix

module use $SBOX/modulefiles
module load lorrax_B
module load lorrax_agent
lxattach

cd $RUN_DIR
echo "[run] SLURM_JOBID=${SLURM_JOBID:-unset}  cwd=$(pwd)"
echo "[run] commit: $(cd $SBOX/sources/lorrax_B && git rev-parse --short HEAD)"
echo "[run] branch: $(cd $SBOX/sources/lorrax_B && git branch --show-current)"
echo "[run] launching gw.gw_jax (R1 post-fix) at $(date)"
LORRAX_NNODES=1 lxrun \
    python3 -u -m gw.gw_jax -i $RUN_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
