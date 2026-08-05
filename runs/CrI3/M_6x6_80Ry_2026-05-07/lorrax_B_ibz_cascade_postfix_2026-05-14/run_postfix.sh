#!/usr/bin/env bash
# R2 — Post-fix verification: CrI3 6×6 80 Ry IBZ cascade-ACTIVE with TRS-aware unfold.
# Reference: lorrax_B_round8_validation_2026-05-14 (1504-centroid non-orbit-closed,
# cascade fell back to full-BZ). Same WFN, same input, same nval/nband/sys_dim.
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
RUN_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_postfix_2026-05-14

module use $SBOX/modulefiles
module load lorrax_B
module load lorrax_agent
lxattach

cd $RUN_DIR
echo "[run] SLURM_JOBID=${SLURM_JOBID:-unset}  cwd=$(pwd)"
echo "[run] commit: $(cd $SBOX/sources/lorrax_B && git rev-parse --short HEAD)"
echo "[run] branch: $(cd $SBOX/sources/lorrax_B && git branch --show-current)"
echo "[run] launching gw.gw_jax (R2 post-fix cascade) at $(date)"
LORRAX_NNODES=4 LORRAX_IMMEDIATE=60 lxrun \
    python3 -u -m gw.gw_jax -i $RUN_DIR/cohsex.in 2>&1 \
    | tee gw.out
echo "[run] finished at $(date)"
