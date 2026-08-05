#!/bin/bash
# Reconstruct Σ^B from V_q for run X and run Y, 80 Ry CrI3 6x6 bispinor gate.
# Launch from run X's directory (so prepare_isdf_and_wavefunctions sees the
# correct cohsex.in and centroid files).
set -uo pipefail

PARENT=/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07
A_DIR=$PARENT/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16
B_DIR=$PARENT/0Y_lorrax_bispinor_ibz_16gpu_2026-05-16

cd "$A_DIR"

if [ -z "${SLURM_JOBID:-}" ]; then echo "SLURM_JOBID not set" >&2; exit 1; fi
if [ -z "${LORRAX_SHIFTER:-}" ]; then echo "LORRAX_SHIFTER not set" >&2; exit 1; fi

LORRAX_NNODES=4 LORRAX_NGPU=4 \
    lxrun python3 -u "$PARENT/reconstruct_sigma_b_80Ry.py" \
    -i "$A_DIR/cohsex.in" \
    --v-q-path-x "$A_DIR/tmp/v_q_bispinor.h5" \
    --v-q-path-y "$B_DIR/tmp/v_q_bispinor.h5" \
    --out-prefix "$PARENT/sigma_b_gate_80Ry" \
    2>&1 | tee "$PARENT/recon.out"
echo "exit=${PIPESTATUS[0]}"
