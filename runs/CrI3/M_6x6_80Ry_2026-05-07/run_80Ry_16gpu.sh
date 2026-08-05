#!/bin/bash
# 16-GPU CrI3 80 Ry bispinor IBZ gate — run both legs sequentially.
# Prerequisite: lorrax_B + lorrax_agent modules loaded, lxattach done,
# SLURM_JOBID exported, transverse centroids generated.
#
# Run A: LORRAX_FORCE_FULL_BZ=1 (forces cascade off — full-BZ reference)
# Run B: LORRAX_FORCE_FULL_BZ unset (cascade auto-activates)
set -uo pipefail

PARENT=/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07
A_DIR=$PARENT/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16
B_DIR=$PARENT/0Y_lorrax_bispinor_ibz_16gpu_2026-05-16

if [ -z "${SLURM_JOBID:-}" ]; then
    echo "SLURM_JOBID not set; run lxattach first." >&2; exit 1
fi
if [ -z "${LORRAX_SHIFTER:-}" ]; then
    echo "LORRAX_SHIFTER not set; load lorrax_B module first." >&2; exit 1
fi

BASE_SHIFTER="$LORRAX_SHIFTER"

# -- Run A: full-BZ reference ---------------------------------------
echo "=== Run A (80Ry full-BZ) start $(date) ==="
cd "$A_DIR"
LORRAX_SHIFTER="$BASE_SHIFTER --env=LORRAX_FORCE_FULL_BZ=1" \
LORRAX_NNODES=4 LORRAX_NGPU=4 \
    lxrun python3 -u -m gw.gw_jax -i "$A_DIR/cohsex.in" \
    2>&1 | tee "$A_DIR/gw.out"
A_EXIT=${PIPESTATUS[0]}
echo "=== Run A exit $A_EXIT  end $(date) ==="
# A_EXIT may be nonzero due to post-Σ qp_wfn write crash; V_q tensor is
# already on disk by then, so we proceed to Run B regardless if V_q wrote.
if [ ! -s "$A_DIR/tmp/v_q_bispinor.h5" ]; then
    echo "Run A FAILED — v_q_bispinor.h5 missing — aborting before Run B." >&2
    exit 2
fi

# -- Run B: IBZ-cascade --------------------------------------------
echo "=== Run B (80Ry IBZ cascade) start $(date) ==="
cd "$B_DIR"
LORRAX_SHIFTER="$BASE_SHIFTER" \
LORRAX_NNODES=4 LORRAX_NGPU=4 \
    lxrun python3 -u -m gw.gw_jax -i "$B_DIR/cohsex.in" \
    2>&1 | tee "$B_DIR/gw.out"
B_EXIT=${PIPESTATUS[0]}
echo "=== Run B exit $B_EXIT  end $(date) ==="
if [ ! -s "$B_DIR/tmp/v_q_bispinor.h5" ]; then
    echo "Run B FAILED — v_q_bispinor.h5 missing." >&2
    exit 3
fi
exit 0
