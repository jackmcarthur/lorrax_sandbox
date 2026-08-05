#!/bin/bash
# 16-GPU CrI3 bispinor IBZ gate v2 — run both legs sequentially.
# Prerequisite: lorrax_B + lorrax_agent modules loaded, lxattach done.
#
# Run A first (LORRAX_FORCE_FULL_BZ=1), then Run B (cascade auto).
# We propagate LORRAX_FORCE_FULL_BZ INTO shifter by extending LORRAX_SHIFTER
# (set by the lorrax_B module) with an extra --env= flag for run A only.
set -uo pipefail

BISP_DIR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14
A_DIR=$BISP_DIR/03_lorrax_bispinor_fullbz_16gpu_2026-05-16
B_DIR=$BISP_DIR/04_lorrax_bispinor_ibz_16gpu_2026-05-16

if [ -z "${SLURM_JOBID:-}" ]; then
    echo "SLURM_JOBID not set; run lxattach first." >&2; exit 1
fi
if [ -z "${LORRAX_SHIFTER:-}" ]; then
    echo "LORRAX_SHIFTER not set; load lorrax_B module first." >&2; exit 1
fi

BASE_SHIFTER="$LORRAX_SHIFTER"

# -- Run A: full-BZ reference ---------------------------------------
echo "=== Run A (full-BZ) start $(date) ==="
cd "$A_DIR"
LORRAX_SHIFTER="$BASE_SHIFTER --env=LORRAX_FORCE_FULL_BZ=1" \
LORRAX_NNODES=4 LORRAX_NGPU=4 \
    lxrun python3 -u -m gw.gw_jax -i "$A_DIR/cohsex.in" \
    2>&1 | tee "$A_DIR/gw.out"
A_EXIT=${PIPESTATUS[0]}
echo "=== Run A exit $A_EXIT  end $(date) ==="
if [ "$A_EXIT" -ne 0 ]; then
    echo "Run A FAILED — aborting before Run B." >&2
    exit 2
fi

# -- Run B: IBZ-cascade --------------------------------------------
echo "=== Run B (IBZ cascade) start $(date) ==="
cd "$B_DIR"
LORRAX_SHIFTER="$BASE_SHIFTER" \
LORRAX_NNODES=4 LORRAX_NGPU=4 \
    lxrun python3 -u -m gw.gw_jax -i "$B_DIR/cohsex.in" \
    2>&1 | tee "$B_DIR/gw.out"
B_EXIT=${PIPESTATUS[0]}
echo "=== Run B exit $B_EXIT  end $(date) ==="
exit $B_EXIT
