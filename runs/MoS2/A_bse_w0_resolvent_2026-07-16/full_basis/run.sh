#!/bin/bash
# Full-basis W_q resolvent, all IBZ q, 1 GPU. Module-free srun+shifter via the
# fixture_run's lxrun_free.sh (PYTHONPATH=lorrax_A/src). Owner request 2026-07-17.
#   usage: JID=<free lx-alloc-jackm node> ./run.sh
set -euo pipefail
JID="${JID:?set JID to the free salloc job id}"
RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16
FIX="$RUN/fixture_run"
SELF="$RUN/full_basis"
MAX_ITER="${MAX_ITER:-300}"
TOL="${TOL:-1e-10}"

JID="$JID" "$RUN/lxrun_free.sh" "$SELF" \
  python3 -u "$SELF/full_basis_wq.py" "$FIX/gnppm_test.in" "$MAX_ITER" "$TOL" \
  2>&1 | tee "$SELF/full_basis_wq.log"
