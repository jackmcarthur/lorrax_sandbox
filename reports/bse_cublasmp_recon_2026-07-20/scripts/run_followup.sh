#!/bin/bash
# Follow-up 16-GPU runs: (a) the N_mu^2-distribution shard audit, (b) the fixed
# n_mu=640 bit-match on the nq=144 12x12 fixture.  Run AFTER run_all_recon.sh.
#   usage: JID=<jid> ./run_followup.sh
set -uo pipefail
JID="${JID:?set JID}"
RD="$(cd "$(dirname "$0")" && pwd)"
LOGS="$RD/../logs"
DIR11=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/04_mos2_12x12_bands_2026-07-18/11_lorrax_exciton_bands_minibz_16gpu
RUN11="$DIR11/run11.sh"
mkdir -p "$LOGS"

echo "=== [A] N_mu^2-distribution shard audit $(date) ==="
JID="$JID" bash "$RUN11" "$RD" python3 -u recon_shard_audit.py \
  > "$LOGS/shard_audit.log" 2>&1
echo "  audit rc=$? ; table:"; grep -aE "tensor|Qraw|S |Sc|A_ref|A_lr|V_delta|T1|V_SRc|zt|VERDICT|shard/proc|allowed|lam |g " "$LOGS/shard_audit.log" | grep -avE "nvshmem" | tail -30

echo "=== [B] n_mu=640 bit-match (12x12 nq=144 fixture) $(date) ==="
JID="$JID" bash "$RUN11" "$RD" python3 -u recon_bitmatch_640.py \
  > "$LOGS/bitmatch_640.log" 2>&1
echo "  bitmatch rc=$? ; tail:"; grep -aE "relF|PASS|FAIL|n_mu|dist\]|saved" "$LOGS/bitmatch_640.log" | grep -avE "nvshmem" | tail -10

echo "=== [C] golden gates + new distributed-recon test (1 GPU) $(date) ==="
JID="$JID" bash "$DIR11/run_tests_1gpu.sh" \
  tests/test_gw_jax_regression.py tests/test_symmetry_unfold.py \
  tests/test_bse_vq_interp.py tests/test_minibz_average.py \
  tests/test_bse_vq_recon_distributed.py \
  > "$LOGS/golden_gates_1gpu.log" 2>&1
echo "  gates rc=$? ; tail:"; tail -6 "$LOGS/golden_gates_1gpu.log"
echo "=== FOLLOWUP DONE $(date) ==="
