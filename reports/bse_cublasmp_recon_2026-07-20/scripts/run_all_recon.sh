#!/bin/bash
# Sequential 16-GPU (4x4) validation driver for the 2-D-distributed cuBLASMp
# V_Q reconstruction.  Runs (a) the n_mu=640 bit-match, (b) the large-n_mu
# capability + replicated-OOM proof, (c) the full exciton bands via the
# distributed-recon path vs the dir10 reference.  Sequential (no GPU-memory
# contention); each step re-invokes nothing — logs land in ../logs.
#   usage: JID=<jid> ./run_all_recon.sh
set -uo pipefail
JID="${JID:?set JID}"
RD="$(cd "$(dirname "$0")" && pwd)"
LOGS="$RD/../logs"
DIR11=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/04_mos2_12x12_bands_2026-07-18/11_lorrax_exciton_bands_minibz_16gpu
RUN11="$DIR11/run11.sh"
REF="$DIR11/ref_dir10_40interp_8v8c.dat"
mkdir -p "$LOGS"

echo "=== [1/4] n_mu=640 bit-match (distributed vs replicated recon) $(date) ==="
JID="$JID" bash "$RUN11" "$RD" python3 -u recon_bitmatch_640.py \
  > "$LOGS/bitmatch_640.log" 2>&1
echo "  bitmatch rc=$? ; tail:"; grep -aE "relF|PASS|FAIL|n_mu|saved" "$LOGS/bitmatch_640.log" | tail -8

echo "=== [2/4] capability + replicated-OOM proof $(date) ==="
JID="$JID" bash "$RUN11" "$RD" python3 -u recon_capability.py 640 4096 16384 32768 \
  > "$LOGS/capability.log" 2>&1
echo "  capability rc=$? ; tail:"; grep -aE "n_mu|per-proc|identity|reconstruct|REPLICATED|->|time:|saved" "$LOGS/capability.log" | tail -40

echo "=== [3/4] full exciton bands via distributed-recon (16 GPU) $(date) ==="
JID="$JID" bash "$RUN11" "$DIR11" \
  python3 -u -m bse.exciton_bands -i exciton_40_8v8c.in \
    --n-val 8 --n-cond 8 --n-eig 8 --block-size 8 --max-iter 40 \
    --vq-mode interp --eigh-backend cusolvermp --a-band 33 \
    --px 4 --py 4 --distributed-recon on \
    --out-prefix exciton_bands_16gpu_drecon \
  > "$LOGS/exciton_drecon.log" 2>&1
echo "  exciton rc=$? ; tail:"; grep -aE "^\[dist\]|distributed-recon|vq_prepare|cold |warm |TOTAL|Wrote|Error|Traceback|AssertionError" "$LOGS/exciton_drecon.log" | tail -25

echo "=== [4/4] compare distributed-recon exciton bands vs dir10 $(date) ==="
DRDAT="$DIR11/exciton_bands_16gpu_drecon.dat"
if [ -f "$DRDAT" ] && [ -f "$REF" ]; then
  JID="$JID" bash "$DIR11/run_1gpu.sh" python3 -u "$RD/../../bse_multinode_2026-07-20/scripts/compare_dat.py" \
    "$REF" "$DRDAT" --label-a dir10 --label-b drecon \
    > "$LOGS/compare_drecon_vs_dir10.log" 2>&1
  echo "  compare rc=$? ; tail:"; tail -15 "$LOGS/compare_drecon_vs_dir10.log"
else
  echo "  MISSING: DRDAT=$DRDAT ($( [ -f "$DRDAT" ] && echo ok || echo no )) REF=$REF ($( [ -f "$REF" ] && echo ok || echo no ))"
fi
echo "=== ALL DONE $(date) ==="
