#!/bin/bash
# BSE exciton-bandstructure driver (single-compile lax.scan over the Q-path).
# 12x12 via bse_k_grid=12 12 1 in exciton.in.  16 GPU / 4 node (--px 4 --py 4,
# square mesh for cusolverMp).  Uses the run-root run_shifter.sh.
#   usage: JID=<jid> [NGPU_MODE=16|1] [EIGH=cusolvermp|off] ./run_exciton.sh
set -uo pipefail
JID="${JID:?set JID}"
MODE="${NGPU_MODE:-16}"; EIGH="${EIGH:-cusolvermp}"
RD="$(cd "$(dirname "$0")" && pwd)"; RUN="$(cd "$RD/.." && pwd)"
SH="$RUN/run_shifter.sh"

if [ "$MODE" = 16 ]; then NN=4; NT=16; PX=4; PY=4
elif [ "$MODE" = 4 ]; then NN=1; NT=4; PX=2; PY=2
else NN=1; NT=1; PX=1; PY=1; EIGH=off; fi

echo "=== exciton_bands ($MODE GPU, eigh=$EIGH) start $(date) ==="
JID=$JID NNODES=$NN NTASKS=$NT GRES=4 \
  EXTRA_ENV="--env=LORRAX_SKIP_VQ_GATES=1" "$SH" "$RD" \
  python3 -u -m bse.exciton_bands -i exciton.in \
    --n-val 8 --n-cond 8 --n-eig 8 --block-size 8 --max-iter 40 \
    --vq-mode interp --eigh-backend "$EIGH" --a-band 30 \
    --px $PX --py $PY --head-minibz-average \
    --out-prefix mos2_exciton_bands > run_exciton.log 2>&1
rc=$?
echo "=== end $(date) rc=$rc ==="
grep -aE "\[bse_k_grid\]|\[gate\]|Q path|htransform|Wrote|SAVED|recon|on-grid|Error|Traceback|Assertion" run_exciton.log | tail -30
exit $rc
