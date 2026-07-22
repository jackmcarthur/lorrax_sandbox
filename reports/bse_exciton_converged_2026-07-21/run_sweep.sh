#!/bin/bash
# TASK 1 — htransform fH band-window sweep on the converged 12x12 / 80 Ry data.
#
# Task placement: 1 task per NODE (4 nodes, 2x2 mesh).  compute_wfns_fi
# REPLICATES fH_R via jax.device_put, which routes through a HOST gather
# (x._value) — up to 50 GiB per process at nb=40.  4 tasks on one node would
# host-OOM (measured by the parent campaign at the vq_interp/htransform caches).
#   usage: JID=<jid> [WINS=16,20,...] [MODE=4n|1] ./run_sweep.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/08_mos2_exciton_converged_2026-07-21
RD=$R/00_htransform_window_sweep
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_exciton_converged_2026-07-21
SH=$R/run_shifter.sh
WINS="${WINS:-16,20,24,28,32,36,40}"
MODE="${MODE:-4n}"
TAG="${TAG:-main}"
if [ "$MODE" = 4n ]; then NN=4; NT=4; PX=2; PY=2
elif [ "$MODE" = 16 ]; then NN=4; NT=16; PX=4; PY=4
elif [ "$MODE" = 8 ]; then NN=4; NT=8; PX=2; PY=4
else NN=1; NT=1; PX=1; PY=1; fi

# GPU zombies from a previous step masquerade as a phdf5 "negative offset"
# crash — reap before every run (KNOWN_SANDBOX_ERRORS).
srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:1 \
     bash -c 'pkill -9 -u $USER -f window_sweep; pkill -9 -u $USER -f exciton_bands; exit 0' >/dev/null 2>&1
sleep 3

echo "=== window sweep start $(date +%T)  windows=$WINS  ${NT} task(s) (${PX}x${PY})"
: > "$REP/logs/window_sweep_${TAG}.log"
rc=0
# ONE WINDOW PER PROCESS.  compute_wfns_fi replicates fH_R (11-50 GiB per
# device depending on nb); running several windows in one interpreter
# fragments the BFC arena and the second window OOMs on a request the card
# could satisfy cold.  A fresh process per window also makes an OOM at one nb
# a data point instead of an abort of the rest of the sweep.
for nb in ${WINS//,/ }; do
  srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:1 \
       bash -c 'pkill -9 -u $USER -f window_sweep; exit 0' >/dev/null 2>&1
  sleep 2
  echo "--- nb=$nb ---" | tee -a "$REP/logs/window_sweep_${TAG}.log"
  JID=$JID NNODES=$NN NTASKS=$NT GRES=4 \
    EXTRA_ENV="--env=WS_RUN=$RD --env=WS_OUT=${OUT:-$REP} --env=WS_WINDOWS=$nb --env=WS_PX=$PX --env=WS_PY=$PY --env=WS_BATCH=${BATCH:-32}" \
    "$SH" "$RD" python3 -u "$REP/window_sweep.py" >> "$REP/logs/window_sweep_${TAG}.log" 2>&1 || rc=$?
done
echo "=== window sweep end $(date +%T) rc=$rc"
grep -aE "gate max|rank=|REFUSED|raised" "$REP/logs/window_sweep_${TAG}.log" | tail -30
exit 0
