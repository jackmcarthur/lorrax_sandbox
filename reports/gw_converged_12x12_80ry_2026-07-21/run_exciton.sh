#!/bin/bash
# Stage 5 — BSE exciton bandstructure on the converged GW restart.
# NATIVE 12x12 (the GW coarse grid already IS 12x12: no bse_k_grid, no
# coarse->fine interpolation).  16 GPU / 4 node, --px 4 --py 4 (square mesh
# for cusolverMp).  Single-compile lax.scan over the Q path.
#   usage: JID=<jid> [SRCRUN=<gw dir>] [NV=8] [NC=8] ./run_exciton.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21
RD=$R/01_lorrax_exciton_bands
SRCRUN="${SRCRUN:-$R/03_lorrax_gw_6x6_80Ry_bse}"
NV="${NV:-8}"; NC="${NC:-8}"
MODE="${NGPU_MODE:-16}"
# NOTE on task placement: 4 tasks on ONE node host-OOMs (slurm reported
# "Detected 1 oom_kill event ... Some of the step tasks have been OOM Killed"
# at the vq_interp/htransform host caches).  Mode 4n spreads the SAME 2x2 mesh
# over 4 nodes, 1 task each, so every rank gets a whole node's RAM.
if [ "$MODE" = 16 ]; then NN=4; NT=16; PX=4; PY=4
elif [ "$MODE" = 4n ]; then NN=4; NT=4; PX=2; PY=2
elif [ "$MODE" = 4 ]; then NN=1; NT=4; PX=2; PY=2
else NN=1; NT=1; PX=1; PY=1; fi
SH=$R/run_shifter.sh

ln -sfn ../_qe6x6/WFN.h5 "$RD/WFN.h5"
ln -sfn "$SRCRUN/tmp" "$RD/tmp"
cp -f "$SRCRUN/centroids.txt" "$RD/centroids.txt"

srun --jobid=$JID --overlap --immediate=60 -N$NN -n$NN --gres=gpu:1 \
     bash -c 'pkill -9 -u $USER -f exciton_bands; pkill -9 -u $USER -f gw_probe; exit 0' >/dev/null 2>&1
sleep 3

echo "=== exciton_bands start $(date +%T)  restart=$SRCRUN  ${NV}v${NC}c  ${MODE} GPU (${PX}x${PY})"
JID=$JID NNODES=$NN NTASKS=$NT GRES=4 EXTRA_ENV="--env=LORRAX_SKIP_VQ_GATES=1" "$SH" "$RD" \
  python3 -u -m bse.exciton_bands -i exciton.in \
    --n-val $NV --n-cond $NC --n-eig 8 --block-size 8 --max-iter 40 \
    --vq-mode interp --eigh-backend cusolvermp --a-band 30 \
    --px $PX --py $PY --head-minibz-average \
    --out-prefix mos2_exciton_bands > "$RD/run_exciton.log" 2>&1
rc=$?
echo "=== exciton_bands end $(date +%T) rc=$rc"
grep -aE "\[gate\]|Q path|full-band htransform|Wrote|Traceback|Error" "$RD/run_exciton.log" | tail -15
exit $rc
