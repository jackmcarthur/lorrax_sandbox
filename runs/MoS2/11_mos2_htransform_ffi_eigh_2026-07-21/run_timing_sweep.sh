#!/bin/bash
# CLEAN per-q timing: one run at a time, one node, nothing else of ours running.
# The allocation is a shared pool, and a concurrent step inflates ms/q by 2-3x
# (measured: cusolverMp at rank 1728 read 8533 ms/q contended vs ~3000 quiet),
# so every number quoted for timing has to come from a serialized sweep.
#
#   usage: JID=<jid> [NODE_OFFSET=0] ./run_timing_sweep.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/11_mos2_htransform_ffi_eigh_2026-07-21
NODE_OFFSET="${NODE_OFFSET:-0}"

for W in "8,4" "26,8"; do
  for B in off cusolvermp slate; do
    T="t_${B}_$(echo "$W" | tr ',' 'v')"
    JID=$JID NNODES=1 PX=2 PY=2 BACKEND=$B ALLOC=async NCOND=2 \
      NODE_OFFSET=$NODE_OFFSET KSTRIDE=144 WINDOWS="$W" TAG="$T" TRIES=2 \
      WD=$R/02_timing "$R/run_gate.sh" > "$R/02_timing/$T.out" 2>&1
    grep -aE "^\[window" "$R/02_timing/$T.out" | tail -1
  done
done
echo "=== timing sweep done $(date +%T)"
