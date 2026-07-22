#!/bin/bash
# fH_q eigh backend sweep: memory HWM + the on-grid/off-grid correctness gate.
#
#   usage: JID=<jid> [NNODES=1] [PX=2] [PY=2] [BACKEND=off] [ALLOC=bfc]
#          [WINDOWS="26,8"] [NCOND=8] [KSTRIDE=24] [TAG=x] [TRIES=3] ./run_gate.sh
#
# NNODES/PX/PY: one JAX PROCESS PER GPU always (NTASKS = 4*NNODES = PX*PY).
#   1 node -> 2x2 (4 ranks), 4 nodes -> 4x4 (16 ranks).  Both FFI eigh wrappers
#   need a SQUARE process mesh, so 2 nodes (8 ranks) is not a usable geometry.
#
# ALLOC=async is REQUIRED for BACKEND=cusolvermp|slate: with the BFC pool at
#   MEM_FRACTION=0.95 the NCCL/CAL side starves and cusolverMpSyevd reports
#   "NCCL error 1 unhandled cuda error" (config/modulefiles/lorrax/*.lua).
#   Under that allocator device.memory_stats() has no peak, so pass ARENA=nvml
#   to get the high-water mark from NVML instead.
#
# TRIES: the allocation is a SHARED pool (agents A-D attach to the same salloc)
#   and a concurrent 16-rank step can host-OOM this one — a bare SIGKILL with no
#   Python traceback.  Retry rather than mis-record a collision as an OOM
#   ceiling; a REAL device OOM comes back through Python as RESOURCE_EXHAUSTED
#   and is recorded in the JSON's "failures" block instead.
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/11_mos2_htransform_ffi_eigh_2026-07-21
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/htransform_distributed_eigh_2026-07-21
NNODES="${NNODES:-1}"; PX="${PX:-2}"; PY="${PY:-2}"
BACKEND="${BACKEND:-off}"; ALLOC="${ALLOC:-bfc}"
WINDOWS="${WINDOWS:-26,8}"; NCOND="${NCOND:-8}"; KSTRIDE="${KSTRIDE:-24}"
TAG="${TAG:-${BACKEND}}"; TRIES="${TRIES:-3}"; ARENA="${ARENA:-off}"
WD="${WD:-$R/00_baseline}"
NTASKS=$((NNODES * 4))

# ONE python invocation per window.  A device OOM in an SPMD run desyncs the
# ranks — some raise RESOURCE_EXHAUSTED, others sit in a collective until srun
# SIGKILLs the step — so it cannot be caught and resumed inside one process.
# One window per process makes the OOM terminal for that window only.
rc=0
IFS=';' read -ra WLIST <<< "$WINDOWS"
for W in "${WLIST[@]}"; do
  WT=$(echo "$W" | tr ',' 'v')
  for try in $(seq 1 "$TRIES"); do
    echo "=== gate $TAG w=$W try $try/$TRIES  $(date +%T)  backend=$BACKEND mesh=${PX}x${PY}"
    JID=$JID NNODES=$NNODES NTASKS=$NTASKS GRES=4 ALLOC=$ALLOC \
      EXTRA_ENV="--env=LORRAX_SKIP_VQ_GATES=1" "$R/run_shifter.sh" "$WD" \
      python3 -u "$REP/eigh_backend_gate.py" -i exciton.in --eqp eqp1.dat \
        --px "$PX" --py "$PY" --windows "$W" --n-cond "$NCOND" \
        --a-band auto --k-stride "$KSTRIDE" --eigh-backend "$BACKEND" \
        --arena-source "$ARENA" --out "$WD/gate_${TAG}_${WT}" \
        > "$WD/run_gate_${TAG}_${WT}.log" 2>&1
    rc=$?
    echo "=== gate $TAG w=$W try $try rc=$rc $(date +%T)"
    # The XLA module peak is a STATIC property of the compiled program, so it
    # is quotable even from a run that a co-tenant's VRAM pushed into OOM.
    grep -ao "Can.t reduce memory use below [0-9.]*GiB" "$WD/run_gate_${TAG}_${WT}.log" | tail -1
    grep -aE "^\[gpu|^\[window|^    (M-Gamma|Gamma-K)|^    \[mem|FAILED|fH_q eigh|SVD of" \
         "$WD/run_gate_${TAG}_${WT}.log" | tail -12
    if [ "$rc" -eq 0 ]; then break; fi
    grep -aE "Killed|Aborted|RESOURCE_EXHAUSTED: Out of memory" "$WD/run_gate_${TAG}_${WT}.log" | tail -2
    sleep 15
  done
done
exit 0
