#!/bin/bash -l
# Agent O — allocator audit launcher.
# Usage: _launch_agent_o.sh <sandbox_dir> <slurm_jobid> <out_file> <variant>
#   variant ∈ { y1 | y2 | y3_50 | y3_95 }
#
#   y1     XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false   (baseline; reproduces Round-7 X3)
#   y2     XLA_PYTHON_CLIENT_ALLOCATOR=default  XLA_PYTHON_CLIENT_PREALLOCATE=false   (BFC, no-preallocate)
#   y3_50  XLA_PYTHON_CLIENT_ALLOCATOR=default  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
#   y3_95  XLA_PYTHON_CLIENT_ALLOCATOR=default  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
#
# All variants use the same Round-7 probe envs (zeta-only, 3 r-chunks, full-BZ)
# and the same X3 sweet-spot cohsex.in (r=24576, b=32, cs=100).

set -u

SB="${1:?sandbox dir}"
JID="${2:?slurm jobid}"
OUT="${3:?output file}"
VAR="${4:?variant: y1|y2|y3_50|y3_95}"

cd "$SB"

module purge 2>&1 >/dev/null || true
module load lorrax_B 2>&1 >/dev/null
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent 2>&1 >/dev/null

export SLURM_JOBID="$JID"
export LORRAX_NNODES=4
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

case "$VAR" in
    y1)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=false"
        ;;
    y2)
        # BFC, no-preallocate; unset MEM_FRACTION explicitly (in case it was set)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=false"
        ;;
    y3_50)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=true \
            --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.50"
        ;;
    y3_95)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=true \
            --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
        ;;
    *)
        echo "Unknown variant: $VAR" >&2
        exit 1
        ;;
esac

PROBE_ENVS="--env=LORRAX_MEM_DEBUG=1 \
    --env=LORRAX_FORCE_FULL_BZ=1 \
    --env=LORRAX_RCHUNK_DEBUG=1 \
    --env=LORRAX_MAX_RCHUNKS=3 \
    --env=LORRAX_EXIT_AFTER_ZETA=1"

# The module sets LORRAX_SHIFTER with platform-allocator --env= flags first.
# Per Shifter, *later* --env= flags override earlier ones, so appending our
# allocator overrides at the end wins for variants y2/y3*.
LORRAX_SHIFTER="$LORRAX_SHIFTER $PROBE_ENVS $ALLOC_ENVS" \
    lxrun python3 -u /pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/_print_env_then_gw.py \
    -i "$PWD/cohsex.in" 2>&1 \
    | tee "$OUT"
