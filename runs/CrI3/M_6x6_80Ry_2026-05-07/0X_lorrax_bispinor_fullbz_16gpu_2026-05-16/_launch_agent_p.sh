#!/bin/bash -l
# Agent P (Round 9a) — r_chunk OOM boundary search post-Round-4/6 fixes.
# Usage: _launch_agent_p.sh <sandbox_dir> <slurm_jobid> <out_file> <r_chunk> [alloc_variant]
#   r_chunk: integer (e.g. 24576, 28672, 32768, 49152, 98304)
#   alloc_variant ∈ { default | y3_95 }
#     default  XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false  (sandbox default)
#     y3_95    XLA_PYTHON_CLIENT_ALLOCATOR=default  XLA_PYTHON_CLIENT_PREALLOCATE=true MEM_FRACTION=0.95
#
# Writes a per-r cohsex.in inline (r_chunk_size=N, band_chunk_size=32, gflat_chunk_size=100).

set -u

SB="${1:?sandbox dir}"
JID="${2:?slurm jobid}"
OUT="${3:?output file}"
RCH="${4:?r_chunk_size}"
VAR="${5:-default}"

cd "$SB"

module purge 2>&1 >/dev/null || true
module load lorrax_B 2>&1 >/dev/null
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent 2>&1 >/dev/null

export SLURM_JOBID="$JID"
export LORRAX_NNODES=4
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

# Generate a per-r cohsex.in (parallel to the canonical cohsex.in but with overrides)
CFG="$SB/cohsex_p_r${RCH}.in"
sed -e "s/^r_chunk_size *=.*/r_chunk_size = ${RCH}/" \
    -e "s/^band_chunk_size *=.*/band_chunk_size = 32/" \
    -e "s/^gflat_chunk_size *=.*/gflat_chunk_size = 100/" \
    "$SB/cohsex.in" > "$CFG"

case "$VAR" in
    default)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=false"
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

LORRAX_SHIFTER="$LORRAX_SHIFTER $PROBE_ENVS $ALLOC_ENVS" \
    lxrun python3 -u /pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/_print_env_then_gw.py \
    -i "$CFG" 2>&1 \
    | tee "$OUT"
