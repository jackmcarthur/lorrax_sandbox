#!/bin/bash -l
# Agent S — Round-11 live verification launcher.
# Usage: _launch_agent_s.sh <sandbox_dir> <slurm_jobid> <out_file> <variant> <cohsex_in>
#   variant ∈ { A1 | A2 | A3 | A3_85 }
#
#   A1     platform alloc + preallocate=false (Round-10 natural pick)
#   A2     platform alloc + preallocate=false (sweet-spot r=24576, b=32, cs=100)
#   A3     default BFC + preallocate=true MEM_FRACTION=0.95 (true peak via memory_stats)
#   A3_85  default BFC + preallocate=true MEM_FRACTION=0.85 (fallback if A3 NCCL OOMs)
#
# All variants use the same Round-7 probe envs (zeta-only, 3 r-chunks, full-BZ).

set -u

SB="${1:?sandbox dir}"
JID="${2:?slurm jobid}"
OUT="${3:?output file}"
VAR="${4:?variant: A1|A2|A3|A3_85}"
COHSEX_IN="${5:?cohsex input file}"

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
    A1|A2)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=false"
        ;;
    A3)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=true \
            --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
        ;;
    A3_85)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=true \
            --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.85"
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
    -i "$COHSEX_IN" 2>&1 \
    | tee "$OUT"
