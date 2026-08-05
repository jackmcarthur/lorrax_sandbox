#!/bin/bash -l
# Run gw_jax for a given subdir.
# Usage: _run_gw.sh <subdir> [variant]
#   variant default = "platform_false"  (= production sandbox default)
#   variant = "bfc_pre95"               (BFC + preallocate=true + MEM_FRACTION=0.95)
set -u

D="${1:?subdir e.g. 3x3x3_nb100}"
VAR="${2:-platform_false}"

cd "$(dirname "${BASH_SOURCE[0]}")/${D}"

module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles 2>/dev/null
module load lorrax_C lorrax_agent 2>/dev/null

export SLURM_JOBID="${SLURM_JOBID:?need SLURM_JOBID}"
export LORRAX_NNODES=1
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

case "$VAR" in
    platform_false)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=false"
        ;;
    bfc_pre95)
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=true \
            --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
        ;;
    *)
        echo "Unknown variant: $VAR" >&2; exit 1 ;;
esac

PROBE_ENVS="--env=LORRAX_MEM_DEBUG=1 \
    --env=LORRAX_RCHUNK_DEBUG=1 \
    --env=LORRAX_MAX_RCHUNKS=3 \
    --env=LORRAX_EXIT_AFTER_ZETA=1 \
    --env=LORRAX_FORCE_FULL_BZ=1"

# Clean any leftover tmp
rm -rf tmp && mkdir -p tmp

OUT="gw_${VAR}.out"
echo "=== $(basename $PWD) variant=$VAR ===" | tee "$OUT"
LORRAX_SHIFTER="$LORRAX_SHIFTER $PROBE_ENVS $ALLOC_ENVS" \
    lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex.in" 2>&1 | tee -a "$OUT"
