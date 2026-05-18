#!/bin/bash -l
# Run gw_jax for a given kgrid subdir.
# Usage: _run_gw.sh <NxNxN> [variant]
#   variant default = "platform_false" (= production sandbox default)
#   variant = "bfc_pre95" → BFC + preallocate=true + MEM_FRACTION=0.95
set -u

KG="${1:?subdir e.g. 2x2x2}"
VAR="${2:-platform_false}"

cd "$(dirname "${BASH_SOURCE[0]}")/${KG}/mu_run"

module purge 2>/dev/null || true
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_B
module load lorrax_agent

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
echo "=== $(basename $PWD) kg=$KG variant=$VAR ===" | tee "$OUT"
LORRAX_SHIFTER="$LORRAX_SHIFTER $PROBE_ENVS $ALLOC_ENVS" \
    lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex.in" 2>&1 | tee -a "$OUT"
