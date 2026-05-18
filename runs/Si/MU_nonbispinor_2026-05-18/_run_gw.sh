#!/bin/bash -l
# Run gw_jax for a given mu subdir, mirroring _run_gw.sh from the bispinor sister sweep.
# Usage: _run_gw.sh <muXXX> [variant]
#   variant default = "platform_false" (= production sandbox default)
#   variant = "bfc_pre95" → BFC + preallocate=true + MEM_FRACTION=0.95
set -u

D="${1:?subdir e.g. mu384}"
VAR="${2:-platform_false}"

cd "$(dirname "${BASH_SOURCE[0]}")/${D}"

module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_A
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
    bfc_pre80)
        # Same allocator as bfc_pre95 but lower MEM_FRACTION to leave
        # headroom for cusolverMp FFI workspaces.  cusolverMp's
        # NCCL-based comm path tries to allocate user-buffer registration
        # outside the XLA arena; at MF=0.95 it returns status=7 INTERNAL
        # on Si 4×4×4.  MF=0.80 leaves ~8 GB/dev (40 GB hbm40g) for the
        # FFI, which is enough on this 7-MB C_q.
        ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
            --env=XLA_PYTHON_CLIENT_PREALLOCATE=true \
            --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.80"
        ;;
    *)
        echo "Unknown variant: $VAR" >&2; exit 1 ;;
esac

PROBE_ENVS="--env=LORRAX_MEM_DEBUG=1 \
    --env=LORRAX_RCHUNK_DEBUG=1 \
    --env=LORRAX_MAX_RCHUNKS=3 \
    --env=LORRAX_EXIT_AFTER_ZETA=1"

# Clean any leftover tmp
rm -rf tmp && mkdir -p tmp

OUT="gw_${VAR}.out"
echo "=== $(basename $PWD) variant=$VAR ===" | tee "$OUT"
LORRAX_SHIFTER="$LORRAX_SHIFTER $PROBE_ENVS $ALLOC_ENVS" \
    lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex.in" 2>&1 | tee -a "$OUT"
