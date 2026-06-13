#!/bin/bash
# Run ζ-fit phase of gw_jax for the production-config non-bispinor sweep.
# Usage: _run_gw.sh <mu_dir> <variant>
#   mu_dir = mu384 | mu1200
#   variant = platform_false | bfc_pre95
#
# Config: noncolin=.true., lspinorb=.true., FR pseudo (matches bispinor agent_t).
# cohsex.in has bispinor=false; cusolverMp left at DEFAULT (on). hbm80g
# allocation lets BFC+0.95 + cuSOLVERMp coexist (4 GB free for NCCL).

set -e
MUDIR=$1
VARIANT=$2
RUN_DIR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/NONBISPINOR_PROD_2026-05-19

cd "$RUN_DIR/$MUDIR"

PROBE_ENVS=" \
  --env=LORRAX_MEM_DEBUG=1 \
  --env=LORRAX_RCHUNK_DEBUG=1 \
  --env=LORRAX_MAX_RCHUNKS=3 \
  --env=LORRAX_EXIT_AFTER_ZETA=1"

case "$VARIANT" in
  platform_false)
    ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
                --env=XLA_PYTHON_CLIENT_PREALLOCATE=false"
    ;;
  bfc_pre95)
    ALLOC_ENVS="--env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
                --env=XLA_PYTHON_CLIENT_PREALLOCATE=true \
                --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
    ;;
  *) echo "unknown variant: $VARIANT"; exit 1 ;;
esac

LORRAX_SHIFTER="$LORRAX_SHIFTER $PROBE_ENVS $ALLOC_ENVS" \
  lxrun python3 -u -m gw.gw_jax -i cohsex.in 2>&1 | tee gw_${VARIANT}.out
