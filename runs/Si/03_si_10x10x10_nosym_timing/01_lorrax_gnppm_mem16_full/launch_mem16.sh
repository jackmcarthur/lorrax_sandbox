#!/usr/bin/env bash
set -euo pipefail
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
SRC=/global/homes/j/jackm/scratchperl/lorrax_sandbox/sources/lorrax/src
SHIFTER="shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 --env=PYTHONPATH=$SRC:$SITE --env=JAX_ENABLE_X64=1 --env=HDF5_USE_FILE_LOCKING=FALSE"
srun --jobid=51141591 --gres=gpu:4 -N 4 -n 16 -c 16 \
  $SHIFTER \
  --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  --env=LORRAX_MEM_PROFILE=1 \
  python3 -u -m gw.gw_jax -i "$(pwd)/cohsex.in" \
  2>&1 | tee gw_mem16_clean.out
