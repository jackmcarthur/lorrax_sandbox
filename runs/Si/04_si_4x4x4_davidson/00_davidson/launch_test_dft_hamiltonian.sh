#!/usr/bin/env bash
set -euo pipefail

: "${JOBID:?export JOBID to an active Perlmutter interactive allocation first}"

SANDBOX=/global/homes/j/jackm/scratchperl/lorrax_sandbox
RUN_ROOT=$SANDBOX/runs/Si/00_si_4x4x4_60band
SITE=$HOME/scratchperl/.isdf/isdf_venvs/isdf_site
PY_PATH=$SANDBOX/sources:$SANDBOX/sources/lorrax/src:$SITE

exec srun --jobid="$JOBID" --gres=gpu:1 -N 1 -n 1 \
  shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \
    --env=PYTHONPATH="$PY_PATH" \
    --env=JAX_ENABLE_X64=1 \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python3 -u -m psp.tests.test_dft_hamiltonian \
    --save "$RUN_ROOT/qe/scf/silicon.save" \
    --pseudo_dir "$RUN_ROOT/qe/scf/silicon.save" \
    --wfn "$RUN_ROOT/qe/nscf/WFN.h5"
