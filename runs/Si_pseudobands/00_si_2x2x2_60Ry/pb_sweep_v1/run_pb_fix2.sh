#!/bin/bash
set -u
cd "$(dirname "$0")"
JID=51682523
PYPATH="/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
for d in v2_50win_s0 v2_100win_s0; do
  echo "rerun $d"
  srun --jobid=$JID --gres=gpu:4 -N 4 -n 16 \
      shifter --module=gpu --image=$IMG \
      --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE \
      --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
      --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
      python3 -u -m psp.run_nscf -i $d/nscf.in --pb-seed 0 > $d/nscf_pb.out 2>&1 || echo "FAIL $d"
done
