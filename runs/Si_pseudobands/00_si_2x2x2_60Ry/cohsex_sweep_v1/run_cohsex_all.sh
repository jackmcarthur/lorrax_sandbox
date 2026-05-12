#!/bin/bash
# 18 COHSEX runs, serial, each on 16 GPUs of jobid 51682523
set -u
cd "$(dirname "$0")"
JID=51682523
PYPATH="/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"

count=0; total=18
for v in 1 2; do for w in 50 100 150; do for s in 0 1 2; do
  d="v${v}_${w}win_s${s}"
  count=$((count+1))
  echo "[$count/$total] $d"
  (cd $d && srun --jobid=$JID --gres=gpu:4 -N 4 -n 16 \
      shifter --module=gpu --image=$IMG \
      --env=PYTHONPATH=$PYPATH \
      --env=HDF5_USE_FILE_LOCKING=FALSE \
      --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
      --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
      python3 -u -m gw.gw_jax -i cohsex.in > gw.out 2>&1) || echo "  FAIL $d"
done; done; done
echo "all COHSEX done"
