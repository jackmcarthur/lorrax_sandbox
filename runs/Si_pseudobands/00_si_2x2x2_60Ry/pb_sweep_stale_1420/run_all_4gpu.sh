#!/bin/bash
# PB generation on 4 GPUs (1 node) per run, serial.
# For a 2x2x2 Si system the parallel speedup flatlines after 4 GPUs; 16 GPUs
# just added all-gather overhead in the previous 18-run sweep.
set -u
cd "$(dirname "$0")"
JID=51692404
PYPATH="/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"

count=0; total=18
for v in 1 2; do for win in 50 100 150; do for seed in 0 1 2; do
  d="v${v}_${win}win_s${seed}"
  count=$((count+1))
  t0=$(date +%s)
  echo "[$count/$total] $d (start $(date +%H:%M:%S))"
  srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 \
      shifter --module=gpu --image=$IMG \
      --env=PYTHONPATH=$PYPATH \
      --env=HDF5_USE_FILE_LOCKING=FALSE \
      --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
      --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
      python3 -u -m psp.run_nscf -i $d/nscf.in --pb-seed $seed > $d/nscf_pb.out 2>&1 \
      || echo "  FAIL $d"
  dt=$(( $(date +%s) - t0 ))
  echo "  elapsed ${dt}s"
done; done; done
echo "all PB done"
