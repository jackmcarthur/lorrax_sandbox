#!/bin/bash
# 18 PB generations, strictly serial, each using all 16 GPUs of jobid 51682523
# Only uses this specific jobid — no stale SLURM_JOBID interference.
set -u
cd "$(dirname "$0")"
JID=51682523
PYPATH="/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"

count=0; total=18
for v in 1 2; do for win in 50 100 150; do for seed in 0 1 2; do
  d="v${v}_${win}win_s${seed}"
  count=$((count+1))
  echo "[$count/$total] $d  (jobid=$JID)"
  srun --jobid=$JID --gres=gpu:4 -N 4 -n 16 \
      shifter --module=gpu --image=$IMG \
      --env=PYTHONPATH=$PYPATH \
      --env=HDF5_USE_FILE_LOCKING=FALSE \
      --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
      --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
      python3 -u -m psp.run_nscf -i $d/nscf.in --pb-seed $seed > $d/nscf_pb.out 2>&1 \
      || echo "  FAIL $d"
done; done; done
echo "all PB done"
