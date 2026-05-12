#!/bin/bash
set -u
cd "$(dirname "$0")"
JID=51682523
PYPATH="/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
for dseed in v1_50win_s0:0 v1_50win_s2:2 v1_100win_s0:0 v1_150win_s0:0 v2_100win_s1:1 v2_150win_s1:1 v2_150win_s2:2; do
  d=${dseed%:*}; s=${dseed#*:}
  echo "rerun $d (seed=$s)"
  srun --jobid=$JID --gres=gpu:4 -N 4 -n 16 \
      shifter --module=gpu --image=$IMG \
      --env=PYTHONPATH=$PYPATH \
      --env=HDF5_USE_FILE_LOCKING=FALSE \
      --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
      --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
      python3 -u -m psp.run_nscf -i $d/nscf.in --pb-seed $s > $d/nscf_pb.out 2>&1 \
      || echo "  FAIL $d"
done
echo "done"
