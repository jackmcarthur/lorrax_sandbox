#!/bin/bash
set -u
cd "$(dirname "$0")"
JID=$(cat /pscratch/sd/j/jackm/lorrax_sandbox/.my_jobid)
PYPATH="/global/homes/j/jackm/software/lorrax_A/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
SH="shifter --module=gpu --image=$IMG --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async"
throttle() { while [ $(jobs -r | wc -l) -ge $1 ]; do wait -n; done; }
for name in v1_k4_45w v1_k6_27w v1_k6_28w v1_k8_22w v1_k8_25w v1_k6_38w v1_k6_42w v1_k6_44w; do for s in 0 1 2; do
  d="${name}_s${s}"
  (srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 --exclusive $SH python3 -u -m psp.run_nscf -i $d/nscf.in --pb-seed $s > $d/nscf_pb.out 2>&1) &
  throttle 4
done; done
wait
echo "PB done"
