#!/bin/bash
# 18 PB runs, 4-way parallel across 4 nodes (1 run per node = 4 GPUs each).
set -u
cd "$(dirname "$0")"
JID=51692404
PYPATH="/global/homes/j/jackm/software/lorrax_A/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
SH="shifter --module=gpu --image=$IMG --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async"

RUNS=()
for v in 1 2; do for w in 50 100 150; do for s in 0 1 2; do
  RUNS+=("v${v}_${w}win_s${s}:$s")
done; done; done

launch_one() {
  local spec=$1
  local d=${spec%:*}
  local seed=${spec#*:}
  srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 --exclusive \
      $SH python3 -u -m psp.run_nscf -i $d/nscf.in --pb-seed $seed \
      > $d/nscf_pb.out 2>&1
}

total=${#RUNS[@]}
count=0
for spec in "${RUNS[@]}"; do
  count=$((count+1))
  echo "[$count/$total] launching $spec"
  launch_one "$spec" &
  # Throttle: keep at most 4 concurrent
  while [ $(jobs -r | wc -l) -ge 4 ]; do
    wait -n
  done
done
wait
echo "all PB done"
