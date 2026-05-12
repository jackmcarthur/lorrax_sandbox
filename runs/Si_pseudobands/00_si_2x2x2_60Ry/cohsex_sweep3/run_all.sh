#!/bin/bash
set -u
ROOT="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_sweep3"
JID=$(cat /pscratch/sd/j/jackm/lorrax_sandbox/.my_jobid)
cd "$ROOT"
PYPATH="/global/homes/j/jackm/software/lorrax_A/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
SH1="shifter --module=gpu --image=$IMG --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async"
throttle() { while [ $(jobs -r | wc -l) -ge $1 ]; do wait -n; done; }

echo "=== STEP 1: centroids ==="
for name in v1_k6_30w v1_k6_35w v1_k6_45w v1_k6_60w v1_k4_60w v1_k4_100w; do
  src="${name}_s0"
  (cd $src && srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --exclusive $SH1 python3 -u -m centroid.kmeans_isdf 2000 --no-plot --seed 42 > centroid.out 2>&1) &
  throttle 16
done
wait
for name in v1_k6_30w v1_k6_35w v1_k6_45w v1_k6_60w v1_k4_60w v1_k4_100w; do
  for s in 1 2; do ln -sf ../${name}_s0/centroids_frac_2000.txt ${name}_s${s}/centroids_frac_2000.txt; done
done

echo "=== STEP 2: dipole + kin_ion ==="
for name in v1_k6_30w v1_k6_35w v1_k6_45w v1_k6_60w v1_k4_60w v1_k4_100w; do for s in 0 1 2; do
  d="${name}_s${s}"
  (cd $d && \
   srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --exclusive $SH1 python3 -u -m psp.get_dipole_mtxels -i cohsex.in > dipole.out 2>&1 && \
   srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --exclusive $SH1 python3 -u -m gw.kin_ion_io_chunked -i cohsex.in > kin.out 2>&1) &
  throttle 16
done; done
wait

echo "=== STEP 3: COHSEX ==="
for name in v1_k6_30w v1_k6_35w v1_k6_45w v1_k6_60w v1_k4_60w v1_k4_100w; do for s in 0 1 2; do
  d="${name}_s${s}"
  (cd $d && srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 --exclusive $SH1 python3 -u -m gw.gw_jax -i cohsex.in > gw.out 2>&1) &
  throttle 4
done; done
wait
echo "all done"
