#!/bin/bash
# Full COHSEX pipeline with 4-way parallelism across 4 nodes.
set -u
ROOT="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_sweep"
JID=51692404
cd "$ROOT"

PYPATH="/global/homes/j/jackm/software/lorrax_A/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
SH1="shifter --module=gpu --image=$IMG --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async"

throttle() {
  local max=$1
  while [ $(jobs -r | wc -l) -ge $max ]; do wait -n; done
}

# Step 1: centroids (6 configs, 1 GPU each, up to 16 parallel on 4 nodes)
echo "=== STEP 1: centroids ==="
for v in 1 2; do for w in 50 100 150; do
  src="v${v}_${w}win_s0"
  (cd $src && srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --exclusive $SH1 python3 -u -m centroid.kmeans_isdf 2000 --no-plot --seed 42 > centroid.out 2>&1 \
      && echo "  done $src centroids") &
  throttle 16
done; done
wait
# Symlink to s1, s2
for v in 1 2; do for w in 50 100 150; do
  for s in 1 2; do
    ln -sf ../v${v}_${w}win_s0/centroids_frac_2000.txt v${v}_${w}win_s${s}/centroids_frac_2000.txt
  done
done; done

# Step 2: dipole + kin_ion (18 of each, 1 GPU each)
echo "=== STEP 2: dipole + kin_ion ==="
for v in 1 2; do for w in 50 100 150; do for s in 0 1 2; do
  d="v${v}_${w}win_s${s}"
  (
    cd $d && srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --exclusive $SH1 python3 -u -m psp.get_dipole_mtxels -i cohsex.in > dipole.out 2>&1 && \
    srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --exclusive $SH1 python3 -u -m gw.kin_ion_io_chunked -i cohsex.in > kin.out 2>&1 && \
    echo "  done $d prep"
  ) &
  throttle 16
done; done; done
wait

# Step 3: COHSEX (18 runs, 4 GPUs each, 4 parallel)
echo "=== STEP 3: COHSEX ==="
for v in 1 2; do for w in 50 100 150; do for s in 0 1 2; do
  d="v${v}_${w}win_s${s}"
  (cd $d && srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 --exclusive $SH1 python3 -u -m gw.gw_jax -i cohsex.in > gw.out 2>&1 \
      && echo "  done $d COHSEX") &
  throttle 4
done; done; done
wait
echo "all done"
