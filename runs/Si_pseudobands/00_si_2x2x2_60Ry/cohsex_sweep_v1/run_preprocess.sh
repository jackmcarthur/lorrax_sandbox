#!/bin/bash
# Preprocess 18 cohsex dirs: centroids (once per V,W), dipole+kin_ion (per seed).
# Uses srun directly with an explicit jobid so no stale SLURM_JOBID matters.
set -u
ROOT="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_sweep"
JID=51682523
PYPATH="/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
SRUN_1GPU="srun --jobid=$JID --gres=gpu:1 -N 1 -n 1"
SHIFTER_ENV="shifter --module=gpu --image=$IMG --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async"

cd "$ROOT"

# Step 1: centroids — once per (V,W), then symlink to s1/s2
for v in 1 2; do for w in 50 100 150; do
  src="v${v}_${w}win_s0"
  echo "=== centroids $src ==="
  (cd $src && $SRUN_1GPU $SHIFTER_ENV python3 -u -m centroid.kmeans_isdf 2000 --no-plot --seed 42 > centroid.out 2>&1) || { echo "FAIL $src centroids"; continue; }
  for s in 1 2; do
    dst="v${v}_${w}win_s${s}"
    ln -sf "../$src/centroids_frac_2000.txt" "$dst/centroids_frac_2000.txt"
  done
done; done

# Step 2: dipole + kin_ion — per seed
for v in 1 2; do for w in 50 100 150; do for s in 0 1 2; do
  d="v${v}_${w}win_s${s}"
  echo "=== dipole+kin $d ==="
  (cd $d && $SRUN_1GPU $SHIFTER_ENV python3 -u -m psp.get_dipole_mtxels -i cohsex.in > dipole.out 2>&1) || { echo "FAIL $d dipole"; continue; }
  (cd $d && $SRUN_1GPU $SHIFTER_ENV python3 -u -m gw.kin_ion_io_chunked -i cohsex.in > kin.out 2>&1) || { echo "FAIL $d kin"; continue; }
done; done; done
echo "preprocessing done"
