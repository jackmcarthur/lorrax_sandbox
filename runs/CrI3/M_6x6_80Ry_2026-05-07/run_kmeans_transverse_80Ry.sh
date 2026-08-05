#!/usr/bin/env bash
# Generate transverse (current-density) centroids for the CrI3 6x6 80 Ry
# bispinor IBZ gate.  Target N_c≈1500 to roughly match the existing 1508
# charge centroids.  Orbit-aware is the default for ntran>1 wfn (so no
# --no-orbit), seed=42 to match the 30 Ry recipe.  Writes
# centroids_frac_<N_unique>_current.txt into cwd.
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
PARENT=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07
RUN_DIR=$PARENT/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16

module use $HOME/modulefiles
module load lorrax_B
module use $SBOX/modulefiles
module load lorrax_agent

cd "$RUN_DIR"

echo "[kmeans-transverse] SLURM_JOBID=${SLURM_JOBID:-unset}  cwd=$(pwd)  date=$(date)"
echo "[kmeans-transverse] commit: $(cd $SBOX/sources/lorrax_B && git rev-parse --short HEAD)"

# Mirror the 80 Ry charge-centroid recipe: 2 GPUs / 1 node.  The
# pivoted-Cholesky M=oversample*N_c FFT plan needs ≥2 GPUs to fit at 80 Ry.
LORRAX_NNODES=1 LORRAX_NGPU=2 lxrun \
    python3 -u -m centroid.kmeans_cli 1500 \
    --seed 42 --density-mode current \
    2>&1 | tee kmeans_transverse_orbit.log
echo "[kmeans-transverse] finished at $(date)"
