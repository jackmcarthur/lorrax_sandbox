#!/usr/bin/env bash
# Regenerate CrI3 6x6 80 Ry centroids with orbit-aware kmeans (default
# when wfn.ntran > 1 and --no-orbit is NOT passed). Target N_c=1504; the
# orbit unfold will inflate to N_unfolded (typically 1508 based on prior
# kmeans_1504_v3 run logs).
set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
RUN_DIR=$SBOX/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_2026-05-14

module use $SBOX/modulefiles
module load lorrax_B
module load lorrax_agent
lxattach

cd $RUN_DIR

echo "[kmeans] SLURM_JOBID=${SLURM_JOBID:-unset}  cwd=$(pwd)  date=$(date)"
echo "[kmeans] commit: $(cd $SBOX/sources/lorrax_B && git rev-parse --short HEAD)"

# 2 GPUs on 1 node (orbit-aware default; single-GPU OOMs on pivoted-Cholesky
# batched FFT plan for M=2252 oversample candidates; 4 GPUs failed because
# nb_total=70 not divisible by 4 in gflat_to_rmu — 70/2 = 35 works).
LORRAX_NNODES=1 LORRAX_NGPU=2 lxrun \
    python3 -u -m centroid.kmeans_cli 1504 --seed 42 \
    2>&1 | tee kmeans_orbit.log
echo "[kmeans] finished at $(date)"
