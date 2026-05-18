#!/bin/bash -l
# Generate scalar centroids for a given kgrid subdir/mu.
# Usage: _kmeans_run.sh <NxNxN> <mu>
set -eu

KG="${1:?kgrid e.g. 2x2x2}"
MU="${2:?mu}"

RUN_DIR="$(dirname "${BASH_SOURCE[0]}")/${KG}"
cd "$RUN_DIR"

module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_B
module load lorrax_agent

export SLURM_JOBID="${SLURM_JOBID:?need SLURM_JOBID}"
export LORRAX_NNODES=1
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

if [ ! -f "centroids_frac_${MU}.txt" ]; then
    echo "=== kmeans scalar kg=${KG} μ=${MU} ===" | tee "kmeans_${MU}_scalar.out"
    lxrun python3 -u -m centroid.kmeans_cli "${MU}" \
        --qe-save "$PWD/qe/scf/silicon.save" \
        2>&1 | tee -a "kmeans_${MU}_scalar.out"
else
    echo "  centroids_frac_${MU}.txt exists, skipping"
fi
