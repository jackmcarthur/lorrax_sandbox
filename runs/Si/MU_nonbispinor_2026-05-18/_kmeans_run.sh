#!/bin/bash -l
# Generate scalar centroids for a given μ. Non-bispinor only needs ONE set.
# Usage: _kmeans_run.sh <mu>
set -u

MU="${1:?mu}"
cd "$(dirname "${BASH_SOURCE[0]}")"

module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_A
module load lorrax_agent

export SLURM_JOBID="${SLURM_JOBID:?need SLURM_JOBID}"
export LORRAX_NNODES=1
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

if [ ! -f "centroids_frac_${MU}.txt" ]; then
    echo "=== kmeans scalar μ=${MU} ===" | tee "kmeans_${MU}.out"
    lxrun python3 -u -m centroid.kmeans_cli "${MU}" \
        --qe-save "$PWD/qe/scf/silicon.save" \
        --seed 42 \
        2>&1 | tee -a "kmeans_${MU}.out"
else
    echo "  centroids_frac_${MU}.txt exists, skipping"
fi
