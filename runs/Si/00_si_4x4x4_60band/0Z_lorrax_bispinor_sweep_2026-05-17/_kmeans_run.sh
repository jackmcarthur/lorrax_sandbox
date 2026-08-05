#!/bin/bash -l
# Generate scalar + current centroid files for a given μ.
# Usage: _kmeans_run.sh <mu>
set -u

MU="${1:?mu}"
cd "$(dirname "${BASH_SOURCE[0]}")"

module purge 2>/dev/null || true
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_B
module load lorrax_agent

export SLURM_JOBID="${SLURM_JOBID:?need SLURM_JOBID}"
export LORRAX_NNODES=1
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

# Scalar centroids (γ̃⁰ charge channel)
if [ ! -f "centroids_frac_${MU}.txt" ]; then
    echo "=== kmeans scalar μ=${MU} ===" | tee "kmeans_${MU}_scalar.out"
    lxrun python3 -u -m centroid.kmeans_cli "${MU}" \
        --qe-save "$PWD/qe/scf/silicon.save" \
        2>&1 | tee -a "kmeans_${MU}_scalar.out"
else
    echo "  centroids_frac_${MU}.txt exists, skipping scalar kmeans"
fi

# Current centroids (γ̃^{1,2,3} transverse channel)
if [ ! -f "centroids_frac_${MU}_current.txt" ]; then
    echo "=== kmeans current μ=${MU} ===" | tee "kmeans_${MU}_current.out"
    lxrun python3 -u -m centroid.kmeans_cli "${MU}" --density-mode current \
        2>&1 | tee -a "kmeans_${MU}_current.out"
else
    echo "  centroids_frac_${MU}_current.txt exists, skipping current kmeans"
fi
