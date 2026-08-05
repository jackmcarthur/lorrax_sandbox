#!/bin/bash -l
# Generate scalar centroid file for a given subdir/mu.
# Usage: _kmeans_run.sh <subdir> <mu>
set -u

D="${1:?subdir e.g. 3x3x3_nb100}"
MU="${2:?mu}"
cd "$(dirname "${BASH_SOURCE[0]}")/${D}"

module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles 2>/dev/null
module load lorrax_C lorrax_agent 2>/dev/null

export SLURM_JOBID="${SLURM_JOBID:?need SLURM_JOBID}"
export LORRAX_NNODES=1
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

if [ ! -f "centroids_frac_${MU}.txt" ]; then
    echo "=== kmeans μ=${MU} in $D ===" | tee "kmeans_${MU}.out"
    # Use QE save (silicon.save) directly to bypass the WFN.h5 nspinor=1
    # broadcast bug in load_wfns.unfold_psi (U_spinor hard-coded 2×2).
    lxrun python3 -u -m centroid.kmeans_cli "${MU}" --seed 42 \
        --qe-save "$PWD/qe/scf/silicon.save" \
        2>&1 | tee -a "kmeans_${MU}.out"
else
    echo "  centroids_frac_${MU}.txt exists, skipping"
fi
