#!/bin/bash
# Generate val_cond-selected centroid files at N_mu ∈ {400, 800, 1200} for
# each (WFN, n_cond) combination. Uses M = 2400 candidate pool, n_keep = 1300,
# pair_mode=val_cond (confirmed best in 2026-04-18 diagnostic study). Writes
# one centroids_frac_ncond{nc}_Nmu{Nm}.txt per combination to the shared
# centroids/ directory.
set -euo pipefail
module load lorrax_B
ROOT=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_valcond_2026-04-18
ASSAY=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_B_assay/pchol_pseudoband_assay.py

run_one() {
    local label=$1 wfn=$2 ncond_list="$3"
    local log=$ROOT/logs/centroids_${label}.log
    cd $ROOT/centroids
    # Each run loops over all ncond values for the same WFN so k-means runs
    # only once per WFN (M=2400 pool is shared across n_cond).
    # Use 4 GPUs on a single node (lxrun's default). A dedicated shifter
    # srun for centroid generation; LORRAX COHSEX later uses different node.
    LORRAX_NGPU=4 lxrun python3 -u $ASSAY \
        --wfn $wfn --wfn-label $label \
        --out $ROOT/centroids/pchol_assay_${label}.json \
        --pair-mode val_cond \
        --n-val 8 \
        --n-cond $ncond_list \
        --M 2400 --n-keep 1300 --seed 42 \
        --kmeans-density valence \
        --save-centroids-dir $ROOT/centroids \
        --save-centroids-N 400 800 1200 \
        2>&1 | tee $log
    # Retag written files with the WFN label prefix.
    for f in $ROOT/centroids/centroids_frac_ncond*_Nmu*.txt; do
        if [ -f "$f" ]; then
            base=$(basename "$f")
            if ! echo "$base" | grep -q "${label}"; then
                mv -n "$f" "$ROOT/centroids/${label}_${base}"
            fi
        fi
    done
}

run_one parabands_4200 \
    /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5 \
    "60 140 208"

run_one pseudo_100sl_216 \
    /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5 \
    "60 140 208"

echo "=== centroid generation done ==="
ls -la $ROOT/centroids/
