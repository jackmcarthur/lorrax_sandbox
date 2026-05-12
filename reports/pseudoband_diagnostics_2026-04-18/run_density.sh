#!/bin/bash
# k-means density-mode comparison. Same M=2400, n_keep=1300 as the original
# sweep — the only knob being varied is *where* the candidate centroids get
# placed on the unit cell (valence-density vs uniform vs pseudo-combined).
set -euo pipefail
module load lorrax_B
SANDBOX=/pscratch/sd/j/jackm/lorrax_sandbox
ASSAY=$SANDBOX/runs/Si_B_assay/pchol_pseudoband_assay.py
OUT=$SANDBOX/runs/Si_B_assay/pseudobands_sweep

for tag in \
    parabands_4200_uniform \
    pseudo_50sl_116_uniform \
    pseudo_100sl_216_uniform \
    pseudo_50sl_116_pseudo_combined \
    pseudo_100sl_216_pseudo_combined; do
    mkdir -p $OUT/$tag
done

run_one() {
    local dir=$1 wfn=$2 label=$3 density=$4 ncond="$5"
    cd $OUT/$dir
    lxrun python3 -u $ASSAY \
        --wfn $wfn --wfn-label $label \
        --out $OUT/$dir/pchol_assay.json \
        --pair-mode isdf_asym \
        --n-val 8 --n-val-sigma 8 \
        --n-cond $ncond \
        --M 2400 --n-keep 1300 --seed 42 \
        --kmeans-density $density \
        2>&1 | tee $OUT/$dir/assay.log
}

# Uniform density on all three WFN types.
run_one parabands_4200_uniform \
        /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5 \
        parabands_4200_uniform uniform "8 16 32 64 128 256 400"

run_one pseudo_50sl_116_uniform \
        /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/02_bgw_pseudobands_50sl/WFN_pseudo.h5 \
        pseudo_50sl_116_uniform uniform "8 16 32 64 108"

run_one pseudo_100sl_216_uniform \
        /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5 \
        pseudo_100sl_216_uniform uniform "8 16 32 64 128 208"

# Pseudo-combined density on pseudoband WFNs only (same effect as valence on parabands).
run_one pseudo_50sl_116_pseudo_combined \
        /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/02_bgw_pseudobands_50sl/WFN_pseudo.h5 \
        pseudo_50sl_116_pseudo_combined pseudo_combined "8 16 32 64 108"

run_one pseudo_100sl_216_pseudo_combined \
        /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5 \
        pseudo_100sl_216_pseudo_combined pseudo_combined "8 16 32 64 128 208"

echo "=== density-mode sweeps done ==="
