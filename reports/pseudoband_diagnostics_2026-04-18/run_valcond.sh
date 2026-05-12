#!/bin/bash
# pair_mode=val_cond on the pseudoband WFNs — chi-like Gram (pure valence ×
# pure conduction cross-term) rather than the sigma-like isdf_asym Gram
# (left=right, full-window squared-density). The squared-density Gram has
# a nearly-uniform diagonal under band-norm rescaling, which is why
# pivoted Cholesky plateaus early. Cross-term may have sharper structure.
set -euo pipefail
module load lorrax_B
SANDBOX=/pscratch/sd/j/jackm/lorrax_sandbox
ASSAY=$SANDBOX/runs/Si_B_assay/pchol_pseudoband_assay.py
OUT=$SANDBOX/runs/Si_B_assay/pseudobands_sweep

for tag in pseudo_50sl_116_valcond pseudo_100sl_216_valcond parabands_4200_valcond; do
    mkdir -p $OUT/$tag
done

run_one() {
    local dir=$1 wfn=$2 label=$3 ncond="$4"
    cd $OUT/$dir
    lxrun python3 -u $ASSAY \
        --wfn $wfn --wfn-label $label \
        --out $OUT/$dir/pchol_assay.json \
        --pair-mode val_cond \
        --n-val 8 \
        --n-cond $ncond \
        --M 2400 --n-keep 1300 --seed 42 \
        --kmeans-density valence \
        2>&1 | tee $OUT/$dir/assay.log
}

run_one parabands_4200_valcond \
        /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5 \
        parabands_4200_valcond "8 16 32 64 128 256 400"

run_one pseudo_50sl_116_valcond \
        /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/02_bgw_pseudobands_50sl/WFN_pseudo.h5 \
        pseudo_50sl_116_valcond "8 16 32 64 108"

run_one pseudo_100sl_216_valcond \
        /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5 \
        pseudo_100sl_216_valcond "8 16 32 64 128 208"

echo "=== val_cond sweeps done ==="
