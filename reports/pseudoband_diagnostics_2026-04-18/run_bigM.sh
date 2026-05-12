#!/bin/bash
# Larger-M sweep: M=4000, n_keep=2000 — ~10× band count, same asymmetric window
# + band_norms recipe. Plus a "norms off" diagnostic on pseudo_100sl to see if
# the max(norm,1) clamp is responsible for the plateau.
set -euo pipefail
module load lorrax_B
SANDBOX=/pscratch/sd/j/jackm/lorrax_sandbox
ASSAY=$SANDBOX/runs/Si_B_assay/pchol_pseudoband_assay.py
OUT=$SANDBOX/runs/Si_B_assay/pseudobands_sweep

for d in parabands_4200_M4000 pseudo_50sl_116_M4000 pseudo_100sl_216_M4000 pseudo_100sl_216_M4000_nonorms; do
    mkdir -p $OUT/$d
done

# ---- M=4000, n_keep=2000 sweep on all three WFNs ----
cd $OUT/parabands_4200_M4000
lxrun python3 -u $ASSAY \
    --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5 \
    --wfn-label parabands_4200_M4000 \
    --out $OUT/parabands_4200_M4000/pchol_assay.json \
    --pair-mode isdf_asym \
    --n-val 8 --n-val-sigma 8 \
    --n-cond 8 16 32 64 128 256 400 \
    --M 4000 --n-keep 2000 --seed 42 \
    2>&1 | tee $OUT/parabands_4200_M4000/assay.log

cd $OUT/pseudo_50sl_116_M4000
lxrun python3 -u $ASSAY \
    --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/02_bgw_pseudobands_50sl/WFN_pseudo.h5 \
    --wfn-label pseudo_50sl_116_M4000 \
    --out $OUT/pseudo_50sl_116_M4000/pchol_assay.json \
    --pair-mode isdf_asym \
    --n-val 8 --n-val-sigma 8 \
    --n-cond 8 16 32 64 108 \
    --M 4000 --n-keep 2000 --seed 42 \
    2>&1 | tee $OUT/pseudo_50sl_116_M4000/assay.log

cd $OUT/pseudo_100sl_216_M4000
lxrun python3 -u $ASSAY \
    --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5 \
    --wfn-label pseudo_100sl_216_M4000 \
    --out $OUT/pseudo_100sl_216_M4000/pchol_assay.json \
    --pair-mode isdf_asym \
    --n-val 8 --n-val-sigma 8 \
    --n-cond 8 16 32 64 128 208 \
    --M 4000 --n-keep 2000 --seed 42 \
    2>&1 | tee $OUT/pseudo_100sl_216_M4000/assay.log

# ---- Diagnostic: --no-band-norms on the worst-plateau pseudoband sweep ----
# If plateau drops substantially, the max(norm,1) clamp is suppressing the
# very pseudoband-dominated diagonal entries that should *guide* the pivots
# (i.e. the per-point amplitude is what pivoted Cholesky needs to see).
cd $OUT/pseudo_100sl_216_M4000_nonorms
lxrun python3 -u $ASSAY \
    --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5 \
    --wfn-label pseudo_100sl_216_nonorms \
    --out $OUT/pseudo_100sl_216_M4000_nonorms/pchol_assay.json \
    --pair-mode isdf_asym \
    --n-val 8 --n-val-sigma 8 \
    --n-cond 8 16 32 64 128 208 \
    --M 4000 --n-keep 2000 --seed 42 \
    --no-band-norms \
    2>&1 | tee $OUT/pseudo_100sl_216_M4000_nonorms/assay.log

echo "=== bigM + nonorms diagnostic sweeps done ==="
