#!/bin/bash
# Run three pivoted-Cholesky sweeps on different WFN types.
# Usage: SLURM_JOBID=<jobid> ./run_all.sh
set -euo pipefail

module load lorrax_B

SANDBOX=/pscratch/sd/j/jackm/lorrax_sandbox
ASSAY=$SANDBOX/runs/Si_B_assay/pchol_pseudoband_assay.py
OUT=$SANDBOX/runs/Si_B_assay/pseudobands_sweep

# 1. Parabands (4200 bands, deterministic, unit norms). Same n_cond ladder as
#    the production 60 Ry assay so the parabands / 60Ry-NSCF comparison is
#    apples-to-apples at each point.
cd $OUT/parabands_4200
lxrun python3 -u $ASSAY \
    --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5 \
    --wfn-label parabands_4200 \
    --out $OUT/parabands_4200/pchol_assay.json \
    --pair-mode isdf_asym \
    --n-val 8 --n-val-sigma 8 \
    --n-cond 8 16 32 64 128 256 400 \
    --M 2400 --n-keep 1300 --seed 42 \
    2>&1 | tee $OUT/parabands_4200/assay.log

# 2. Pseudobands (50 slabs → 116 bands total). Sweep up to 108 (all
#    pseudobands = 116 − 8 valence). Band norms ON (this is the reason
#    this sweep exists).
cd $OUT/pseudo_50sl_116
lxrun python3 -u $ASSAY \
    --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/02_bgw_pseudobands_50sl/WFN_pseudo.h5 \
    --wfn-label pseudo_50sl_116 \
    --out $OUT/pseudo_50sl_116/pchol_assay.json \
    --pair-mode isdf_asym \
    --n-val 8 --n-val-sigma 8 \
    --n-cond 8 16 32 64 108 \
    --M 2400 --n-keep 1300 --seed 42 \
    2>&1 | tee $OUT/pseudo_50sl_116/assay.log

# 3. Pseudobands (100 slabs → 216 bands total). Sweep up to 208.
cd $OUT/pseudo_100sl_216
lxrun python3 -u $ASSAY \
    --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5 \
    --wfn-label pseudo_100sl_216 \
    --out $OUT/pseudo_100sl_216/pchol_assay.json \
    --pair-mode isdf_asym \
    --n-val 8 --n-val-sigma 8 \
    --n-cond 8 16 32 64 128 208 \
    --M 2400 --n-keep 1300 --seed 42 \
    2>&1 | tee $OUT/pseudo_100sl_216/assay.log

echo "=== all sweeps done ==="
