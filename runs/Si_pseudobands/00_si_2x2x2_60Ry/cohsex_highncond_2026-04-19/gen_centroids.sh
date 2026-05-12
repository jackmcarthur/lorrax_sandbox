#!/bin/bash
# val_cond centroids for n_cond ∈ {100, 300, 600, 1000}, saving at
# N_μ ∈ {400, 1200, 3200, 6000}. M=12000 pool, n_keep=6000 (so the
# largest N_μ is 50% of pool, same fraction as previous studies).
set -euo pipefail
module load lorrax_B
ROOT=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_highncond_2026-04-19
ASSAY=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_B_assay/pchol_pseudoband_assay.py
PARA_WFN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5

cd $ROOT/centroids
LORRAX_NGPU=4 lxrun python3 -u $ASSAY \
    --wfn $PARA_WFN --wfn-label parabands_4200 \
    --out $ROOT/centroids/pchol_assay.json \
    --pair-mode val_cond \
    --n-val 8 \
    --n-cond 100 300 600 1000 \
    --M 12000 --n-keep 6000 --seed 42 \
    --kmeans-density valence \
    --save-centroids-dir $ROOT/centroids \
    --save-centroids-N 400 1200 3200 6000 \
    2>&1 | tee $ROOT/logs/gen_centroids.log
echo "=== centroid files generated ==="
ls -la $ROOT/centroids/centroids_frac_ncond*_Nmu*.txt | wc -l
