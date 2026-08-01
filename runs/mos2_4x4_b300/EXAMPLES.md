# Worked example: MoS2 4x4, 300 bands (2026-07-31)

Complete certified chain on Frontera: NSCF(300b) -> centroids (2979/3000,
rank gate 270/270) -> dipole -> kin-ion -> GW (chi/W at 300 bands, sigma
window 26v+18c) -> htransform (DFT + QP) -> plot. DFT gap 1.72 eV,
G0W0 2.43 eV (K). Jobs 7884642..7884861; scorecard sections BC-BE.

Files here are the small inputs/harnesses/outputs. Heavy artifacts stay at
their original paths (provenance — the scorecard cites them):

| artifact | path |
|---|---|
| deck + WFN + baselines | /scratch2/08271/jackmc/mos2_4x4_test/ (WFN_b300.h5, kin_ion_b300.h5, dipole_b300.h5, centroids_b300.txt) |
| GW run dir (sigma_mnk.h5 356 MB, restart tensors) | /scratch2/08271/jackmc/mos2_4x4_test/run_b300_gw/ |
| P=16 A/B outputs | /scratch2/08271/jackmc/mos2_4x4_test/*_p16* (scorecard BD) |
| pinned 800c parity baseline | /scratch2/08271/jackmc/mos2_4x4_test/_archive/2026-07-30/runs/run_800c_merged/ |

Harness notes: post_b300.sbatch shows the single-process env (JAX_PROCESS_COUNT=1
pin, libfabric in LD_LIBRARY_PATH); gw_ht_b300.sbatch the 8x2 multi-process GW
block; make_eqp_htformat.py converts BGW-column eqp files to the htransform
--eqp-file format (Ry, full-BZ blocks).
