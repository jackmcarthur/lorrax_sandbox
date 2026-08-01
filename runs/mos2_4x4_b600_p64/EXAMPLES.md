# Worked example: MoS2 4x4, 600 bands, 4775 centroids, P=64 (2026-08-01)

Scale shakeout of the whole driver chain at the P=64 production geometry.
Everything green, no OOM, no refusal.  `RESULTS.md` is the full record
(stage tables, scaling comparison against b300/P=16, the distributed-tier
diagnostic leg).  CLAIMS rows 37-40.

Jobs 7885312 (bundle) / 7885313 (P=4 harness smoke) / 7885315 (kmeans +
dipole + kin-ion) / 7885316 (GW, local zeta tier) / 7885322 (htransform
dft+qp + plot) / 7885323 (GW, distributed zeta tier).

Files here are the small inputs/harnesses/outputs.  Heavy artifacts stay
at their original paths:

| artifact | path |
|---|---|
| campaign root (all logs, VmHWM samples, frozen src, bundle) | /scratch2/08271/jackmc/b600_p64/ |
| centroid set (4775c, 600-band window) | .../run_deck/centroids_frac_4775_b600_c4800.txt |
| operators | .../run_deck/{dipole_b600.h5 322 MB, kin_ion_b600.h5 184 MB} |
| GW run dirs (sigma_mnk.h5 45 MB + 14 GB zeta restart tensor each) | .../run_gw/ and .../run_gw_dist/ |
| band structures | .../run_ht_{dft,qp}/bandstructure.dat |
| source WFN (NOT copied — 1024 bands, shared deck) | /scratch2/08271/jackmc/mos2_4x4_test/WFN_b1024.h5 |

Harness notes: `harness/env_common.sh` is the outer path/env block,
`harness/inner_common.sh` the container-side block (a transcription of
the certified `gw_dev.sbatch` runner, plus a per-rank /proc VmHWM
sampler because this harness runs python as a child rather than
`exec`ing it — see CLAIMS row 40 before quoting sacct MaxRSS anywhere).
`harness/run_leg.sh` is the one instrumented launch used by every leg.
