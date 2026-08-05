# BSE figures — MoS2 GN-PPM GW + BSE (agent/bse-figures) — WORKLOG

Run dir: `runs/MoS2/A_bse_figures_2026-07-20/`
Worktree: `sources/worktrees/lorrax_A_figures` (branch `agent/bse-figures`, base `agent/bse-kgrid` @ 964c682)
Allocation: 56222279 (fresh 4-node/16-GPU; the handed 56206654 had <15 min left).

STATUS: **figures HELD** pending the coordinator's GN-PPM scissor-evaluation decision
(reports/gw_scissor_eval_2026-07-20/). GW outputs left intact for that eval.

## Route decision (STEP 0)
- Existing NSCFs (04_mos2_12x12, A_bse_w0_resolvent/mos2_6x6) both have **mnband=82** — far
  short of the ~200 needed. Regenerated a fresh NSCF.
- **Economy route chosen: 6×6 GN-PPM GW → 12×12 BSE via `bse_k_grid 12 12 1`.**
  Verified in code (bse/bse_io.py:587-732): the fine-grid ψ/ε are generated from the COARSE
  ISDF `ctilde` by the htransform (`compute_wfns_fi(kgrid_co=coarse, kgrid_fi=fine)`), W is
  coarse→fine zero-padded (`pad_W_R_to_grid`), V_q0 rebuilt at the fine mini-BZ head. **No
  separate 12×12 WFN is needed** — only the 6×6 coarse WFN + GW restart. So only ONE 6×6
  200-band NSCF was built.

## QE (SCF 12×12 → NSCF 6×6, 210 bands) — done, verified
- N_val = 26 (Mo z=14 + 2·S z=6). ifmax=26, nrk=36, nspinor=2, mnband=210. DFT gap 1.73 eV
  (highest occ −5.275, lowest unocc −3.543) — sensible PBE MoS2.
- Timing: SCF 22s, NSCF 21s, pw2bgw 35s, wfn2hdf 16s (~100s total).

## GN-PPM GW producer (00_lorrax_gw_gnppm) — runs clean, QP numbers partly unphysical
- compute_mode=gn_ppm, use_ppm_sigma=true, nval=26, ncond=74 (σ for bands 1..100),
  nband=200 (sum-over-states), 30 Ry, screening_method=minimax, ppm_omega_p=2.0.
- Band-window mapping (common/meta.py): sigma range = [nelec−nval, nelec+ncond) = [0,100);
  sum-over-states = nband = 200.
- Preprocessing gotchas found (workarounds applied):
  1. **Centroid pruning OOM**: default `--oversample 1.5` OOMs the pivoted-Cholesky Gram on
     1 GPU at 1600 centroids. Fix = `--oversample 1.0` (documented CrI3 workaround).
  2. **dipole.h5 IS required** for the GW: the q=0 Coulomb head defaults to
     `wcoul0_source=s_tensor`, built from the dipole S(ω) tensor (gw/head_correction.py:152).
     Skipping dipole → "Failed to resolve q=0 Coulomb head". (gw_jax itself reads only
     kin_ion.h5, but the head resolver reads dipole.h5.)
- Timing (16 GPU): kin_ion 21s + dipole 60s + gw_jax 129s ≈ 3.5 min.

### IBZ-cascade / TRS symmetry bug (matches memory `project_trs_blind_sym_bug`)
- Orbit-aware centroids (default) → the ζ/V_q/Σ IBZ-only cascade activated
  ("q-IBZ reduction: 20 IBZ / 36 full-BZ") → the buggy IBZ→full-BZ V_q unfold →
  symmetry-breaking, erratic Σ.
- **Workaround applied**: regenerated centroids with `--no-orbit` → orbit closure fails →
  "falling back to full-BZ on disk" → all 36 q computed directly (cascade bypassed).
  Centroids: `centroids_frac_1600.txt` (orbit off).

### Raw eqp gap verification (post full-BZ; independent of interpolation)
- **Direct gap at K (k-blk 14, K=⅓,⅓,0) = 2.61 eV** — DFT 1.73 → GW 2.61, **opens correctly**
  (K VBM −5.655, K CBM −3.045). Near-gap GW at K is sound.
- **Raw global gap = min_k Eqp(b27) − max_k Eqp(b26) = 1.436 eV (nominally closing)** — but
  corrupted by a spurious Γ VBM outlier: band 26 at Γ pushed UP +0.86 eV (Eqp −4.537 vs
  Edft −5.399, wrong sign). VBM ranking: Γ −4.537 (outlier), k11/k31 −5.514, K −5.655.
- Discounting the Γ outlier → VBM at K → gap = **2.55 eV** (in the 2.5–2.8 eV G0W0 range).
- Residual defects (per-k self-energy, NOT interpolation): symmetry-breaking between
  equivalent k (k1=(0,⅙)/k6=(⅙,0) differ 0.7 eV in Eqp; V_H itself differs 0.77 eV there);
  sig_c(Edft) imaginary parts blow up (1e2–1e5 eV) at several k; some sig_c wrong-signed.
- Context: LORRAX GW→eqp for MoS2 has never produced a validated global gap in this sandbox
  (runs/MoS2/01_mos2_4x4_cohsex_gnppm variants C–J, all nonsense gaps). The far-from-gap
  GN-PPM Σ−Vxc (semicore, high conduction) that must stay in the htransform subspace is the
  suspect — being evaluated by reports/gw_scissor_eval_2026-07-20/.

## htransform interpolation (GW-bands figure machinery) — works, window-sensitive
- DFT bands via htransform: interp window [0,50] gave ctilde-orthogonality 2e-14, **on-grid
  recon 0.00 meV** (perfect). Window [0,90] gave orthogonality 0.207, recon 465 meV (the
  run-10 "too many interp bands" failure). Modest window required.
- CAVEAT (coordinator): htransform needs a CONTIGUOUS subspace incl. the deep semicore; the
  QP scissor fed into fH carries extreme far-from-gap corrections (band 1 shift −13 eV; bands
  39–77 shift +11…+186 eV) that corrupt the QP interpolation. Resolution pending scissor eval.

## BSE exciton (01_lorrax_exciton_bands) — set up, NOT run (held)
- exciton.in: bse_k_grid=12 12 1, 8v8c BSE window (CLI --n-val 8 --n-cond 8). Uses DFT
  energies (driver does NOT apply QP scissor) → independent of the GW eqp issue.
- **Open code concern (exciton_bands + bse_k_grid)**: driver passes `kgrid_co=(nkx,nky,nkz)`
  from the DENSIFIED data (=fine 12×12) to `compute_wfns_fi` (exciton_bands.py:472) while its
  `ctilde` is from the coarse 6×6 WFN → `build_fH_R` ifftn reshapes 36-k into 12×12 → shape
  crash expected. Loader densification itself uses the correct coarse grid (bse_io.py:659).
  Likely one-line fix: `kgrid_co=(int(meta.nkx),int(meta.nky),int(meta.nkz))`. NOT yet
  confirmed empirically (1-GPU smoke OOM'd at 62 GB unsharded densification; needs 16 GPU).

## RESUMED after coordinator "blocker resolved" — recovered-D3h + 16 GPU (mesh bug)
Coordinator: the catastrophic Σ_c was a DEVICE-COUNT bug (4-GPU/2×2 mesh); on 16 GPU with
the recovered-D3h centroids (`_vhsym_exp/kmeans_wd/centroids_frac_1496.txt`) Σ_c(valence) +
V_H are good.  New production GW dir: `02_lorrax_gw_d3h_16gpu/` (1496 recovered centroids, 16 GPU).

### Sanity-gate on 02 (recovered-D3h, 16 GPU, IBZ cascade active):
- GATE 1 (Re sigC(Γ VBM, k0 n25)) = **+0.197 eV** ✓ (target +0.2, not −47) — valence Σ_c GOOD.
- GATE 3 (V_H symmetry across VBM k-star) = spread ~1e-4 eV ✓ (k1/k5/k6 all 427.8026) —
  the V_H-fix landed (was 0.77 eV split on non-recovered centroids).
- GATE 2 (K direct gap) = **FAIL**: conduction Σ_c still broken — sigC.Re(CBM, K n26) = −4.48 eV
  (valence sigC +0.57), pushes conduction BELOW valence → sorted-QP gap 0.000 eV (metallic/inverted).
  Per coordinator STOP rule, did NOT make the GW figure from this Σ_c.
- Re-running 02 with `LORRAX_FORCE_FULL_BZ=1` (full-BZ zeta, needed by the BSE; may also bypass
  the IBZ-unfold path in Σ_c → re-checking the conduction gate).

## LORRAX code fixes applied (branch agent/bse-figures)
1. `src/bse/exciton_bands.py:472` — `kgrid_co` for the Q-path htransform must be the COARSE
   ctilde grid (`meta.nkx/nky/nkz`), NOT the densified fine grid from `data` (that is the
   `bse_k_grid` fine grid, correctly used by `k_frac`).  Without this, `bse_k_grid` fine
   densification crashes in `build_fH_R`'s ifft-reshape.  No-op when bse_k_grid off.  CONFIRMED
   empirically (crash before fix; got past it after).
2. `src/bandstructure/htransform.py:108` — `streaming_galerkin_solve` trims the n_μ
   sharding-pad (`psi_rmu_Y[..., :n_mu]`) before the dense SVD.  `load_centroids_band_chunked`
   pads n_μ to a multiple of the device count (1496→1504 on 16 GPU) but the reshape used the
   true n_mu → crash for centroid counts not divisible by the device count.  CONFIRMED (crash
   before; got past after).  No-op when divisible.

## BSE blockers resolved in sequence (all on 16 GPU):
- kgrid_co crash → fixed (#1).
- htransform 1496-pad reshape crash → fixed (#2).
- vq_interp "needs FULL-BZ zeta (has nq=20 IBZ)" → recovered-D3h has orbit closure → IBZ
  cascade → IBZ zeta.  Fix: re-run GW with `LORRAX_FORCE_FULL_BZ=1` → full-BZ zeta.

## BSE run — additional blockers past the 3 above (all real, resolved):
4. `run_gates` (vq_interp reference gate battery) OOMs at 1496 centroids (58 GB replicated
   alloc).  Fix: env opt-out `LORRAX_SKIP_VQ_GATES=1` added in `bse/vq_interp.py`
   build_vq_evaluator (default keeps gates; fit + driver on-grid gate still validate).
5. vq_interp C_q eigh shards the 36 q-points (6×6) over the flattened 16-device mesh — 36 not
   divisible by 16.  This is a device/q-count sharding constraint, UNRELATED to the GW Σ_c mesh
   bug.  The BSE is mesh-invariant (prior runs 10@4-GPU ≡ 11@16-GPU).  36 IS divisible by 4 →
   ran the BSE on **4 GPU (2×2 square mesh, cusolverMp)**.  (16-GPU would need q-axis padding to
   48 in prepare_coarse — deferred; not a physics issue.)

## BSE SUCCESS — all new infra confirmed working (4 GPU, 2×2, cusolverMp):
- `[bse_k_grid] coarse 6x6x1 → fine 12x12x1 (144 k-pts)` — coarse→fine densification.
- `[bse_k_grid] V_q0 exchange tile via vq_interp eval_vq(Q=0), fine mini-BZ head <v_LR>=13.23`.
- `[bse_k_grid] W zero-padded in R 6x6x1→12x12x1 (exact trig-interp)` — coarse-W pad.
- `[lorrax cusolverMp] library 0.7.2, grid: 2x2` — distributed eigh.
- htransform: nk=36, nb=48, rank=1728 (clean); BSE window 8v8c = bands [26,34) + 14 guards.
- Q path: 56 points Γ-M-K-Γ; block-Lanczos solve running.

## Gates
- CPU golden gates PASS: test_coarse_w_pad.py + test_bse_kgrid.py::test_parse_grid_spec (13 passed).
- GPU gates (test_exciton_bands, test_bse_kgrid GPU, test_gw_jax_regression) pending.
