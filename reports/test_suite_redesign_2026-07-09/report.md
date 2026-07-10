# Test-suite redesign — tiered gates, from-restart invariances, fixture shrink

**Date:** 2026-07-09/10 · **Checkout:** lorrax_D, branch `agent/driver-transparency`
**Commits:** `df5befe` (charge_density jax-import fix) → `3df3476` (fixture
shrink + re-freeze) → `dccefba` (tiered suite) → `f52bbc9` (Tier-3 triage)
→ `f3a982a` (second merge pass)

## Directive

The suite must run with a PLAIN invocation — `LORRAX_NGPU=1 lxrun python3 -m
pytest -q tests` on 1 GPU — fast (target ≤ ~4 min), no xdist/srun overrides
required (xdist path kept optional; conftest worker→GPU pin kept).  Cover
every actually important feature with ~4 end-to-end fixtures max; pare
non-GW module tests to a handful.

## Result

| | old suite (d03c857) | new suite (f3a982a) |
|---|---|---|
| pytest wall (1 GPU, plain invocation) | **578.9 s** (9:38) | **218.2 s / 257.6 s** (two consecutive runs) |
| tests | 258 passed / 9 skipped | **176 passed / 24 deselected / 0 failed** (identical both runs) |
| e2e `gw.gw_jax` subprocess launches | 18 (many full-pipeline duplicates) | 13 (5 fresh full pipelines, 1 fresh bispinor pad twin, 6 cheap from-restart, 1 static-from-restart) |
| unit-test files | 37 (+2 in `active/`) | 17 (+2 gate files; +5 behind `-m extra`, deselected by default) |

Final-number runs (identical tree, back-to-back on nid001096, 1 A100):

```
run 1: 176 passed, 24 deselected, 1 warning in 218.24s (0:03:38)
run 2: 176 passed, 24 deselected, 1 warning in 257.62s (0:04:17)
```

## Architecture (what changed and why)

A GW package's suite has three jobs, and the new layout maps one file per job.

### Tier 1 — frozen e2e pins (`tests/test_gw_jax_regression.py`, 4 gates)

Fresh full-pipeline runs (ζ-fit → V_q → χ₀ → W → [PPM fit → 4-branch Σ] →
head → QP extraction → writers) against frozen references:

| gate | fixture | covers |
|---|---|---|
| `si_cohsex_3d` | Si 4×4×4, 960c | **BGW anchor** (0.12 meV MAE) — untouched byte-for-byte |
| `cohsex` | MoS2 WFNsmall, 60c | the ONLY IBZ-stored WFN (kgrid 3×3, nrk=4, ntran=12) ⇒ ψ k-unfold + 12-op group run e2e only here; nspinor=2 static SX/COH; K_POINTS path.  Kept — ~8 s, unique coverage |
| `gnppm` | MoS2 3×3, **shrunk** 399c | dynamic workhorse: minimax screening → GN-PPM fit → 4-branch Σ → analytic head → eqp writers; IBZ cascade ACTIVE and now log-asserted (frozen values alone cannot see a silent cascade deactivation — IBZ ≡ full-BZ numerically) |
| `bispinor` | MoS2 3×3 ns=2, **shrunk** 256c/209c | **upgraded COHSEX → bispinor GN-PPM**: dynamic Σ_c on screened charge W + Σ^B fold-in; 4 ζ channels, 7 V_q tiles, transverse γ̃ |

### Tier 2 — invariance gates from prepared state (`tests/test_invariance_gates.py`, 7 gates)

The gnppm Tier-1 run is a session-scoped pytest fixture; its run dir
(incl. `tmp/isdf_tensors_399.h5`) is the prepared state.  Each variant
copies the state (the driver mutates the restart file in place —
`persist_w0_and_head` writes W0 + head scalars back) and re-runs with
`restart = true`, skipping the dominant ζ-fit + V_q build (17 s fresh →
2–9 s in-process per variant).  Self-checking, no frozen refs:

| gate | legs | tolerance (measured on the shrunk fixture) |
|---|---|---|
| restart ≡ fresh | session vs restart-baseline | 1e-6 (bit-identical) |
| μ-pad flip gnppm | restart-baseline vs restart+PAD=12 | census/Σ_X exact; Σ_C ≤ 2e-4 eV (measured 0.0) |
| μ-pad flip bispinor | fresh vs fresh+PAD=4 | full bit-identity (measured identical) |
| kij ↔ kij_stream | restart-baseline vs restart+stream | 5e-12 (measured 4.5e-13) |
| SC-iter-1 ≡ one-shot | restart-baseline vs restart+SC | 1e-6 (bit-identical) |
| fixed-point rotations | restart+fixed_point vs frozen npy | 1e-6 (ref re-frozen; produced twice, bit-identical) |
| IBZ ≡ full-BZ | restart-IBZ static vs **FRESH** full-BZ static | 1e-6 (measured exactly 0) |

The IBZ gate's full-BZ leg deliberately stays a FULL fresh pipeline: the
restart V_q was IBZ-built, so restart-IBZ vs fresh-full-BZ still covers
the ζ IBZ writes and the V_q unfold (the TRS-blind-bug class), not just
the W-solve.  Cascade activation is log-asserted on both legs
(`W.slice_to_ibz` / `unfold=full-BZ` markers; the fallback warning
`orbit closure failed` asserted absent).

### Tier 3 — 17 unit files + `extra` marker

Only what gates cannot see (see triage table).  `-m extra` (deselected by
default via pyproject `addopts = "-m 'not extra'"`; an explicit CLI `-m`
overrides): tooling / experimental / out-of-repo-fixture suites.

Ordering/parallel safety: session state is never mutated (variants copy
first) ⇒ order-independent.  xdist still works (worker→GPU conftest pin
kept); session fixtures replicate per xdist worker — correct but
partially redundant, an accepted xdist-only cost (plain serial is the
contract).

## Fixture shrink results

### gnppm_debug (the workhorse) — 642c/80b → 399c/46b

| knob | old | new |
|---|---|---|
| centroids (orbit-closed) | 642 | **399** (`kmeans_cli 400 --seed 42`, orbit-aware) |
| ncond / nband | 54 / 80 | **20 / 46** (nval = 26 fixed) |
| WFN.h5 | 45 MB, 82 bands | unchanged — truncation would not change runtime (the loader reads only the requested band window) and would add a fresh ~27 MB blob to git history |
| kgrid | 3×3 | unchanged (IBZ cascade keeps its 5/9 structure) |

Validation grid (all 1 GPU A100; details in the fixture README):

* fresh run ×2 → **bit-identical** (sigma_diag, eqp0/1; timestamp-only diff)
* `LORRAX_EXTRA_MU_PAD=12` → Σ_X/Σ_C/Σ_XC **bit-identical**, eqp |Δ| = 3e-9
  (the OLD 642c fixture drifted 6.3e-5 eV on Σ_C through the near-singular
  PPM fit; the shrunk census has no mode at the validity threshold — pad
  gate bound kept at 2e-4 eV for defect headroom)
* IBZ vs `LORRAX_FORCE_FULL_BZ=1` static: **exactly 0** on all Σ columns;
  cascade active (`n_q_ibz=5`, `unfold=IBZ→full`)
* restart ≡ fresh **bit-identical**; static-COHSEX-from-dynamic-restart
  works (2.0 s in-process), bit-identical to fresh static
* kij↔kij_stream 4.5e-13; SC-1 ≡ one-shot bit-identical
* GN-PPM census healthy: invalid modes 8378/1432809 (0.58 %)
* re-frozen same-code: `sigma_diag_gnppm_ref.dat`,
  `eqp_rotations_fixedpoint_ref.npy` (from-restart fixed_point ×2,
  bit-identical; carries the pre-toggle 620b501 eigh-family pin forward)

### bispinor_debug — COHSEX 640c/668c → **GN-PPM** 256c/209c

The directive's question "does bispinor × gn_ppm run at HEAD?" — **yes**,
with one lost-wiring find: the dynamic head resolver needs
`whead_imfreq` alongside `vhead`/`whead_0freq` (all 0.0 in this fixture's
explicit-bypass scheme); without it the run dies in `persist_w0_and_head`
at the imaginary probe frequency.  With it, bispinor GN-PPM runs
end-to-end (Σ^B folds into sigX; deterministic), so the gate was upgraded
to bispinor GN-PPM (small ω-grid: −4…4 eV, step 1.0) and frozen.

| attempt | result |
|---|---|
| nband 32 → 16 | **fails, correctly**: 26 occupied states ⇒ band edges (0,22,26,30) need nband ≥ 30; `BandSlices` raises.  nband stays 32 |
| centroids 640/668 → 256/209 | 116 s → **40 s** in-process (the four ζ-fits: 93 s → 28 s) |
| centroids → 128/104 | 39.5 s — **no further gain**: the floor is per-channel r-chunk streaming over the 30×30×120 grid at 60 Ry (ngkmax 5545), not centroid count.  Kept the better-conditioned 256/209 |

Properties preserved: charge set NON-orbit-closed (`--no-orbit`; charge
tiles full-BZ-direct, closure-fallback warning asserted in the gate),
transverse set orbit-closed (TT tiles `unfold=IBZ→full`) — same as the
old fixture.  Validation: rerun **bit-identical**; `EXTRA_MU_PAD=4` (the
historic 668→672 transverse-LU catastrophe class) **bit-identical**
including Σ_C; census 18494/589824 invalid (3.14 %).

Coverage delta from the mode change: static Σ_SX/Σ_COH on W⁰⁰ with
*bispinor* wavefunctions is no longer separately e2e-pinned; the static
kernels stay pinned by `cohsex` (nspinor=2) and `si_cohsex_3d` (scalar +
BGW anchor), and all bispinor-specific machinery (Σ^B, 4 ζ channels, 7
V_q tiles, transverse γ̃) is in the GN-PPM gate.

## Pin mapping — all 13 pre-redesign gates

| # | old gate | new home | notes |
|---|---|---|---|
| 1 | `test_gw_jax_matches_reference[cohsex]` | same, Tier 1 | unchanged fixture + ref |
| 2 | `test_gw_jax_matches_reference[gnppm]` | `test_gnppm_matches_reference` | shrunk fixture, same-code re-freeze + NEW cascade log assert |
| 3 | `test_gw_jax_matches_reference[si_cohsex_3d]` | same, Tier 1 | untouched (BGW anchor) |
| 4 | `test_gw_jax_matches_reference[bispinor]` | `test_bispinor_gnppm_matches_reference` | mode upgraded COHSEX → GN-PPM; new frozen ref (coverage delta above) |
| 5 | `test_gnppm_fixed_point_reproduces_frozen_qp_rotations` | Tier 2 `test_fixed_point_frozen_qp_rotations` | now from restart; ref re-frozen same-code |
| 6 | `test_ibz_full_bz_equivalence` | Tier 2 `test_ibz_equals_full_bz` | IBZ leg from restart + cascade log asserts; full-BZ leg still a fresh full pipeline |
| 7 | `test_g1_kij_vs_kij_stream_parity` | Tier 2 `test_kij_stream_parity` | from restart; same 5e-12 pin |
| 8 | `test_g2_branch_window_tiles_are_frozen` | `test_sigma_ppm_gates.py` (G2 only) | unchanged (synthetic freeze) |
| 9 | `test_mu_pad_extent_invariance[gnppm_pad12]` | Tier 2 `test_mu_pad_flip_invariance_gnppm` | from restart; identical assertions |
| 10 | `test_mu_pad_extent_invariance[bispinor_pad4]` | Tier 2 `test_mu_pad_flip_invariance_bispinor` | fresh ×2 (bispinor restart unsupported — gw_init.py); now on the GN-PPM fixture, still full bit-identity |
| 11 | `test_sc_iteration1_equals_one_shot` | Tier 2, same name | from restart (two cheap runs instead of two full pipelines) |
| 12 | `test_restart_mu_pad_roundtrip` | kept as-is (Tier 3) | synthetic writer/loader roundtrip, already cheap |
| 13 | G3 head negative-branch (`test_head_correction.py`) | kept as-is (Tier 3) | pure numpy |

NEW pins that did not exist before: restart ≡ fresh (Tier 2); IBZ-cascade
activation log assert (Tier 1 gnppm); the bispinor dynamic-Σ path (Tier 1).

## Triage table — every pre-redesign test file

Old per-file wall from the measured baseline (`--durations=60`, 1 GPU;
sub-0.3 s files shown as ~0.1–0.2).

| file (old) | s | verdict | protection now lives in |
|---|---|---|---|
| test_gw_jax_regression.py | 196.7 | rewritten | Tier 1 (4 gates) + Tier 2 ports (rows 5–6 above) |
| test_mu_pad_invariance.py | 242.9 | **deleted** (ported) | Tier 2 pad-flip gates (rows 9–10) |
| test_sc_oneshot_equivalence.py | 37.5 | **deleted** (ported) | Tier 2 (row 11) |
| test_sigma_ppm_gates.py | 37.0 | trimmed to G2 | G1 → Tier 2 (row 7) |
| test_kmeans_sharded.py (694 L) | 11.4 | **deleted** → `test_kmeans_smoke.py` | ONE hex end-to-end driver test; kmeans is a fixture-regen tool — breakage fails loudly at regen.  Unit-level PBC/metric/sharded-parity checks dropped (accepted per directive) |
| test_rchunk_gflat_pair.py | 6.8 | **deleted** | accumulate/sphere round-trips → `test_compute_all_V_q_g_flat` (which absorbed the per-q sphere tests); to_rchunk/phase identities → `test_wfn_transforms`; e2e → Tier-1 gates |
| test_zq_from_psi_sm_bit_identity.py | 6.1 | kept (+psi_g_store) | core z_q streaming kernel vs one-shot reference incl. short-bc / γ¹ / pseudobands edge cases no gate reaches |
| test_wfn_transforms.py | 5.9 | kept | FFT/gather transforms vs independent numpy references |
| test_sternheimer_solvers.py | 5.0 | **`-m extra`** | psp/orbital-mag tooling, not the GW pipeline |
| test_pivoted_cholesky.py | 5.0 | **deleted** | fixture-generation tool (kmeans prune path); loud wrapper guards at regen + kmeans smoke.  Unit-level protection dropped, accepted |
| test_wfn_loader_eager.py | 3.5 | kept (+phdf5_clamp) | loader contract + bispinor lift; clamp = the 16-GPU H5Dread crash regression |
| test_compute_V_q_bispinor_g_flat.py | 3.3 | kept (+v_q_bispinor_helpers) | 7-tile einsum reference + TT projector identities |
| test_head_wing_schur.py | 0.8 | **`-m extra`** | gw/experimental module (staged for promotion) |
| test_restart_pad_roundtrip.py | 0.7 | kept | disk-stores-LOGICAL-μ-extent contract |
| test_io_callback_nested.py | 0.7 | **deleted** | the same io_callback×scan×shard_map×all_gather nesting is exercised on the production kernel by test_zq_from_psi_sm_bit_identity |
| test_per_q_sphere.py | 0.5 | merged | → test_compute_all_V_q_g_flat |
| test_compute_all_V_q_g_flat.py | 0.4 | kept | V_q orchestrator vs per-q einsum reference |
| test_aot_memory.py | ~0.3 | **`-m extra`** | planner tooling; the GPU cuFFT probes container-skip anyway |
| test_band_chunk_size_floor.py | ~0.2 | kept | world-size band-chunk floor (16-GPU all_gather-dim-zero crash class) |
| test_band_partition.py | ~0.2 | kept | QSGW off-diag masking + scissor — invisible to one-shot gates |
| test_head_correction.py | ~0.2 | kept | G3: PPM head negative-branch sign (Bug A) + static head closed forms |
| test_mf_isdf_header_roundtrip.py | ~0.2 | merged | → test_file_io (home of `_make_fake_wfn`) |
| test_minimax_assets.py | ~0.2 | merged | → test_minimax_quadrature |
| test_real_axis_quadrature.py | ~0.2 | merged | → test_minimax_quadrature |
| test_planner_refit_2026-05-17.py | ~0.2 | **deleted** | dated model-refit archaeology; the load-bearing floor lives in test_band_chunk_size_floor; planner regressions at scale manifest as OOM/crash, accepted |
| test_psi_g_store.py | ~0.1 | merged | → test_zq_from_psi_sm_bit_identity (slicer contracts under the kernel that uses them) |
| test_q_ibz_and_centroid_perm.py | ~0.1 | merged | → test_symmetry_unfold |
| test_trs_unfold_centroid_perm.py | ~0.1 | merged | → test_symmetry_unfold |
| test_unfold_psi_trs.py | ~0.1 | merged | → test_symmetry_unfold (TRS-blind-bug unit guards; ψ-unfold otherwise e2e-covered ONLY by the cohsex gate) |
| test_qp_solver_config.py | ~0.1 | kept | pure-python config/toggle validation |
| test_R_proper_cri3.py | ~0.1 | **`-m extra`** | needs out-of-repo CrI3 WFN + reports npz; skips where absent |
| test_sigma_x_bispinor.py | ~0.1 | kept | γ̃ algebra + (0,0)-no-op scalar-reduction guard |
| test_slab_io_ffi_contract.py | ~0.1 | merged | → test_file_io |
| test_v_q_bispinor_helpers.py | ~0.1 | merged | → test_compute_V_q_bispinor_g_flat |
| test_wfn_loader_phdf5_clamp.py | ~0.1 | merged | → test_wfn_loader_eager |
| test_zeta_loader.py | ~0.1 | merged | → test_zeta_reader → test_file_io (pad/valid_mu cases all kept; ONE near-duplicate header-surface test dropped) |
| test_zeta_reader.py | ~0.1 | merged | → test_file_io |
| active/test_eqp_bgw.py | ~0.1 | moved | tests/test_eqp_bgw.py (BGW eqp.dat byte-contract); `tests/active/` removed |
| active/test_reshard_all_to_all.py | skip | **`-m extra`** | perf-pattern check (all-to-all emission); container-skipped anyway |

## Timing per tier (old vs new, run 1)

| tier | old | new |
|---|---|---|
| Tier-1-equivalent e2e | 350.4 s (incl. 309 s of bispinor COHSEX runs) | 86.3 s (bispinor 42.7, si 21.1, gnppm 14.4, cohsex 8.1) |
| Tier-2-equivalent gates | ~163 s (full-pipeline duplicates) | 100.3 s (bispinor pad twin 44.5 dominates; 6 restart variants 8–14 s each) |
| Tier 3 units + collection | ~65 s (37 files) | ~31.6 s (17 files) |
| **total** | **578.9 s** | **218.2 s** |

## Surprises / notes

1. **bispinor GN-PPM lost wiring**: any dynamic run using explicit head
   overrides needs `whead_imfreq` too — undocumented input key; the gate
   input now carries it.
2. **`charge_density.py` NameError**: the wfn_ibz ρ fallback (no QE save
   dir) was missing `import jax` — first exercised by the fixture regen;
   fixed in `df5befe`.
3. **Shrink floors**: gnppm fresh wall is launch/compile-dominated (~17 s
   in-process at 642c and 399c alike); the bispinor floor is r-chunk
   streaming over the FFT grid (equal wall at 256c and 128c).  Further
   speedups came from restart variants, not smaller fixtures.
4. **Static-from-dynamic-restart works**: a static COHSEX input restarts
   from a GN-PPM run's tensors file (2 s in-process) — this is what makes
   the IBZ gate cheap.
5. **Pad-flip became bit-identical on BOTH shrunk fixtures** (the old
   gnppm fixture had a real 6.3e-5 eV Σ_C pad sensitivity through
   near-singular PPM modes; neither shrunk census has a mode at the
   validity threshold).  The 2e-4 eV gate bound is retained for defect
   headroom.
6. Suite writes ~0.8 GB of restart-state copies under the pytest tmp root
   (6 × 136 MB); node-local tmp absorbs this.
7. `skills/checkpoint/SKILL.md` updated: plain 1-GPU invocation is the
   standard, golden-gate names refreshed, `deselected` note added.
   `KNOWN_SANDBOX_ERRORS.md`: module-load-in-non-login-shell entry added.
8. xdist sanity (optional path, never required): 4-GPU `-n 4 --dist load`
   run after the serial runs — **176 passed in 172.0 s (2:52)**; session
   fixtures rebuild per worker as designed.
