# V_q Consolidation — Birds-Eye Audit

**Repo:** `sources/lorrax_D` @ `agent/memplanner-cleanup`
**Scope:** the ~3000-line r-space "tile" V_q deletion + the surviving G-flat-only V_q path.
**Method:** 4-dimension adversarial audit (gflat-correctness / cutoff-feature / residuals / quality), every finding re-verified against source; plus direct source reads of the cutoff plumbing.
**Date:** 2026-07-02

---

## 1. Verdict — sound, but only *partly* high-standard

**Partly.** The consolidation is **functionally sound**: the deletion left exactly one live V_q path (G-flat), it is coherent, correctly guarded, and covered by targeted unit tests. There is **no blocker and no HIGH finding** — every "medium" the audit raised collapsed to "low" on verification, because all of them are dead code or stale prose with **zero runtime/correctness impact**. Where it falls short of "high standard" is *cleanliness*: the deletion pass did not finish its own job. It left a whole superseded FFT-era kernel trio (~135 lines) in `compute_vcoul.py` that **duplicates the live contract kernel** — a direct violation of this codebase's own single-source-of-truth / no-parallel-paths rule, which is the entire point of a consolidation PR — plus dead parameters on the live seam and ~a dozen dangling references (some with fabricated line numbers) to the now-deleted `v_q_tile.py`, including in the two authoritative navigation docs. None of that is a bug; all of it undercuts the claim that this is a finished, high-standard consolidation. Fixable in one focused cleanup pass.

---

## 2. Blockers / HIGH findings

**None.** No blocker, no HIGH.

The one finding that touches a *physics-correctness claim* (rather than pure hygiene) is worth naming explicitly even though it is low-severity today:

- **Bispinor CC tile is not bit-identical to the scalar V_q for 3D + `mc_average_vcoul_body`.** The scalar path (`compute_all_V_q_g_flat`) builds a mini-BZ-averaged head and injects `v_head_miniBZ` into `compute_v_q_per_G`; the bispinor CC builder (`v_q_bispinor.py:127`) calls the same function with **no head table**, so the `G=0` body slot stays `8π/|q|²`. `gw_init.py:809` asserts the CC tile "is bit-identical to the scalar charge V_q" and then reads it back as the downstream scalar V_q (`:913`). For a 3D magnet with `mc_average` on, these diverge. **Harmless right now** — every sandbox bispinor system (CrI3, VI3) is `sys_dim=2`, where the head term is a no-op — and even in 3D the effect is a small body-averaging refinement of one `G=0` slot per `q≠0`, not a wrong answer. But the load-bearing comment is currently **false in general**; it should be either honored (plumb the head into the CC builder) or narrowed to state the `sys_dim=2` / no-mc-average precondition.

### Cutoff feature — does the 2×-|G| range work end-to-end?

**Yes, the plumbing is coherent, guarded, and unit-tested** (I read the path directly; I did not re-run it — see §4).

- `bare_coulomb_cutoff` defaults to `ecutrho` (`gw_init.py:578`), i.e. ~4·ecutwfc in energy = **2× the wavefunction G-radius**, which is the intended "2×-|G|" Coulomb sphere.
- The sphere is built by `compute_per_q_bare_coulomb_components` (`common/coulomb_sphere.py`) as `{G : |q+G|² ≤ vcoul_cutoff_ry}` per IBZ q, padded to `ngkmax` with a sentinel — bit-exactly matching the WFN.h5 per-q layout, and unit-tested against an independent reference in `tests/test_per_q_sphere.py`.
- The V_q kernel masks `v(q+G)→0` outside the sphere (`compute_vcoul.py:336`, `vcoul_cutoff_ry` gate) and reads ζ̃ at every G inside it.
- The one real end-to-end hazard — ζ stored on a *narrower* sphere than the Coulomb sum needs — is **explicitly guarded**: `gw_init.py:582-587` raises a clear `ValueError` if `bare_coulomb_cutoff > zeta_cutoff`, and both are capped at `ecutrho`.
- `tests/test_compute_all_V_q_g_flat.py` drives the full G-flat charge path with a finite cutoff and checks it against a reference builder; the bispinor equivalent is `tests/test_compute_V_q_bispinor_g_flat.py`.

So the cutoff feature is wired correctly and defended against the obvious misuse. Caveat: no BGW-vs-LORRAX numerical parity run for the 2×-|G| cutoff was executed *in this audit* (§4).

---

## 3. Ranked cleanups that would genuinely raise the standard

1. **Delete the dead FFT-era kernel trio in `compute_vcoul.py`** — `fft_integer_axes` (39-51), `_v_q_per_q_g_chunked_jit` (58-94), `compute_v_q_per_q_g_chunked` (97-172): **zero callers** anywhere in `src/` or `tests/`, and `_v_q_per_q_g_chunked_jit.body()` computes the *identical* `V += conj(L)·v @ R.T` G-chunked contract as the live `_g_chunk_body` (`v_q_g_flat.py:112-121`, whose own comment says it "Replaces the historical static Python loop"). This is the single biggest standard-raiser: ~135 lines, removes the parallel old/new implementation and the latent edit-the-wrong-copy hazard, and lets the stale module docstring (lines 1-20, still describing "μ-chunked FFT / ν-chunked contraction / redundant FFT work" that nothing here does) be rewritten to describe what survives (`compute_v_q_per_G`, `build_v_head_miniBZ_avg_3d`, `compute_all_V_q`). *(Keep `compute_vcoul.py` — it is NOT a dead file.)*

2. **Strip dead parameters from the live seams and fix the "two paths" docstring.**
   - `compute_all_V_q` (`compute_vcoul.py:342`) still accepts `n_rmu`, `n_rtot`, `budget_bytes`, `use_g_flat_zeta` — none forwarded; dispatch keys off `zeta_layout`, not `use_g_flat_zeta`. Its docstring still advertises a second branch to the deleted `v_q_tile.compute_V_q_tile`, but the body just raises `NotImplementedError`. Keep it as a thin adapter, drop the four params, rewrite the docstring to "single live path + raises for legacy layouts."
   - Note the audit's alarming phrasing "memory budget no longer bounds V_q" is **overstated**: working-set memory *is* bounded by the live `cfg.memory.vq_g_chunk_size` knob (`gw_init.py:957`) and mesh-sharded ζ slabs. `budget_bytes` is a superseded knob, not a lost safety rail — just delete it.
   - On the G-flat seam: `bvec` on `_compute_V_q_g_flat_one_tile` (`v_q_g_flat.py:276`) is unused (the closures already capture it; both call sites pass it needlessly); `async_prefetch` on `compute_all_V_q_g_flat` (488) is a documented no-op. Remove both.

3. **Resolve the bispinor CC bit-identity claim** (the §2 item) — either inject `v_head_miniBZ` into the CC builder so it truly matches the scalar V_q in 3D, or narrow the `gw_init.py:809` comment to its actual precondition (`sys_dim=2` / no mc-average). This is the only cleanup touching a correctness *assertion* rather than pure prose.

4. **Sweep the dangling `v_q_tile.py` references.** ~a dozen docstrings/comments still cite the deleted module, several with concrete-but-fabricated line numbers: `v_q_g_flat.py` (lines 4, 27-28, 593/604/622/639/654), `v_q_bispinor.py` (500, 518), `common/coulomb_sphere.py` (4-11, advertising the deleted `compute_bare_coulomb_sphere_idx`), `file_io/zeta_reader.py:34`, `centroid/orbit_syms.py:264`, `common/isdf_fitting.py:2194`, `common/wfn_transforms.py:1255`. `_unfold_g0_ibz_to_full` now lives locally (`v_q_g_flat.py:561`, and its own provenance comment at :559 is already correct); `_unfold_v_q_ibz_to_full` is gone (replaced by `common.symmetry_maps.unfold_v_q`). Point readers at the real locations.

5. **Update the authoritative docs.** `docs/architecture/codebase.md` (lines 318, 509-510, 598) and `docs/theory/physics.md` (310, 316) still name the deleted `compute_all_V_q_from_zeta_h5` / `make_v_munu_chunked_kernel` as the canonical V_q entry point + child kernel. Because these are the read-order navigation docs, the staleness actively misdirects. Live entry is `compute_all_V_q → v_q_g_flat.compute_all_V_q_g_flat` (charge) / `v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5` (bispinor). *(`docs/theory/isdf-zeta-vq.md` is already correct.)*

6. **Fix the broken import in `tests/archive/test_chunked_wfn_loading.py:54-56`** (imports deleted `compute_V_q_from_zeta_h5` / `compute_all_V_q_from_zeta_h5`). Trivial and CI-invisible — `pyproject.toml:61` `norecursedirs=["archive"]` means pytest never collects it — but it is a real dead reference; delete or repair.

---

## 4. Verified vs. unverified

**Verified by direct source read in this audit:**
- `src/gw/v_q_tile.py` is deleted; the live V_q path is G-flat only (`compute_all_V_q` → `compute_all_V_q_g_flat` / `compute_V_q_bispinor_g_flat_to_h5`), non-G-flat branch raises.
- The FFT-era kernel trio in `compute_vcoul.py` has **zero callers** (grep across `src/` + `tests/`); it duplicates the live `_g_chunk_body`.
- Cutoff plumbing is coherent and **guarded**: default `bare_coulomb_cutoff = ecutrho` (2×-|G|), `bare_coulomb_cutoff ≤ zeta_cutoff ≤ ecutrho` enforced with a `ValueError`, `vcoul_cutoff_ry` masks `v` outside the sphere.
- Sphere construction (`test_per_q_sphere.py`) and the full G-flat charge/bispinor V_q assembly (`test_compute_all_V_q_g_flat.py`, `test_compute_V_q_bispinor_g_flat.py`) have dedicated unit tests that exercise a finite cutoff.
- The bispinor CC builder omits `v_head_miniBZ`; the scalar path injects it — the two genuinely diverge for 3D + mc_average.
- All ~dozen dangling `v_q_tile.py` doc/comment references exist verbatim as cited.

**Not verified (honest gaps):**
- **I did not execute pytest** (login-node compute restriction). I confirmed the tests *exist and target the right surfaces*; I did not observe them pass. "Tests pass" is an inference, not an observation from this session.
- **No BGW-vs-LORRAX numerical parity run** for the 2×-|G| cutoff was executed here. The G-flat reference report (`reports/zeta_v_q_g_flat_reference_2026-05-12/`) exists but was not re-run. The cutoff feature is verified *structurally*, not *numerically*, in this audit.
- **The 3D + mc_average CC divergence is unverified numerically** — no 3D bispinor system exists in the sandbox to trigger it, so its magnitude is reasoned-about, not measured.

---

**Bottom line:** the V_q consolidation is correct and the surviving path is well-formed and defensible; it is not yet *clean*. One deletion (item 1) plus a param/doc sweep (items 2, 4, 5) would close the gap between "sound" and "high standard," and item 3 is the one place to double-check a physics claim before calling it done.

---

## RESOLVED — 2026-07-02 (commit `6809e64`)

All 6 cleanups executed + verified (3 e2e gates + 241 unit passed, only the 4 known
container-env failures). Net −1201 L. Bispinor separately VERIFIED e2e (bit-identical
to the pre-deletion reference — see `BISPINOR_E2E_2026-07-02.md`).
- **#1** dead FFT-era trio (`fft_integer_axes`/`_v_q_per_q_g_chunked_jit`/`compute_v_q_per_q_g_chunked`,
  ~135 L) deleted + 7 orphaned imports + module docstring rewritten.
- **#2** dead params stripped (`compute_all_V_q`: n_rmu/n_rtot/budget_bytes/use_g_flat_zeta
  + call site + orphaned m_budget; `_compute_V_q_g_flat_one_tile`: bvec; `compute_all_V_q_g_flat`:
  async_prefetch — the last cascaded to 3 test sites + a now-vacuous async-vs-sync test, deleted).
- **#3** bispinor CC bit-identity comment narrowed to sys_dim=2/no-mc-average (comment-only).
- **#4** ~dozen dangling v_q_tile refs swept across 8 files.
- **#5** nav docs (codebase.md ×3, physics.md ×2) repointed to the live entry chain.
- **#6** deleted tests/archive/test_chunked_wfn_loading.py (imported deleted surfaces + an old package layout).

**Remaining honest gaps (not closed here):** no BGW numerical-parity run for the 2×|G|
cutoff (verified structurally); the 3D+mc-average CC divergence is unmeasured (no 3D bispinor
system exists to trigger it). Both are future numerical-validation work, not cleanliness debt.
