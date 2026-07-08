# Σ_PPM — the flow of ideas, and the cleanup

_2026-07-04. MAP.md-style map of the plasmon-pole self-energy (the most important area:
`gw/ppm_sigma.py` 1702 L, `gw/head_correction.py` 832, `gw/ppm_pipeline.py` 409,
`gw/gw_driver_helpers.py` 285). Synthesized from a 5-agent audit + the BGW invalid-pole
research. Companion to MAP.md. Part 1 = the flow. Part 2 = the cleanup ledger._

---

## Part 1 — the flow (one Σc(ω) evaluation, W → QP)

The whole engine turns a screened interaction `W(q)` into a frequency-dependent self-energy
`Σc(ω)` via a plasmon-pole model, then reads it at the QP energy. Eight stages:

```
 W(q), V(q)                                                      [w_isdf → poles]
   │  S0  fit: two-point (static W^c(0) + probe W^c(ω_p)) → PPM poles
   ▼
 (Ω_q, B_q, valid_mask)   Ω²<0 ⇒ "invalid pole"                  minimax_screening.fit_gn_ppm_from_wc_pair
   │  S1  physics state: eF, E_cond/H_val, B_mask = (Ω>0 & valid)
   ▼
 _SigmaPhysicsState        ← the invalid-pole gate lives here    ppm_sigma._prepare_sigma_state:313
   │  S2  branch: Σc(ω)=Σc over 4 (±ω)×(cond/val) branches  (Σc(−ω)=−Σc(ω)*)
   ▼
 _SigmaBranch ×4
   │  S3  windows: per branch, host-side minimax τ-windows       _build_three_sigma_windows / _single
   ▼        core(crossing,imag) + a_stripe(Laplace) + b_slab(Laplace)
 _SigmaWindow (τ nodes, masks, refs, projector)
   │  S4  τ-integrate (Python loop — a lax.scan cost MoS2 3×3 ~80%)
   ▼        W(τ)=Σ_q B_q e^{-i(Ω-E_ref)τ}·mask_B  →  G(τ) FFT  →  ψ*σψ reduce-scatter
 σ_k(τ) sharded (m_X,n_Y)
   │  S5  project onto ω: contrib[ω]=pref·e^{i·sign·ω·t}·P(σ)    _HostOmegaAccumulator (async D2H)
   ▼
 Σc(ω) on the ω-grid       (SigmaOmegaResult)                    compute_sigma_c_ppm_omega_grid:1488
   │  S6  + q→0 HEAD (added, diagonal-in-band)                   head_correction.compute_ppm_head_sigma_kij
   │  S7  eval at E_DFT (interp) → on-shell fixed-point QP solve  ppm_pipeline → gw_jax:649
   ▼
 E^QP
```

**The head is a parallel scalar track** (`head_correction.py`, 3 consumers): the `G=G'=0`
divergence is zeroed out of the ISDF body tensors and re-added as (a) the **PPM dynamic head**
`Σ^c_n(ω)=R_h/(V·N_k)[f_n/(ω−ε_n+Ω_h) + (1−f_n)/(ω−ε_n−Ω_h)]` (the Σc track above, S6),
(b) the **static COHSEX head** (Σ^X/SX/COH diagonal shifts), (c) the **rank-1 (μ,ν) head**
`Δ=(head/V)·conj(ζ₀)⊗ζ₀` (BSE/W-builder). One front-end `HeadResolver` memoizes the head
samples (`override` → `epshead` → `s_tensor`).

**The invalid-pole gate (S1) is the crux for BGW parity.** A pole with `Ω²<0` is "invalid";
`_prepare_sigma_state` currently *drops* it (`B_mask = B_mask_raw & valid`) — BGW mode 0. But
the fit already baked a 2 Ry fallback (BGW mode 2) and the config default says `static_limit`
(BGW mode 3). Three modes at three layers, innermost wins silently. See §2C.

---

## Part 2 — the cleanup ledger

Three classes (the audit's discipline): **(a) live**, **(b) truly dead → delete**,
**(c) disconnected-but-wanted → re-wire, never delete**.

### 2A. DEAD → delete (class b) — all 0 reads repo-wide, no orphaned machinery

**Config knobs** (delete key + `_DEFAULTS` + dataclass field + parse + mirror):
`ppm_sigma_scale`, `ppm_sigma_flip_neg` (user-confirmed abandoned), `ppm_sigma_debug_static_norm`,
`ppm_static_cohsex_check`, `sigma_debug_quadrature`(+`_samples`), `write_w_copies_debug`(+`_file`),
`debug_hartree`, `debug_omega`.

**Dead functions / classes:**
- `ppm_sigma.fit_gn_ppm` (:723) — 0 callers (distinct from the live `fit_gn_ppm_from_wc_pair`).
- `head_correction.fit_head_gn` (:348) + `fit_head_gn_from_samples` (:489) — 0 callers.
- `ppm_sigma._ReduceScatterGpuAccumulator` (:934, ~72 L) — never instantiated; **first migrate its
  reduce-scatter layout docstring to `_make_project_ri_reduce_scatter` (:425)**.
- `mask_B_mode="explicit"` branch (:276-279) + `_SigmaWindow.mask_B` field — dead, but it's the
  dataclass DEFAULT (:164) → must flip the default to `"all"` in the same edit or a bare
  `_SigmaWindow()` hits the `raise`.

**Dead imports (SYMBOL IS LIVE in w_isdf — delete only the import line, NOT the function):**
- `ppm_sigma.py:71` `build_static_minimax_window_pair` (live at w_isdf.py:457).
- `ppm_sigma.py:74` `solve_laplace_minimax_imag_interval` (live at w_isdf.py:468).

**Debug cruft:** always-on `[DBG-PPM]` prints (`ppm_sigma.py:1202,1204,1206,1210,1442,1448,1672,1674`),
statically-dead `if False else` ternary (:1587) → gate behind `print_fn` or delete.

### 2B. The structural win — collapse the dead mirror `PPMSigmaRuntimeOptions`

`gw_driver_helpers.PPMSigmaRuntimeOptions` (23 fields) is a **parallel copy of `config.ppm`/
`config.debug` that has rotted**: only **8** fields are read off the object (`omega_grid_ev/ry`,
`sigma_regularization_ry`, `sigma_edge_factor`, `sigma_omega_batch_size`,
`sigma_omega_accumulation`, `sigma_kij_h5_path`, `fermi_reference`). The rest are dead plumbing,
or *redundant mirrors of a value the code already reads straight off `config`* (`sigma_freq_debug_*`
← `config.debug` at gw_jax:817; `sigma_at_dft_extrapolate` ← `config.ppm` at gw_jax:673; `omega_p_ry`,
`ppm_fallback` rebuilt from `config.ppm` directly). This is the "no parallel paths / single source
of truth" smell in its purest form. **Target: pass `config.ppm` + `config.debug` (+ the derived
ω-grid) directly to `compute_sigma_c_ppm_omega_grid`; delete `PPMSigmaRuntimeOptions` entirely.**
The 2 class-(c) carriers below must be *wired*, not dropped, when the mirror goes.

### 2C. DISCONNECTED-WANTED → re-wire (class c) — the two the user needs

**`ppm_invalid_mode`** (BGW `invalid_gpp_mode`) — **DONE 2026-07-08** (wired 2026-07-04;
`static_limit` implemented + made DEFAULT on agent/memplanner-cleanup, commits fdf89c2+):
- `"zero"`/`"skip"` → `B_mask = B_mask_raw & valid` (drop; BGW mode 0).
- `"2ry"` → keep the fit's 2 Ry fallback pole (BGW mode 2).
- `"static_limit"`/`"infinity"` (DEFAULT, = BGW default mode 3) → drop the dynamical pole and add
  the analytic ω-independent **static-COHSEX term of the mode** via the two `cohsex_sigma` kernels:
  `Σ_static = sigma_sx(G_occ, Wc0·inv_mask) + sigma_coh(Wc0·inv_mask)` (`Wc0_q` retained on
  `PPMBuildResult`; = `−2·B/Ω`). NOTE: BGW mode 3 keeps BOTH the static SEX (occ) and CH terms
  (`mtxel_cor.f90` ω̃→∞: `ssx→−I_ε`, `sch→−½I_ε`) — the research note's "SX pole → 0" was wrong.
  Validated three-way vs BGW m0/m2/m3 on Si 4×4×4
  (`reports/bgw_invalid_mode_refs_2026-07-08/lorrax_mode_table3.dat`).
- `"imaginary"` → explicit "unsupported" error (needs a complex-Ω path; user didn't ask).

**`sigma_at_dft_energies`.** The at-DFT interp (`_eval_sigma_c_at_dft_energies`, ppm_pipeline:166)
runs **unconditionally as a diagnostic**; the flag that would make ⟨nk|Σ(E_nk^DFT)|nk⟩ the
**authoritative** QP energy (vs the on-shell fixed-point solve, gw_jax:649) is orphaned. Re-wire at
the QP-solve dispatch (gw_jax:649/673): `True` → at-DFT authoritative (true one-shot G0W0);
`False` → keep the fixed-point solve.

### 2D. Two confirmed live bugs (physics — need gate + BGW check)

- **Bug A — head sign flip** (`head_correction.py:317-322`). Negative-Ω² branch: `omega_h`
  is `abs(Ω²)^0.5` (positive) while `B_h = −w1·Ω²` keeps the signed (negative) Ω², so
  `R_h = B_h/(2Ω_h)` flips sign → the whole q→0 head Σc (hundreds of meV) comes out with the wrong
  sign. **Fix:** `B_h = −w1·abs(omega_h_sq)` in the negative branch. Repro: default GN path when
  `Ω²≤0` (noisy `s_tensor`/`epshead` head or small probe ω). Only the GN two-point fitter hits it.
- **Bug B — streamed head drop** (`ppm_pipeline.py:126-127`). In `KIJ_STREAM` mode
  `sigma_c_kij=None` → `_inject_analytic_head` early-returns → the q→0 head is never added *and*
  never written to the stream h5 → head-less Σc on large-ω-grid streamed runs. **Fix:** inject the
  analytic head into the stream accumulator (or forbid streaming without a head path).

### Execution order (value × safety)
1. **2A dead-delete** — delete-only, gated by the 5 e2e gates. Safe now.
2. **2B mirror collapse** — pass `config` directly, delete `PPMSigmaRuntimeOptions`. Gated (a wrong
   field wiring fails the gates). Carries 2C's fields — wire them here.
3. **2C re-wire** `ppm_invalid_mode` (+`sigma_at_dft_energies`) — physics; new gate wanted (a fixture
   with invalid poles) + a BGW parity check. `static_limit` needs the `Wc0` retention.
4. **2D bug fixes** — physics; each needs a targeted repro + gate (Bug A shifts the head sign where
   it triggers; Bug B changes streamed Σc). Do with a BGW head-parity check.
