**MoS2 3x3 No-Sym COHSEX: bare cutoff + exact static head terms**

Fresh LORRAX COHSEX rerun for `runs/MoS2/02_mos2_3x3_nosym/` with `640` centroids, `bare_coulomb_cutoff = 30.0`, and the cleaned static head implementation in `src/gw/head_correction.py`. The static q=0 head is now applied exactly in band space:
- `Sigma^X_h = -v_h / (Omega_cell * N_k)` on occupied bands only
- `Sigma^SX_h = -W_h / (Omega_cell * N_k)` on occupied bands only
- `Sigma^COH_h = +0.5 * (W_h - v_h) / (Omega_cell * N_k)` on all bands

The comparison target is the existing BGW no-sym COHSEX reference in `00_bgw_cohsex/sigma_hp.log`.

Run:
- LORRAX: `runs/MoS2/02_mos2_3x3_nosym/04_lorrax_cohsex_barecut_headnk/`
- BGW: `runs/MoS2/02_mos2_3x3_nosym/00_bgw_cohsex/sigma_hp.log`

Timing from `gw.out`:
- total recorded: `49.080 s`
- `prepare_isdf_and_wavefunctions`: `19.056 s`
- `gw_jax.chi0_W`: `29.991 s`
- `gw_jax.pipeline`: `1.372 s`

Comparison over `48` matched `(k, band)` entries:

| Quantity | New MAE | Old no-sym MAE | Change |
|---|---:|---:|---:|
| `sigTOT` vs BGW `Sig'` | `58.560 meV` | `71.343 meV` | `-12.783 meV` |
| `sigSX` vs BGW `SX = X + (SX-X)` | `80.241 meV` | `84.043 meV` | `-3.802 meV` |
| `sigCOH` vs BGW `CH'` | `65.462 meV` | `72.695 meV` | `-7.233 meV` |
| `(sigSX - sigX) + sigCOH` vs BGW `SX-X + CH'` | `51.285 meV` | `61.041 meV` | `-9.756 meV` |
| `sigX` vs BGW `X` | `34.687 meV` | `34.687 meV` | `+0.000 meV` |

Additional residual summaries:
- `sigTOT - Sig'` mean signed residual: `+13.628 meV`
- `sigSX - SX` mean signed residual: `+79.090 meV`
- `sigCOH - CH'` mean signed residual: `-65.462 meV`
- `(sigSX - sigX + sigCOH) - (SX-X + CH')` mean signed residual: `+28.581 meV`
- `sigTOT` max absolute residual: `74.050 meV`

Takeaway:
- the exact static COHSEX head terms improve every reported COHSEX comparison metric over the older no-sym baseline
- the largest gain is in total static self-energy: `71.3 -> 58.6 meV` MAE vs BGW `Sig'`
- the screened correlation combination `(SX-X) + CH'` also improves materially: `61.0 -> 51.3 meV`

**Follow-up refactor**

After validating the static formulas, the head-handling code was cleaned up in the source tree:
- all active head-source resolution now lives in `src/gw/head_correction.py`
- explicit overrides `vhead`, `whead_0freq`, and `whead_imfreq` are now resolved through the same path used by static COHSEX and GN-PPM head diagnostics
- the old `G0_mu_nu`-based rank-1 μν injection path was removed from `gw_jax.py`
- `G0_mu_nu` is still written to restart data for diagnostics, but is no longer part of the active self-energy path
- the dead `head_correction_fn` hook was removed from the minimax PPM builder

Validation for the refactor itself was source-side:
- `uv run python -m pytest -q` → `13 passed, 1 warning`
- no new sandbox production run was needed because the refactor preserved the already-validated static-head physics and primarily cleaned up source routing / override handling
