# BGW invalid-PPM-pole handling — research reference (saved 2026-07-08)

From the BGW-source research pass (2026-07-04). All BGW paths under
`sources/BerkeleyGW/`; docs under `docs/docs_bgw/`.

## The invalid condition
A pole is invalid when the fitted squared mode frequency is negative: `dble(wtilde2) < 0`.
- HL-GPP (`freq_dep=1`): `wtilde2 = Omega2 / I_epsggp`, `Sigma/mtxel_cor.f90:772`.
- GN-GPP (`freq_dep=3`): `wtilde2 = |dFreqBrd(2)|²·I_epsRggp(2)/(I_epsRggp(1)−I_epsRggp(2))`, `:826`.
(Separate `cycle` skips for `|Omega2|<TOL_Small` / `|I_epsggp|<TOL_Small` = "no screening", not invalid-pole.)

## The keyword + branch
`sigma.inp` keyword: **`invalid_gpp_mode <int>`**, default −1 (`Sigma/inread_sig.f90:82,472-473`;
`Common/typedefs.f90:262`; doc `docs/docs_bgw/sigma.inp:260-267`). Branch at
`mtxel_cor.f90:778-794` (HL) and `:828-844` (GN):

| mode | wtilde set to | Σc effect | LORRAX name |
|---|---|---|---|
| 0 | 0 | Omega2=0 → SX and CH terms = 0; **pole dropped** | `zero`/`skip` |
| 1 | `i·√\|wtilde2\|` | damped/imaginary pole (BGW-1.x default) | `imaginary` (unsupported) |
| 2 | `2·ryd` = 2 Ry | finite real pole at 2 Ry, residue rebuilt | `2ry` |
| 3 / default(−1) | `1/TOL_ZERO` = 1e12 eV ≈ ∞ | `delw→−1`, CH → −½·I_eps **static COHSEX**; SX pole → 0 | `static_limit` |

Key mechanism: BGW overrides only `wtilde`; the residue is REBUILT downstream as
`Omega2 = wtilde²·I_eps` (`mtxel_cor.f90:1813-1846`), so the mode consistently rescales the pole.

## LORRAX mapping (state as of commit 3cad3dd+)
- Fit (`minimax_screening.fit_gn_ppm_from_wc_pair:408`): `good = Ω²>0`; invalid → `Ω = fallback_omega`
  (default 2.0 Ry) and `B = −½·Wc0·Ω` — i.e. the fit already bakes BGW mode 2 for invalid entries.
- Kernel (`ppm_sigma._prepare_sigma_state`): `ppm_invalid_mode` wired 2026-07-04: `zero`/`skip` drops
  (B_mask &= valid, = BGW 0, the default); `2ry` keeps the fit's fallback pole (B_mask = B_mask_raw,
  = BGW 2). `static_limit`/`infinity` raise NotImplementedError; `imaginary` unsupported.
- **static_limit implementation note**: Ω→∞ is NON-INTEGRABLE on the τ-grid (B·e^{−iΩτ} with B∝Ω).
  Must be an analytic term instead: for invalid poles add the static Coulomb-hole **−½·Wc0** to Σc.
  Since the fit sets `B = −½·Wc0·Ω` exactly, **−½·Wc0 = B_q/Ω_q elementwise** — recoverable from the
  existing PPMBuildResult (Ω for invalid entries is the finite fallback value, so B/Ω is well-defined
  and equals −½·Wc0 exactly). Retaining Wc0 explicitly on PPMBuildResult is the clearer data-seam.
- BGW's default is mode 3 → LORRAX won't match a default-BGW run wherever invalid poles occur until
  static_limit is implemented and made default. Every system (incl. MoS2) produces some invalid poles
  (~meV effect); user-confirmed 2 Ry was useful on a CO example and MoS2.
