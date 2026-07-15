# src/bse/bse_pseudopoles.py + pseudopoles_eval.py + pseudopoles_sweep.py — deep-read notes (699 + 179 + 339 LOC)

Audit date: 2026-07-15, lorrax_D checkout. Task named base `agent/slate-linalg-ffi`
(e18d0e5); working tree is at adc2197 (`agent/ppm-fit-conditioning`) — verified
identical for all three files: `git diff agent/slate-linalg-ffi HEAD -- src/bse/bse_pseudopoles.py
src/bse/pseudopoles_eval.py src/bse/pseudopoles_sweep.py` → empty, no uncommitted edits.

## Headline

**`bse_pseudopoles.py` cannot be imported at HEAD.** Three pieces of wiring were
lost when the old `src/isdf/bse_isdf/` package was consolidated into `src/bse/`
(commit 906dd31): two functions it imports from `bse_ring_comm`, one kwarg on
each matvec builder, and one kwarg on the sharded restart loader. The cleanup
commit a0da0a5 (2026-04-29) explicitly recorded "`bse_pseudopoles.py` — pseudopole
W approx, **broken on main (import error)**" and deleted it; fe5e3e8 (same day)
restored it as "**active dev**" per user request — but did not restore the missing
`bse_ring_comm`/`bse_io` pieces. Per the parsed-but-unread-≠-dead rule this is
**LOST WIRING for a feature the user wants**, not dead code. The historical
implementations still exist verbatim at `81ca040:src/isdf/bse_isdf/bse_ring_comm.py:815-865`
and are small (~50 lines, wrapping `build_realspace_random_transition_generator`
which *does* survive at `bse_ring_comm.py:719`).

`pseudopoles_eval.py` and `pseudopoles_sweep.py` are pure numpy/h5py
post-processors of the producer's H5 output; both import and run fine, but their
only producer is the broken module, so at HEAD they can only consume legacy files.

## Purpose

Windowed **pseudopole compression of the correlation screened interaction
W_c(ω)** in the ISDF r_μ basis, from the BSE/Casida Hamiltonian: FEAST contour
filtering with **density-biased random seeds**, Rayleigh–Ritz in a "brightness"
(density-response-weighted) subspace, plus a stochastic tail. Output: per energy
window w, a few poles/residues `{ω_p, d_p[μ], w_p}` such that

```
W_c[μ,ν](z) ≈ Σ_p  w_p · d_p[μ] · conj(d_p[ν]) / (z − ω_p)
```

(the eval formula, `pseudopoles_eval.py:16` and `:98`). Design doc:
`src/bse/context/tda_and_pseudopoles.md` (Steps A–G below mirror its section 4)
and `src/bse/context/bse_feast_instructions.md`.

Physics as written in code (`run_pseudopoles`, bse_pseudopoles.py:212–487):

```
Step A  seeds (283–308):
   η ~ real N(0,1), shape (1, n_mu_pad, nk)                       [density space]
   u  = V_q0 η                                                     (inside generator)
   f[b,c,v,k]    = Σ_{μ,s} ψ_c[k,c,s,μ]·conj(ψ_v[k,v,s,μ])·u[b,μ,k] / √nk   = d†u
   fbar          = same with ψ→conj(ψ)                              = dᵀu
   TDA:     φ = f                    non-TDA: φ = stack([f, −fbar], axis=0)
   (comment 284–287: "Do not assume fbar == conj(f) for complex Bloch spinors")

Step B  FEAST filter (_feast_filter 158–200 → bse_feast._get_feast_runner:217):
   TDA:     x_filt = Σ_j 2·Re[ w_j (z_j − H)⁻¹ x ]     (conjugate-symmetry, half contour)
   non-TDA: x_filt = Σ_j w_j (z_j − H)⁻¹ x             (nodes pre-doubled with conj,
                                                        bse_pseudopoles.py:279–281)
   z_j, w_j from feast_ellipse_quadrature(window, n_quad, gamma=0.2) in eV;
   divided by ry_to_ev inside the runner (bse_feast.py:246–247) → solves in Ry.

Step C  whitening (_orthonormalize 49–75):
   S[i,j] = Σ_d conj(V_flat[i,d])·V_flat[j,d];  eigh; keep λ ≥ s_cutoff·λ_max
   b_a = Σ_j s_evecs[j,a]·v_j / √λ_a           (Euclidean metric, even non-TDA)

   reduced operator (_build_reduced_h 98–125):
   H_w[i,j] = Σ_d conj(V_flat[i,d])·(S·v_j)_flat[d];  Hermitized ONLY if use_tda.

Step D  density snapshots (_compute_density_snapshots 128–155):
   TDA:     w(μ) = Σ_k [ V_q0 · Σ_{c,v,s} conj(ψ_c)ψ_v X ](μ,k) / √nk   = V(dX)
   non-TDA: w    = V(dX + d*Y)                       ← calls readout_full, MISSING
   C_w[μ,a] = w_a(μ) columns (host numpy, np.stack(cols, axis=1), line 155).

Step E  brightness eigendecomp (353–364):
   G_w = C_wᴴ C_w (Hermitized);  eigh → σ², descending;  Wb = top-p_keep evecs,
   Wd = rest.  σ = √max(σ²,0).

Step F  bright pseudopoles (366–437):
   H_b = Wbᴴ H_w Wb;  np.linalg.eig → (Ω_p, y_p), sorted by Re Ω;
   discard Ritz values outside [a,b]/ry_to_ev (372–386, leakage guard);
   g_p = Wb y_p;   d_p = C_w g_p
   non-TDA J-norm (391–431):  J_w[i,j] = ⟨X_i|X_j⟩ − ⟨Y_i|Y_j⟩  (_build_j_metric
     78–95, metric of S = [[A,B],[−B*,−A*]]);  N_p = y_pᴴ (Wbᴴ J_w Wb) y_p;
     d_p /= √Re(N_p) when Re(N_p) > 1e-6, else skipped with a print.
   anti-resonant doubling (433–437):  ω ← [Ω, −Ω];  d ← [d, conj(d)];
     w ← [+1…, −1…].   Justified by the Lehmann comment (394–409):
     chi(z) = Σ_s [ F_s F_sᴴ/(z−Ω_s) − F_s* F_sᵀ/(z+Ω_s) ],
     F_s = Σ_cv [X_s^{cv} ρ_cv + Y_s^{cv} ρ*_cv]; the −1 weight on the conj(d)
     pole at −Ω reproduces −F*Fᵀ/(z+Ω). ✓ (per-element check: w·d[μ]·conj(d[ν])
     /(z−ω) = (−1)·conj(F[μ])·F[ν]/(z+Ω).)

Step G  stochastic tail (439–471):
   z random unit vector in dark subspace;  ω = zᴴ (Wdᴴ H_w Wd) z (Rayleigh
   quotient, complex);  d = C_w Wd z;  window-filtered; rescaled by
   α = √(B_disc/B_tail), B_disc = Σ_{i>p_use} σ²_i, B_tail = Σ|d_tail|².
   Tail weights all +1 — NO anti-resonant partners and NO J-norm even in
   non-TDA mode (471: weights_tail = ones).
```

The BSE operator itself comes from `bse_ring_comm`: TDA `H = D + V − W` acting on
X(b,c,v,k); non-TDA `S = [[A,B],[−Bᴴ,−Aᴴ]]` acting on stacked [X,Y]
(bse_ring_comm.py:443–450, :496). RPA mode (`--rpa`) sets `include_W=False`
→ kernel D + V only (matvec returns before touching W_R, bse_ring_comm.py:446–447).

Category: **physics: experimental W_c(ω) compression stage (pseudopole
representation for downstream Σ/χ), currently dead-in-place — producer broken at
import since the src/ consolidation; eval/sweep are working host-side diagnostics
of its H5 output.**

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox runs/skills/scripts)

| symbol | callers (grep evidence) |
|---|---|
| `bse_pseudopoles.main` | **NONE FOUND** — only `if __name__ == "__main__"` (line 698–699). No `python -m bse.bse_pseudopoles` in sandbox skills/scripts (grep → empty), no hits in `runs/Si/04_si_4x4x4_bse/**` run scripts (`*.sh/*.md/*.log` grep → empty), no pyproject script (pyproject.toml:36 note: `lorrax-bse` entry was removed). Cannot run anyway: ImportError at line 31. |
| `run_pseudopoles` | `bse_pseudopoles.py:659` (own main) only |
| `write_pseudopoles_h5` | `bse_pseudopoles.py:678` (own main) only |
| `pseudopoles_eval.main` | `__main__` (177–178); usage self-documented as post-processor of the producer H5 |
| `pseudopoles_eval.load_pseudopoles`, `reconstruct_Wc_columns` | `tests/archive/projects/test_isdf/sweep_12v12c_plot.py:9` — `from bse_isdf.pseudopoles_eval import ...` — **stale package path** (`bse_isdf` no longer exists; archived project script, broken at HEAD) |
| `pseudopoles_sweep.main` | `__main__` (338–339); docstring self-documents `python -m bse.pseudopoles_sweep` (lines 8–14) |
| module docs mention | `docs/architecture/codebase.md:117` (file listing only) |

`src/bse/__init__.py` is a bare docstring — no re-exports. No getattr/string
dispatch found (`grep -rn "bse_pseudopoles\|pseudopoles_eval\|pseudopoles_sweep" src tests tools scripts`
→ only self-references + the archived script above).

## Function tables

### bse_pseudopoles.py (699 LOC)

| function | lines | role |
|---|---|---|
| `WindowSpec` (dataclass) | 41–46 | (name, a_eV, b_eV, note). **Duplicate** of `bse_feast.WindowSpec` (bse_feast.py:38) and `bse_feast_dense_debug.WindowSpec` (:20); main re-wraps bse_feast instances field-by-field (line 618). |
| `_orthonormalize(filtered, s_cutoff)` | 49–75 | Euclidean whitening of FEAST-filtered vectors; overlap S built on device, eigh on host; returns (basis, coeffs, s_evals). |
| `_build_j_metric(basis)` | 78–95 | J[i,j] = ⟨X_i\|X_j⟩ − ⟨Y_i\|Y_j⟩ for non-TDA; basis shape doc "(n, 2, 1, nc, nv, nk)". Host numpy result. |
| `_build_reduced_h(matvec, basis, data, use_tda)` | 98–125 | H_w = Vᴴ S V via 9-arg matvec calls (`matvec(v, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)` — matches `_matvec_impl` signature at bse_ring_comm.py:443). Hermitize iff TDA. |
| `_compute_density_snapshots(...)` | 128–155 | C_w columns; TDA via `snapshot_op` (exists, bse_ring_comm.py:775), non-TDA via `readout_full` (**missing at HEAD**). `w_mu[0]` selects trial 0 (nb_trial=1). |
| `_feast_filter(X_batch, z, w, matvec, data, diag_h, ...)` | 158–200 | Optional fp32 cast path: `_build_gmres_data_fp32(data)` (only 8 keys: ψ×4, eps×2, V_q0, W_q — bse_feast.py:281–291) then re-derives `W_R = jnp.fft.ifftn(W_q, axes=(2,3,4), norm="ortho")` **unconditionally** (177–178) even in RPA mode where the c128 dict holds `W_R = W_q` (placeholder). Harmless — matvec ignores W_R when include_W=False — but a wasted FFT and a silent placeholder mismatch. Returns filtered batch + per-solve GMRES iteration counts. |
| `_create_mesh_xy(px, py)` | 203–209 | devices[:px·py].reshape(px,py) → Mesh("x","y"). **4× duplicated**: bse_feast.py:686, bse_w_exact.py:28, feast_zolo_sweep.py:98, feast_ellipse_mixed_sweep.py:74. |
| `run_pseudopoles(data, mesh_xy, windows, *, m0, n_quad, p_keep, n_tail, gmres_max_iter, gmres_tol, seed, ry_to_ev, s_cutoff, quadrature, use_tda, gmres_fp32, include_W)` | 212–487 | Steps A–G per window (see Purpose). Builds matvec with `v_couples_k=not include_W` (231–249, **kwarg no longer exists**). **Mutates caller's dict**: `data["W_R"] = ifftn(data["W_q"])` or `= data["W_q"]` (251–254). `quadrature=="zolotarev"` → hard ValueError inside the loop (276–277); only "ellipse" is ever passed (main:672) → dead branch. Empty-basis window → `{"omega_bright": zeros(0), "d_bright": zeros((0,0))}` placeholder (335). Results dict per window also carries intermediates H_w, C_w, J_w, sigma, s_evals (473–485). |
| `write_pseudopoles_h5(output_file, windows, results, *, ...)` | 490–547 | Writer. Root attrs: use_tda, include_W, ry_to_ev, m0, n_quad, p_keep, n_tail, gmres_max_iter, gmres_tol, seed. Per-window group `w.name` with attrs a_eV/b_eV and datasets: omega_bright_ry, omega_bright_eV (=·ry_to_ev), d_bright **(n_poles, n_mu — stored transposed, `d_bright.T` at :534/475)**, weight_bright, omega_tail_ry/_eV, d_tail (n_tail, n_mu, stored untransposed — same layout), weight_tail, sigma, s_evals, H_w, C_w, J_w. Missing keys default to empty via `r.get`. |
| `main(argv)` | 550–695 | argparse → mesh → `_find_restart_file(args.input)` (bse_io.py:756: globs `tmp/isdf_tensors_*.h5` then `isdf_tensors_*.h5` beside the cohsex input) → `load_bse_data_from_restart_sharded(..., use_nohead=args.nohead, ...)` (**kwarg no longer exists**) → `estimate_spectral_bounds_sharded` (bse_feast.py:695; returns e_min_ry/e_max_ry_raw; E_max buffered by (1+args.buffer)) → windows from `build_default_windows_eV(e_max_eV)` (bse_feast.py:849: [0,2] eV + [2,E_max], collapsed if E_max<2) OR KPM-derived (`--windows-kpm` → `bse_kpm.run_kpm_dos(...)["windows_ry"]`, 619–641) OR user `--window1/--window2` (642–652, "auto" upper bound → E_max via `_parse_window_arg`, bse_feast.py:1061). Then run_pseudopoles + write. `timing.reset()/section()/report()` from common.timing. |

Module-level side effect: `jax.config.update("jax_enable_x64", True)` at import
(line 38).

### pseudopoles_eval.py (179 LOC) — pure numpy/h5py, imports fine

| function | lines | role |
|---|---|---|
| `Poles` (dataclass) | 42–46 | omega_ry (n_poles,), d (n_poles, n_mu), weight (n_poles,) ±1. |
| `_iter_window_groups(h5, windows)` | 49–55 | sorted group names, optional filter. |
| `load_pseudopoles(path, *, windows)` | 58–92 | Concatenates bright+tail poles across window groups; reads weight_bright/weight_tail with all-ones fallback. |
| `reconstruct_Wc_columns(poles, *, z_ry, cols)` | 95–117 | Per-element verified: `tmp[p,j] = conj(d_p[cols_j])·w_p/(z−ω_p)`; `(tmp.T @ d)[j,μ] = Σ_p w_p·d_p[μ]·conj(d_p[cols_j])/(z−ω_p) = Wc[μ, cols_j]` ✓ returns (n_cols, n_mu) — rows are the requested Wc columns. |
| `_parse_cols(col_str, n_mu)` | 120–125 | CSV of 0-based μ indices; silently drops out-of-range entries; empty → all μ. |
| `main(argv)` | 128–174 | Reads producer attrs (fallbacks: ry_to_ev=**13.605693122994** — CODATA, ≠ producer default 13.6056980659; use_tda=0; harmless since producer always writes attrs, but a fork), evaluates at z = (omega_ev + i·eta_ev)/ry_to_ev, writes `Wc_from_pseudopoles.h5` (datasets: columns int32, Wc (n_cols,n_mu)) — format matches `bse_w_exact.py --write-kind Wc` and the SOS/Dyson reference per module docstring (27–29). |

Docstring (1–30) restates the Lehmann convention and names the residue channel:
`d_s[μ] = (V·(dX + d*Y))[μ]` non-TDA, `(V·(dX))[μ]` TDA — matching
`bse_w_exact.py`'s response channel.

### pseudopoles_sweep.py (339 LOC) — pure numpy/h5py + matplotlib, imports fine

| function | lines | role |
|---|---|---|
| `_detect_format(h5_path)` | 26–35 | "intermediates" if any group has H_w+C_w (current producer ALWAYS writes them → always this path for fresh files), else "final" if d_bright+omega_bright_ry, else "unknown". |
| `_reconstruct_from_final(...)` | 38–95 | Truncates the first p_use **stored** poles per window (stored order = ascending Re Ω from producer sort, i.e. lowest-energy-first, NOT brightest-first), then for non-TDA re-appends anti-resonant partners with `−omega.conj()` (:76) and weights [+1,−1]. Ignores stored `weight_bright` and all tail datasets. Evaluation math per-element identical to eval ✓ (Wc = d_cat @ (coeffs[:,None]·d_cat[cols,:].conj().T), [μ,j] = Σ_p d[μ,p]·w_p·conj(d[cols_j,p])/(z−ω_p), then .T). |
| `_reconstruct_from_intermediates(...)` | 98–203 | Re-runs Steps E/F from H_w/C_w/J_w for each p_keep — ~55-line near-verbatim copy of bse_pseudopoles Steps E–F (brightness eigh, Ritz, window filter, J-norm with same 1e-6 floor, anti-resonant doubling with `−evals_b` at :177). No tail reconstruction (stored tail ignored — so sweep curves exclude the tail the producer would add). Empty-result path returns `np.zeros((n_mu, len(cols)))` (:192) — **transposed** relative to the normal return `Wc.T  # (n_cols, n_mu)` (:203). |
| `get_max_poles_per_window(h5_path)` | 206–216 | max over windows of H_w.shape[0] (basis size) or, fallback, d_bright.shape[0] (which is 2·n_res for non-TDA final files). |
| `get_sigma_per_window(h5_path)` | 219–240 | stored sigma, or recomputed from C_w. |
| `main(argv)` | 243–335 | p_keep sweep 1..max vs a reference Wc H5 (`--ref`, dataset "Wc", same (n_cols,n_mu) layout): rel error ‖Wc−Wc_ref‖/‖Wc_ref‖ and overlap α = Re Σ conj(Wc)·Wc_ref / Σ\|Wc_ref\|². `use_tda = bool(h5.attrs.get("use_tda", 1))` (:259) — fallback **1**, vs eval's fallback 0. 3-panel matplotlib PNG (σ spectra / error / α). Module constant RY_TO_EV = 13.6056980659 (:23). |

## Flags / CLI args consumed

`bse_pseudopoles.py main` (553–586): `-i/--input` (required, cohsex input; used
to locate `isdf_tensors_*.h5` and for head/n_occ overrides inside the loader),
`--n-val` (4), `--n-cond` (4), `--px` (1), `--py` (1), `--m0` (6, seeds/window),
`--n-quad` (8, per half contour), `--p-keep` (4, bright modes/window), `--n-tail`
(2), `--gmres-max-iter` (10), `--gmres-tol` (1e-2), `--gmres-fp32` (store_true),
`--seed` (0), `--ry-to-ev` (13.6056980659), `--rpa` (store_true → include_W=False,
D+V kernel), `--tda` (store_true; **default is full non-TDA**), `--nohead`
(store_true → dead: feeds the missing `use_nohead` kwarg), `--out`
(bse_pseudopoles.h5), `--s-cutoff` (1e-6), `--windows-kpm` (store_true),
`--windows-kpm-count` (4), `--kpm-n-moments` (200), `--kpm-n-random` (4),
`--kpm-seed` (0), `--kpm-n-energy-pts` (2000), `--kpm-n-lanczos` (100),
`--buffer` (0.05, E_max headroom), `--window1 A B` / `--window2 A B` (B may be
"auto").

`pseudopoles_eval.py main` (129–141): `--poles` (required), `--out`
(Wc_from_pseudopoles.h5), `--omega-ev` (0.0), `--eta-ev` (0.0), `--cols` (CSV,
default all μ), `--windows` (CSV group filter, default all).

`pseudopoles_sweep.py main` (244–252): `--poles` (required), `--ref` (required),
`--omega-ev` (0.0), `--eta-ev` (0.0), `--cols` ("0,1,…,8"), `--plot-file`
(sweep_pkeep.png), `--ry-to-ev` (13.6056980659).

No LorraxConfig/cohsex.in keys are read directly by these files; the loader
parses head overrides / WFN path out of the `-i` input file (bse_io.py:487+).

## Sharding / PartitionSpec assumptions

2-D device mesh `("x","y")` = (conduction, valence) band axes, from
`make_bse_shardings` (bse_ring_comm.py:46–63):

- BSE trial vectors: `sh.X = P(None,'x','y',None)` on (b, nc, nv, nk); non-TDA
  `sh.X_full = P(None,None,'x','y',None)` on (2, b, nc, nv, nk) — seeds
  constrained at bse_pseudopoles.py:299/302–306, basis vectors re-constrained per
  snapshot at 149/152.
- Density-space seeds: `sh.S = P(None,'y',None)` on (1, n_mu_pad, nk) (line 296).
- ψ tensors dual-resident: `psi_*_X = P(None,None,None,'x')`, `psi_*_Y =
  P(None,None,None,'y')` on (nk, nb_pad, nspinor, n_mu_pad) (loader,
  bse_io.py:449–470).
- `V_q0 = P('x','y')` (μ,ν tile); `W_q = P('x','y',None,None,None)` on
  (μ,ν,nkx,nky,nkz); `eps = P(None,None)` replicated.
- Band padding: n_cond_pad divisible by px, n_val_pad by py; μ extent padded to
  grid_x·grid_y multiple (`padded_mu_extent`, bse_io.py:444–447);
  `build_realspace_random_transition_generator` raises if not (bse_ring_comm.py:736–737).
- `run_pseudopoles`'s `data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2,3,4))`
  (line 252) runs a plain jnp FFT over the *replicated* k axes of the ('x','y')
  μν-sharded W_q — the matvec builders instead use the custom-partitioned FFT
  helpers to avoid all-gathers (bse_ring_comm.py:395–404); one-shot here, so cost
  is amortized, but it is the naive form.

## Host vs device residency

Device (sharded): ψ×4, eps×2, V_q0, W_q/W_R, seeds, FEAST-filtered vectors,
whitened basis list, diag preconditioner. Host numpy (via `jax.device_get`):
overlap S (line 58), reduced H_w (:122), J_w (:95), density columns C_w (:154,
gathered one column at a time — `w_mu[0]` per basis vector), and everything in
Steps E–G (numpy eigh/eig on p ≲ m0-sized matrices). The two post-processing
modules are host-only (no jax import). No io_callback usage; the full restart
read is `load_bse_data_from_restart_sharded`'s per-shard reads.

## TDA vs full-BSE handling

Explicit dual-path throughout, keyed on `use_tda` (CLI `--tda`; **default
non-TDA**): matvec builder (TDA vs `_full` on stacked [X,Y]); seeds (f vs
[f,−fbar]); FEAST contour (half contour + 2·Re conjugate symmetry vs
conj-doubled nodes, lines 279–281 vs runner flag); H_w Hermitization (TDA only);
density channel (V(dX) vs V(dX+d*Y)); J-metric normalization and anti-resonant
pole doubling (non-TDA only, lines 391–437). Known gap: **tail poles never get
anti-resonant partners or J-norms** even in non-TDA mode (lines 439–471) —
approximate by design but undocumented.

## Spin / nspinor handling

ψ arrays carry an explicit nspinor axis (nk, nb, nspinor, n_mu). All pair
densities contract it: generator `M_X[k,c,v,m] = Σ_s conj(ψ_c[k,c,s,m])·ψ_v[k,v,s,m]`
(bse_ring_comm.py:761), snapshot same pattern (:882–884). I.e. **charge channel
only** — spinor-summed products, no spin-flip/transverse channels, no separate
nspin index. Works transparently for nspinor=1 or 2; no bispinor-specific
branches. Singlet/triplet distinction is not represented.

## Coupling to gw/ and isdf/

- Input is the **gw_jax restart bundle** `isdf_tensors_*.h5` (V_qmunu, W0_qmunu
  gated on attr W0_ready, G0_mu_nu rank-1 head projector, vhead/whead scalars,
  psi_full_y, enk_full) — located next to the cohsex input (bse_io.py:756–764).
- The loader injects the q→0 head via `gw.head_correction.apply_q0_head_rank1_sharded`
  (bse_io.py:506–509) with dual-sharded g0_X/g0_Y copies.
- No direct import of isdf/ modules in these three files; the ISDF μ basis is
  consumed implicitly through the restart tensors. bse-internal imports only:
  bse_feast, bse_ring_comm, bse_io, bse_kpm (lazy, line 620), plus common.timing.

## Suspects

### Broken (lost wiring — the producer is un-runnable at HEAD)

1. **ImportError at module import** — bse_pseudopoles.py:31–32 imports
   `build_density_drive_operators` and `build_density_readout_operator_full`
   from `.bse_ring_comm`; neither is defined anywhere at HEAD
   (`grep -rn "def build_density_drive_operators\|def build_density_readout_operator_full" . --include='*.py'`
   → no hits). They existed at `81ca040:src/isdf/bse_isdf/bse_ring_comm.py:815`
   and `:849` (f = d†(Vη), fbar = dᵀ(Vη) via conj-ψ trick; readout_full =
   snapshot(X) + snapshot(Y, conj ψ)). Dropped in the 906dd31 consolidation;
   their building block `build_realspace_random_transition_generator` survives
   (bse_ring_comm.py:719). Even `--help` fails.
2. **`v_couples_k` kwarg gone** — bse_pseudopoles.py:239/248 pass
   `v_couples_k=bool(not include_W)` but HEAD `build_bse_ring_matvec`
   (bse_ring_comm.py:340–348) and `build_bse_ring_matvec_full` (:487–495) accept
   only (mesh, nkx, nky, nkz, timed, low_mem, include_W) → TypeError once the
   import is fixed. Historical signature had `*, v_couples_k: bool = False`
   forwarded as `couple_k` into apply_V_ring (81ca040 lines 364/413) — the
   RPA-mode V-couples-k physics knob is now unreachable. Same breakage in
   `bse_kpm.run_kpm_dos` (bse_kpm.py:120–137), so the `--windows-kpm` path is
   doubly broken.
3. **`use_nohead` kwarg gone** — bse_pseudopoles.py:599 passes
   `use_nohead=args.nohead` but HEAD `load_bse_data_from_restart_sharded`
   (bse_io.py:358–368) has no such parameter → TypeError. Historically supported
   (81ca040 bse_io.py:307–322, selecting V_qmunu_nohead/W0_qmunu_nohead
   datasets). Also passed by bse_kpm.py:370. `--nohead` is lost wiring for the
   headless-V/W A/B-testing knob.

Failure order for `python -m bse.bse_pseudopoles`: (1) kills it at import; fixing
(1) exposes (3) in main; fixing (3) exposes (2) in run_pseudopoles.

### Bugs (post-processors)

4. **Anti-resonant pole placement disagrees between producer and sweep formats**:
   producer uses `−evals_b` (bse_pseudopoles.py:435) and sweep-intermediates
   matches (`−evals_b`, pseudopoles_sweep.py:177), but sweep-final uses
   `−omega.conj()` (pseudopoles_sweep.py:76). For a complex Ritz value Ω = a+ib
   (np.linalg.eig of the non-Hermitian H_b generally yields complex Ω) the two
   place the anti-pole at −a−ib vs −a+ib — reconstructions differ between the
   two sweep branches and one of them from the producer. (Casida-structure
   symmetry pairs (Ω, −Ω*), so `−conj` is arguably the correct one — the
   *producer* and the intermediates path carry the questionable convention.)
5. **Sweep final-format non-TDA double count**: stored `d_bright` for non-TDA
   files already contains the anti-resonant rows (producer concatenates before
   writing, bse_pseudopoles.py:433–437,475). `_reconstruct_from_final` slices
   the first p_use rows and re-doubles (pseudopoles_sweep.py:65–78); for
   p_keep > n_res the slice includes stored anti-resonant rows which then get
   weight +1 and a fabricated partner. The sweep upper bound for such files is
   d_bright.shape[0] = 2·n_res (get_max_poles_per_window:215), so the top half
   of the sweep is wrong. Mitigated in practice: current producer always writes
   H_w/C_w so `_detect_format` routes fresh files to the intermediates path;
   only legacy final-only files hit this.
6. **Transposed empty-result shape**: `_reconstruct_from_intermediates` returns
   `np.zeros((n_mu, len(cols)))` when no window yields in-window poles
   (pseudopoles_sweep.py:192) while the normal path returns `(n_cols, n_mu)`
   (:203) — `np.linalg.norm(Wc − Wc_ref)` in main then raises a broadcast
   ValueError whenever n_mu ≠ n_cols (e.g. small p with strict windows).

### Redundancy / cruft

7. `_reconstruct_from_intermediates` re-implements Steps E–F of run_pseudopoles
   nearly verbatim (~55 lines incl. the 1e-6 j_floor) — two sources of truth for
   the brightness→Ritz→J-norm pipeline; both drop the tail term and stored
   weights when sweeping. Violates the no-redundancy rule.
8. `feast_zolotarev_quadrature` imported (bse_pseudopoles.py:18) but never
   called; `quadrature` parameter is hardwired to "ellipse" (main:672) with a
   dead per-window `raise ValueError` branch for "zolotarev" (276–277).
9. Duplicate `WindowSpec` dataclass (bse_pseudopoles.py:41 vs bse_feast.py:38 vs
   bse_feast_dense_debug.py:20) plus field-by-field re-wrap at :618; 5×
   duplicated `_create_mesh_xy` across bse modules (:203, bse_feast.py:686,
   bse_w_exact.py:28, feast_zolo_sweep.py:98, feast_ellipse_mixed_sweep.py:74).
10. Constant forks: eval's fallback ry_to_ev = 13.605693122994
    (pseudopoles_eval.py:144) vs 13.6056980659 everywhere else; `use_tda` attr
    fallback 0 in eval (:145) vs 1 in sweep (:259). Only reachable on
    hand-crafted files (producer always writes both attrs).
11. Stale archive import `bse_isdf.pseudopoles_eval`
    (tests/archive/projects/test_isdf/sweep_12v12c_plot.py:9) — pre-consolidation
    package name; broken, archived.

### Weird

12. `run_pseudopoles` mutates the caller's `data` dict in place
    (`data["W_R"] = …`, lines 251–254), and `_feast_filter` recomputes
    `W_R = ifftn(W_q)` unconditionally in the fp32 branch (177–178) even in RPA
    mode where the c128 convention is `W_R = W_q` (placeholder) — benign only
    because `_matvec_impl` returns before touching W_R when include_W=False
    (bse_ring_comm.py:446–447).
13. `jax_enable_x64` flipped at import time (line 38) — global config side
    effect on any importer.
14. Seeds η are **real** normal draws (float32/float64, lines 291–295) in a
    complex pipeline — intentional (density-space source), but easy to misread.
15. Tail pseudopoles: complex Rayleigh-quotient ω kept as-is, no J-norm, no
    anti-resonant partner, all-+1 weights, energy-window-filtered *after*
    generation (439–471) — several silent approximations stacked in one block.
