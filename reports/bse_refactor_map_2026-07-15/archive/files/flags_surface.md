# BSE input-parameter surface (flags/config/env) — deep-read notes

Audit date: 2026-07-15, lorrax_D checkout. Working tree is on
`agent/ppm-fit-conditioning` (218aeb8), but
`git diff --stat e18d0e5 HEAD -- src/bse src/solvers` is **empty** — the
BSE/solvers surface below is byte-identical to the stated audit base
`agent/slate-linalg-ffi` (e18d0e5).

Scope: every user-facing knob across `src/bse/` and `src/solvers/` —
argparse arguments of every CLI, every cohsex.in key the BSE code reads,
every environment variable. This is the BSE analogue of the GW program's
`archive/FLAGS.md`.

Category: **config / input surface** (cross-cutting; no physics kernels of
its own — kernel index-math audits live in the per-module notes files).

## Purpose

The BSE stack is driven entirely by CLI flags + a minimal ad-hoc re-parse
of `cohsex.in`; there is **no** BSE section in the LorraxConfig schema and
`gw.gw_config.read_lorrax_input` is never imported by `src/bse`
(`grep -n "gw_config\|read_lorrax_input" src/bse/*.py` → empty). What the
flags steer is the sharded iterative solution of

```
H·X = D·X + V·X − W·X            (bse_jax.py:86, `D_term + V_term - W_term`)
```

with, as written in code:

```
ΔE[c,v,k]   = eps_c[k,c] − eps_v[k,v]                    (bse_preconditioner.py:41)
(D·X)[b,c,v,k] = ΔE[c,v,k] · X[b,c,v,k]                  (bse_jax.py:89-91)

M[k,c,v,μ]  = Σ_s conj(ψ_c[k,c,s,μ]) · ψ_v[k,v,s,μ]      (bse_jax.py:94-95, "kcsm,kvsm->kcvm")

V-term (bse_jax.py:98-121, exchange kernel, q=0 block only):
  S[b,N,k] = Σ_{c,v} M_Y[k,c,v,N] · X[b,c,v,k]           ("kcvN,bcvk->bNk", psum over "x")
  S       /= √Nk
  U[b,M,k] = Σ_N V_q0[M,N] · S[b,N,k]                    ("MN,bNk->bMk", psum over "y")
  (V·X)[b,c,v,k] = Σ_M conj(M_X[k,c,v,M]) · U[b,M,k] / √Nk   (psum_scatter "x" on c)

W-term (bse_jax.py:124-160, direct kernel; k↔k′ coupling enters via the
q = k−k′ convolution done as 3-D FFTs over the k axes):
  R[b,c,k,s,N]     = Σ_v conj(ψ_v[k,v,s,N]) · X[b,c,v,k]     ("kvsN,bcvk->bcksN")
  T[b,M,N,t,s,k]   = Σ_c ψ_c[k,c,t,M] · R[b,c,k,s,N]         ("kctM,bcksN->bMNtsk", psum "x")
  T_R = ifftn_k(T),  W_R = ifftn_q(W_q),  U_R = W_R[μ,ν,R]·T_R,  U_q = fftn(U_R)
  A[b,c,N,s,k]     = Σ_{M,t} conj(ψ_c[k,c,t,M]) · U[b,M,N,t,s,k]  ("kctM,bMNtsk->bcNsk")
  (W·X)[b,c,v,k]   = Σ_{s,N} ψ_v[k,v,s,N] · A[b,c,N,s,k] / √Nk   (psum "y", psum_scatter "x")
```

Energies are **Ry internally**; every CLI converts for reporting via an
explicit `--*ry-to-ev` / `--units-ev-per-ry` flag (BGW writes eV — see
`STATUS.md` "Index ordering" items 1-2 before calling any of this a bug:
valence-axis flip and Ry→eV on file write are deliberate BGW-compat
conversions in `write_eigenvectors_stream`, bse_io.py:23).

## Entry points (CLIs) — grep evidence

| module (`python -m …`) | evidence of use at HEAD |
|---|---|
| `bse.bse_jax` | runs logs: `runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/lorrax_bse_bgweqp_headfix2.out:2` ("python3 -u -m bse.bse_ja…"), 14 more hits in runs/; recommended usage `src/bse/STATUS.md:100-105` |
| `bse.davidson_absorption` | `runs/Si/C_bse_davidson_profile_2026-04-29/REPORT.md` + make_comparison.py ("python3 -u -m bse.davidson_absorption"); own docstring davidson_absorption.py:14 |
| `bse.absorption_haydock` | `src/bse/STATUS.md:108-111` (canonical vs-BGW route); BGW_COMPARE.md cookbook |
| `bse.absorption_eigvecs` | `src/bse/STATUS.md:114-116` |
| `bse.bse_kpm` | delegated to by `bse_jax --kpm-dos` (bse_jax.py:512-537); docstring bse_kpm.py:11 |
| `bse.bse_feast` | delegated to by `bse_jax` default (non-`--lanczos`) route (bse_jax.py:539-604) |
| `bse.bse_pseudopoles` | `__main__` only (bse_pseudopoles.py:698); referenced by tests/archive/projects/test_isdf/sweep_12v12c_plot.py:33 (under the old `bse_isdf` package name) |
| `bse.pseudopoles_eval` | `__main__` (pseudopoles_eval.py:177); archive script import sweep_12v12c_plot.py:9 |
| `bse.pseudopoles_sweep` | `__main__` only (pseudopoles_sweep.py:338) |
| `bse.bse_w_exact` | `__main__` only (bse_w_exact.py:177); its output format is the `--ref` input of pseudopoles_sweep |
| `bse.feast_sweep`, `bse.feast_zolo_sweep`, `bse.feast_ellipse_mixed_sweep` | `__main__` only; no invocation found in runs/skills/scripts/tools/docs |
| `bse.bse_feast_dense_debug` | `__main__` only; numpy-only synthetic sanity check (no input file) |
| `bse.test_bse`, `bse.test_davidson_bse` | `__main__` smoke/bench scripts, NOT in pytest path (cleanup commit a0da0a5 message confirms); test_davidson docstring shows lxrun usage |
| `bse.write_eigenvectors` | `__main__` (write_eigenvectors.py:234); library import from test_bse.py:39 |
| `bse.eigvals_to_eps2` | `__main__`; docstring: "Used by BGW_COMPARE.md as the canonical … tool" (eigvals_to_eps2.py:25) |
| `src/solvers/*` | **no CLI in any solvers file** (grep `__main__\|argparse` → none); library only, re-exported via solvers/__init__.py |

Library entry consumed by pytest: `tests/test_eqp_bgw.py:78` imports
`bse.bse_io.read_bgw_eqp`.

## Common input plumbing (what `-i/--input` actually does)

Every restart-based CLI takes `-i/--input <cohsex.in>` and uses it for
four things — none of which go through the GW config parser:

1. **Restart bundle discovery** — `_find_restart_file` (bse_io.py:756-764):
   globs `<dir(input)>/tmp/isdf_tensors_*.h5` then `<dir>/isdf_tensors_*.h5`,
   first hit wins (sorted). The input file's *contents* are irrelevant here;
   only its directory matters. (`test_bse.py` has its own private
   `find_restart_file` duplicate, test_bse.py:~280-306.)
2. **`wfn_file` key** — `_parse_wfn_path` (bse_io.py:584-600): line-parses
   `key = value` pairs, takes `wfn_file` (default `"WFN.h5"`), resolves
   relative to the input dir. Used to read `ifmax`→n_occ
   (`resolve_n_occ`, bse_io.py:603-664), `kgrid` fallback
   (bse_io.py:402-406), `cell_volume` (bse_io.py:495-501), and the
   SymMaps IBZ→full map for eqp unfolding (bse_io.py:711-725).
3. **Head overrides** — `_parse_head_overrides` (bse_io.py:667-695): reads
   keys `vhead` and `whead_0freq` (case-insensitive, `float` then
   `complex`); blank value = absent. Priority: cohsex.in override →
   restart-file `vhead`/`whead` datasets (bse_io.py:467, 487-492).
   These feed the q=0 rank-1 head injection
   `gw.head_correction.apply_q0_head_rank1_sharded` (bse_io.py:503-507).
4. **n_occ resolution order** (bse_io.py:610-624): explicit `--n-occ` →
   WFN.h5 `ifmax` via `wfn_file` → `mean_enk < fermi_energy` hint →
   ValueError. The old "largest gap" auto-detect was deliberately removed
   (docstring lines 620-624).

So the **complete cohsex.in key set read by BSE code** is exactly:
`wfn_file`, `vhead`, `whead_0freq`. All three are also legal GW-parser
keys (gw_config.py:348-349, 362 in COHSEX_INPUT.md docs §`vhead`/
`whead_0freq`/`wfn_file`), but BSE re-parses them with two private ad-hoc
parsers instead of importing the config layer. `whead_imfreq` (the GW
imaginary-frequency sibling) is **not** read by BSE — only the ω=0 head.

## Flag catalog per CLI

### `bse.bse_jax` (bse_jax.py:352-484) — main dispatcher

Routing: `--ring-test` / `--ring-check` / `--debug-parallelism` short-circuit;
`--kpm-dos` → re-execs `bse_kpm.main(argv)` (bse_jax.py:512-537);
default (no `--lanczos`) → re-execs `bse_feast.main(argv)` (539-604);
`--lanczos` → `_preview_lanczos` (609-625), **TDA only** (606-607 raises
without `--tda`). NOTE `args, _ = parser.parse_known_args()` (line 484):
unknown/misspelled flags are **silently ignored**.

| flag | line | meaning | default | gotchas |
|---|---|---|---|---|
| `-i/--input` | 353 | cohsex.in (see plumbing above) | None | required for all routes except `--ring-test`/`--debug-parallelism` (parser.error at 492/508) |
| `--n-val` / `--n-cond` | 354-355 | bands below/above n_occ in the BSE window (Kramers pairs for SOC) | 4 / 4 | clamped to available with a warning (bse_io.py:422-427) |
| `--px` / `--py` | 356-357 | 2-D device mesh factors | 1 / 1 | forwarded ONLY to feast/kpm/ring-check routes; the `--lanczos` path **ignores them** — sharded branch uses `create_mesh_2d()` auto-factorization of ALL devices (bse_jax.py:229; bse_ring_comm.py:31-43), single-device branch hardcodes `1, 1` (bse_jax.py:295-300). STATUS.md:105 recommends `--px 2 --py 2` with `--lanczos` — silently ignored |
| `--n-eig` | 358 | eigenvalues to report | 5 | lanczos route only; FEAST route uses `--feast-ritz-count` instead |
| `--feast-n-lanczos` | 359 | Lanczos steps for FEAST spectral bounds | 10 | → feast `--n-lanczos` |
| `--feast-buffer` | 360 | E_max buffer fraction | 0.05 | → feast `--buffer` |
| `--feast-n-quad1/2` | 361-362 | quadrature points, FEAST iter 1 / 2+ | 4 / 8 | → feast `--n-quad1/2` |
| `--feast-quadrature` | 363-365 | `ellipse` \| `zolotarev` | ellipse | |
| `--feast-units-ev-per-ry` | 366-371 | Ry→eV for FEAST report | 13.6056980659 | |
| `--feast-ritz-count` | 372 | Ritz values per window | 4 | |
| `--gmres-max-iter` / `--gmres-tol` / `--gmres-seed` | 373-375 | shifted-solve GMRES controls | 10 / 1e-2 / 0 | `--gmres-seed` is ALSO reused as feast's `--kpm-seed` (bse_jax.py:587) |
| `--gmres-fp32` | 376-377 | FP32 data + GMRES for shifted solves | off | |
| `--tda` | 378-379 | Tamm-Dancoff | off | help says "Default is full non-TDA", but the `--lanczos` route *requires* `--tda` (line 606-607) |
| `--rpa` / `--bse` | 380-383 | kernel selection | **RPA is the default**: `use_rpa = args.rpa or not args.bse` (515, 542); lanczos `include_W = not (args.rpa or not args.bse)` (616) | omitting `--bse` silently drops the W term — footgun |
| `--kpm-window-count` | 384-385 | KPM-derived FEAST windows | 4 | doubles as `--n-windows` for the kpm-dos route (525) |
| `--lanczos` | 386 | run Lanczos preview instead of FEAST | off | |
| `--feast-window1/2 A B` | 387-398 | window override in eV, `auto` allowed for B | None | → feast `--window1/2` |
| `--write-eigs [N]` | 399-405 | write eigenvectors.h5 (`const=-1` = all n_eig) | None | lanczos route only; BGW-format via `write_eigenvectors_stream` (v-flip + Ry→eV) |
| `--max-lanczos-iter` | 406-412 | total Krylov dim bound | None (auto `max(30, min(200, dim//2))`, 270/322) | with `--block-size>1` divided by block size (273) |
| `--block-size` | 413-419 | block-Lanczos block | 1 | |
| `--lanczos-rtol` | 420-428 | Ritz-change early exit (>0 enables while_loop; block only) | 0.0 (fixed iters) | |
| `--lanczos-check-every` | 429-434 | convergence check cadence | 4 | |
| `--n-reorth` | 435-445 | partial reorth window; −1 = full reorth | −1 | full reorth essential for spinor BSE (help text; STATUS.md:15) |
| `--matvec-kind` | 446-454 | `ring` (shard_map+ppermute, low mem) \| `gather` (all_gather) \| `simple` (plain jit + sharding constraints) | **ring** | STATUS.md:16 claims simple is "default" — doc/code mismatch. Plumbed via `data["matvec_kind"]` dict entry (239) → bse_lanczos.py:155 |
| `--gather-t` | 455-459 | deprecated alias for `--matvec-kind=gather` (622) | off | cruft |
| `--solver` | 460-469 | `lanczos` \| `davidson` (diag ΔE precond) | lanczos | → `solver_kind`, bse_lanczos.py:188 |
| `--kpm-dos` | 470 | run KPM DOS and exit | off | **route currently crashes** — see Suspects |
| `--kpm-n-moments/-random/-lanczos` | 471-473 | Chebyshev M / stochastic R / bound steps | 100 / 4 / 100 | |
| `--kpm-emin-ev/emax-ev` | 474-475 | bound overrides | None | forwarded only when set (532-535) |
| `--kpm-plot-file` | 476 | DOS plot path | bse_dos_kpm.png | |
| `--eqp` | 477 | BGW eqp1.dat QP corrections | None | **lanczos route only** — not forwarded to feast/kpm argv (516-537, 543-602); IBZ→full unfold via SymMaps when input_file given, else 0.01 eV nearest-energy matching (bse_io.py:698-753) |
| `--n-occ` | 478 | occupied-band count override | None (WFN ifmax) | lanczos route only, same non-forwarding |
| `--ring-test` / `--ring-check` / `--components` | 479-482 | ring matvec smoke / vs-single-device check (+ per-term breakdown) | off | ring-check needs `-i` |
| `--ring-timing` | 481 | **parsed, never read** | off | dead flag — its consumer `ring_matvec_timing` was deleted in commit a0da0a5 ("removed unreachable ring_matvec_timing(...) call … plus the --repeat/--warmup CLI args that only fed it") but the flag itself survived |
| `--debug-parallelism` | 483 | random-data demo | off | |

### `bse.bse_feast` (bse_feast.py:1076-1153)

`-i/--input` (required), `--n-val`/`--n-cond` (4/4), `--px`/`--py` (1/1 —
here they ARE used: `_create_mesh_xy(px,py)` takes the first px·py of
`jax.devices()`, bse_feast.py:686-692), `--n-lanczos` (10, min bound
steps), `--n-lanczos-max` (50), `--lanczos-rtol` (5e-4), `--lanczos-atol`
(0.0), `--lanczos-patience` (2) — adaptive E_max Lanczos; `--buffer`
(0.05, E_max×(1+buffer), line 1193), `--n-quad1` (4, used ONLY in the
`n_quad_schedule` for FEAST iter 1, line 1272), `--n-quad2` (8, also used
for the printed QuadratureSpec, 1245), `--feast-ritz` (off — without it
the CLI only prints bounds/windows/diagonal counts), `--feast-ritz-count`
(4), `--rpa` (off → `include_W=not args.rpa` everywhere), `--windows-kpm`
(+ `--windows-kpm-count` 4, `--kpm-n-moments` 100, `--kpm-n-random` 4,
`--kpm-seed` 0, `--kpm-n-energy-pts` 2000, `--kpm-n-lanczos` 100) — KPM
DOS-partitioned windows; `--s-cutoff` (1e-6, overlap regularization floor
as fraction of max(S)), `--feast-iter` (2 subspace iterations),
`--quadrature` (ellipse|zolotarev), `--gmres-max-iter` (10), `--gmres-tol`
(1e-2), `--gmres-seed` (0), `--gmres-fp32` (off), `--tda` (off),
`--units-ev-per-ry` (13.6056980659), `--window1/2 A B` ('auto' → E_max,
`_parse_window_arg`, 1224-1236). No `--eqp`, no `--n-occ` — FEAST always
runs on DFT energies. Enables the JAX persistent compile cache on entry
(1157-1164).

### `bse.bse_kpm` (bse_kpm.py:326-357)

`-i/--input` (required), `--n-val`/`--n-cond` (4/4), `--px`/`--py` (1/1,
used via `_create_mesh_xy`), `--n-moments` (200; cost = R·M matvecs),
`--n-random` (4), `--n-lanczos` (100), `--buffer` (0.05), `--emin-ev` /
`--emax-ev` (None — skip Lanczos bound when set), `--seed` (0),
`--n-energy-pts` (2000), `--n-windows` (10), `--plot-file`
(bse_dos_kpm.png), `--ry-to-ev` (13.6056980659), `--rpa` (off), `--tda`
(off; non-TDA doubles the reported dimension, 378-379), `--nohead` (off —
"Use headless V/W0 arrays if present (V_qmunu_nohead, W0_qmunu_nohead)").
**`--nohead` is lost wiring AND the whole CLI is currently broken** — see
Suspects #1.

### `bse.bse_pseudopoles` (bse_pseudopoles.py:553-586)

Same base (-i, n-val/cond 4/4, px/py 1/1) plus: `--m0` (6, biased random
seeds per window), `--n-quad` (8, per half contour), `--p-keep` (4, bright
modes kept per window), `--n-tail` (2, stochastic tail pseudomodes),
`--gmres-max-iter` (10), `--gmres-tol` (1e-2), `--gmres-fp32` (off),
`--seed` (0), `--ry-to-ev` (13.6056980659), `--rpa`/`--tda` (off),
`--nohead` (off — same lost wiring), `--out` (bse_pseudopoles.h5),
`--s-cutoff` (1e-6), `--windows-kpm` (+count 4, kpm-n-moments 200,
kpm-n-random 4, kpm-seed 0, kpm-n-energy-pts 2000, kpm-n-lanczos 100),
`--buffer` (0.05), `--window1/2` (None). Quadrature is hardcoded
`"ellipse"` (line 672). Same `use_nohead=` crash as bse_kpm (line 599).

### `bse.bse_w_exact` (bse_w_exact.py:53-78)

`-i` (required), `--n-val`/`--n-cond` (4/4), `--px`/`--py` (1/1),
`--omega-ev` (0.0), `--eta-ev` (0.0) — evaluation point z = (ω+iη)/RyToeV;
`--cols` (None = all μ columns; comma-sep 0-based), `--n-cols` (None,
random sample when --cols absent), `--seed` (0), `--gmres-max-iter` (10),
`--gmres-tol` (1e-2), `--gmres-fp32` (off), `--rpa`/`--tda` (off — TDA
picks `build_bse_ring_matvec` vs `_full`, lines 103-106), `--ry-to-ev`
(13.6056980659), `--out` (Wc_exact.h5). Output attrs record use_tda /
include_W / gmres params (161-169).

### `bse.pseudopoles_eval` (pseudopoles_eval.py:129-141)

`--poles` (required, H5 from bse_pseudopoles), `--out`
(Wc_from_pseudopoles.h5, bse_w_exact-compatible), `--omega-ev` (0.0),
`--eta-ev` (0.0), `--cols` (None = all), `--windows` (None = all window
groups, comma-sep names). Reads `ry_to_ev` from the poles file attrs with
fallback **13.605693122994** (line 144) — a *different* constant from the
13.6056980659 used everywhere else in the package.

### `bse.pseudopoles_sweep` (pseudopoles_sweep.py:244-252)

`--poles` (required), `--ref` (required, reference Wc H5 from
bse_w_exact), `--omega-ev` (0.0), `--eta-ev` (0.0), `--cols`
("0,1,2,3,4,5,6,7,8"), `--plot-file` (sweep_pkeep.png), `--ry-to-ev`
(13.6056980659). Sweeps p_keep 1..max, prints rel-error/alpha, plots.

### `bse.feast_sweep` (feast_sweep.py:558-568)

`-i` (required), `--n-val`/`--n-cond` (4/4), `--full` (216 configs),
`--focused` (~66), neither → minimal; `--output`
(feast_sweep_results.json), `--units-ev-per-ry`, `--full-diag` (off —
exact eigenvalues via dense diag; otherwise uses hardcoded
`EXACT_EIGENVALUES_EV`, feast_sweep.py:549-554, valid ONLY for the
isdf_tensors_600 / n_val=4 / n_cond=4 / 3×3×1 Si dataset). Mesh hardcoded
1×1 (line 570) — no px/py.

### `bse.feast_zolo_sweep` (feast_zolo_sweep.py:262-278) / `bse.feast_ellipse_mixed_sweep` (feast_ellipse_mixed_sweep.py:254-269)

Both: `-i` (required), `--n-val`/`--n-cond` (4/4), `--px`/`--py` (1/1),
`--n-ritz` (8), `--gmres-max-iter` (20), `--gmres-tol` (list; zolo default
[1e-2,1e-3], ellipse [1e-2]), `--buffer` (0.05), `--n-lanczos` (10),
`--seed` (0), `--units-ev-per-ry`. Zolo adds `--n-quad` [4,6,8],
`--feast-iter` [1,2], `--rho-scale` [0.5,1.0,1.5] (all swept as Cartesian
product); ellipse adds `--n-quad1` [4] (list) × `--n-quad2` (8, scalar).
Both hardcode window [0,2] eV and the same Si-only
`EXACT_EIGENVALUES_EV`.

### `bse.bse_feast_dense_debug` (bse_feast_dense_debug.py:137-147)

Synthetic numpy-only: `--n` (200), `--seed` (0), `--n-ritz` (4),
`--n-quad` (4), `--gamma` (0.4, ellipse aspect), `--e-min` (1.0),
`--e-split` (2.0), `--e-max` (80.0), `--feast-iter` (1), `--solve-noise`
(0.0, simulates GMRES tolerance). No input file, no JAX.

### `bse.absorption_haydock` (absorption_haydock.py:299-327)

`-i` (required), `--n-val`/`--n-cond`/`--n-occ` (required, required,
required), `--eqp` (None), `--dipole` (dipole.h5, from
psp.get_dipole_mtxels), `--V-cell` (required, bohr³), `--n-iter` (200
Haydock steps per polarization), `--n-spin` (1; "2 only for collinear
spin-polarised"), `--n-spinor` (**default 2**), `--eta-eV` (0.1),
`--omega-min/max-eV` (0/15), `--n-omega` (1500), `--out-prefix`
(absorption_haydock), `--no-eps1` (skip Kramers-Kronig), `--matvec-kind`
(ring|gather|simple, default ring; `simple` → `build_bse_simple_matvec`,
else ring with `low_mem=(kind=="ring")`, lines 202-208). `include_W=True`
is **hardcoded** (204, 207) — no RPA switch; TDA implicit (resonant-only
recursion). Requires ≥2 devices (158-161: "Haydock route currently
requires the sharded matvec"). ε₂ prefactor `16π²/(V_cell·N_k·n_spin·n_spinor)`
matches BGW absh.f90 (STATUS.md:122). **Footgun:** `--n-spinor` default 2
halves ε₂ for scalar-relativistic runs unless overridden (compare
absorption_eigvecs which infers it).

### `bse.absorption_eigvecs` (absorption_eigvecs.py:91-117)

`--eigenvectors` (eigenvectors.h5, BGW format), `--dipole` (dipole.h5),
`--n-occ` (required), `--V-cell` (required), `--n-spin` (1), `--n-spinor`
(None → **inferred from `spin_kernel` in eigenvectors.h5**, line 124),
`--eta-eV` (0.1), `--omega-min/max-eV` (0/15), `--n-omega` (1500),
`--out-prefix` (absorption_eigvecs), `--no-eps1`. Pure numpy
post-processing (no -i, no restart bundle, no JAX mesh).

### `bse.davidson_absorption` (davidson_absorption.py:57-73)

`-i` (required), `--n-val`/`--n-cond` (required), `--n-occ` (None → WFN
ifmax), `--eqp` (None, recommended; explicit n_occ slice, NOT
nearest-energy matching — comment lines 90-92), `--dipole`
(**dipole_p_only.h5** — the `--skip-vnl` dipole for BGW `use_momentum`
parity), `--V-cell` (required), `--n-eig` (20), `--n-random-init` (5),
`--max-iter` (80), `--tol` (1e-7), `--out-prefix` (eigenvalues_davidson).
Matvec hardcoded `build_bse_simple_matvec(..., include_W=True)` (line
122): TDA + full kernel only, no --rpa/--tda/--matvec-kind. Mesh =
`create_mesh_2d()` auto (line 76).

### `bse.test_davidson_bse` (test_davidson_bse.py:188-207)

`-i` (required), `--n-val`/`--n-cond` (8/8), `--n-occ` (4 — note:
*defaulted*, unlike everywhere else), `--n-eig` (20), `--n-random-init`
(5), `--max-iter` (60), `--tol` (1e-6), `--eqp` (None), `--bgw-h5`
(**absolute sandbox default**
`/pscratch/.../runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8/eigenvectors.h5` —
machine-specific), `--no-W` (RPA-like debug).

### `bse.test_bse` (test_bse.py:310-321)

`-i` (required), `--n-val`/`--n-cond` (4/4), `--n-eig` (10), `--max-iter`
(50), `--n-warmup` (2), `--n-bench` (10), `--no-jit-lanczos` (Python-loop
Lanczos), `--write-eigenvectors` (None, H5 path). Single-device loader
(`load_bse_data_from_restart`, defined locally test_bse.py:42), private
restart-file finder.

### `bse.write_eigenvectors` (write_eigenvectors.py:178-210)

Positional `input_npz` (needs 'eigenvalues'+'eigenvectors' arrays),
`-o/--output` (eigenvectors.h5), `--n-val`/`--n-cond` (required),
`--nkx/nky/nkz` (1/1/1). Offline npz→BGW-H5 converter.

### `bse.eigvals_to_eps2` (eigvals_to_eps2.py:114-134)

`--files` (required, ≥1 BGW-format eigenvalues.dat), `--eta-eV` (0.05),
`--kernel` (lorentzian|gaussian, default lorentzian), `--n-max` (None =
all states; truncation for fair finite-Krylov comparison), 
`--omega-min/max-eV` (0/8), `--n-omega` (4001), `--out`
(eps2_compare.png), `--no-plot`, `--label` (None). Reads V_super, n_spin,
n_spinor from the file headers — no flags for them.

## Environment variables

| var | consumer | meaning / default |
|---|---|---|
| `JAX_ENABLE_X64` | bse_jax.py:10, test_bse.py:22 (`setdefault "1"`); runtime/__init__.py:56 | fp64 on by default; caller export wins |
| `LORRAX_EXTRA_MU_PAD` | runtime/padding.py:58 via `padded_mu_extent` at bse_io.py:442, 877 | TEST-ONLY extra μ-pad rows for device-count-invariance testing at fixed P; 0/unset in production ("NEVER set this in production", padding.py:54) |
| `ISDF_JAX_PROFILE_DIR` | common/jax_profile.py:12 (trace dir); echoed at test_bse.py:333-334 | enables JAX profiler tracing around `jax_profile.trace_section("bse_test")` |
| `STERN_DEBUG` | solvers/sternheimer_solve.py:44 (`bool(int(env,'0'))`, gate at :301) | Sternheimer solver debug prints; default 0 |
| `JAX_PLATFORMS`, `JAX_PROCESS_COUNT`/`JAX_NUM_PROCESSES`/`SLURM_NTASKS`, `JAX_PROCESS_INDEX`/`SLURM_PROCID`, `JAX_COORDINATOR_ADDRESS`/`SLURM_NODELIST`/`SLURMD_NODENAME`/`HOSTNAME`, `CUDA_VISIBLE_DEVICES`, `_LORRAX_JAX_DISTRIBUTED_DONE` | runtime/__init__.py:56-152, called from bse_jax.py:12 and absorption_haydock.py:26 | multi-process bootstrap (`init_jax_distributed`). Only bse_jax and absorption_haydock call it — the other multi-device CLIs (bse_feast run directly, davidson_absorption, bse_kpm, …) rely on single-process multi-GPU or on being exec'd through bse_jax |
| `XLA_FLAGS=--xla_force_host_platform_device_count=N` | advisory only — quoted in error strings bse_ring_comm.py:858, 914 | needed for CPU-only ring tests |
| `LORRAX_NGPU` | **not read by any src/ code** — appears only in usage docstrings (davidson_absorption.py:14, test_davidson_bse.py:15, STATUS.md:100) | consumed by the sandbox `lxrun` launcher, not by LORRAX |

## Sharding / mesh assumptions tied to flags

- `--px/--py` → `Mesh(devices[:px*py].reshape(px,py), ("x","y"))`
  (bse_feast.py:686-692). `create_mesh_2d()` (bse_ring_comm.py:31-43)
  instead auto-factorizes ALL devices with px = largest divisor ≤ √N.
- Canonical PartitionSpecs (`make_bse_shardings`, bse_ring_comm.py:46-63):
  `X = P(None,"x","y",None)` (batch, c on x, v on y, k replicated);
  `psi_x/psi_y = P(None,None,None,"x"/"y")` (μ sharded, bands replicated);
  `V = P("x","y")`; `W = P("x","y",None,None,None)`; `eps` replicated.
- `--matvec-kind` selects ring (shard_map+ppermute, low mem) / gather
  (all_gather) / simple (plain jit, XLA auto-partition) — bse_lanczos.py:155-164.
- Host-vs-device residency at load (`bse_io.py`): ψ(μ), V_q0, W_q are read
  **per-process as local HDF5 slabs into numpy** (e.g. `_read_wq_sharded`
  local block bse_io.py:328-347) and assembled with
  `jax.make_array_from_process_local_data` (line 352) — no process ever
  holds the full W_q host-side. Disk stores the LOGICAL μ extent;
  in-memory re-pads to `padded_mu_extent(n_rmu, grid_x*grid_y)`
  (bse_io.py:441-444). Band axes padded to mesh multiples
  (`_pad_axis_to_multiple`, 449-453); eqp overrides must re-pad likewise
  (bse_jax.py:260-261, davidson_absorption.py:104-107).

## TDA vs full-BSE, spin/nspinor per route

- `--tda` exists on: bse_jax, bse_feast, bse_kpm, bse_pseudopoles,
  bse_w_exact (choosing `build_bse_ring_matvec` vs `_full`,
  bse_w_exact.py:103-106; non-TDA doubles dim, bse_kpm.py:378-379).
- Lanczos preview (`bse_jax --lanczos`) is TDA-only by hard error
  (bse_jax.py:606-607). davidson_absorption, test_davidson_bse,
  absorption_haydock are TDA-only implicitly (resonant matvec, no flag).
- Spin knobs exist only on the absorption post-processing CLIs:
  `--n-spin` (default 1) and `--n-spinor` (eigvecs: infer from
  `spin_kernel`; haydock: default 2). The solver CLIs have no spin flags —
  nspinor is taken from the ψ array shape (e.g. bse_jax.py:138).

## Coupling to gw/ and isdf/

- `gw.head_correction.apply_q0_head_rank1_sharded` / `apply_q0_head_rank1`
  (bse_io.py:504, 818) — the only gw imports in src/bse; consume the
  `vhead`/`whead_0freq` config keys above plus restart `G0_mu_nu`.
- The entire BSE input side hangs off the **gw_jax restart bundle**
  `isdf_tensors_*.h5` (datasets `V_qmunu`, `W0_qmunu` + `W0_ready` attr
  (falls back to V_qmunu when absent — bse_io.py:376-379, i.e. unscreened
  W!), `psi_full_y`, `enk_full`, `G0_mu_nu`, `vhead`, `whead`, `kgrid`).
  No direct isdf/ imports; the ISDF coupling is entirely through this file
  contract (three accepted V/W layouts: 8-D legacy, 6-D transitional,
  3-D flat-q + `kgrid` attr — bse_io.py:280-324).

## Suspects

### Broken at HEAD (confirmed by signature read)

1. **`bse_kpm.main` and `bse_pseudopoles.main` crash on load**:
   both unconditionally pass `use_nohead=args.nohead` (bse_kpm.py:370,
   bse_pseudopoles.py:599) to `load_bse_data_from_restart_sharded`, whose
   signature at HEAD is
   `(restart_file, n_val=4, n_cond=4, fermi_energy=0.0, mesh_xy=None,
   pad_bands=True, *, input_file=None, cell_volume=None, n_occ=None)`
   (bse_io.py:358-369) — **no `use_nohead` parameter** → TypeError on
   every invocation, `--nohead` or not. `bse_jax --kpm-dos` delegates to
   `bse_kpm.main` (bse_jax.py:536) so that route dies too. The FEAST
   `--windows-kpm` path is unaffected (calls `run_kpm_dos` directly with
   the already-loaded data). History: a0da0a5 deleted bse_pseudopoles as
   "broken on main (import error)"; fe5e3e8 restored the files without
   reconciling them with the rewritten loader.

### Lost wiring (do NOT delete without deciding the feature's fate)

2. **`--nohead`** (bse_kpm.py:355, bse_pseudopoles.py:570): help promises
   headless arrays `V_qmunu_nohead` / `W0_qmunu_nohead`; no code anywhere
   at HEAD reads datasets by those names (`grep -rn nohead src/` → only
   the two argparse sites + the crashing kwargs). This was a head-on/off
   A/B debugging knob; the machinery behind it is gone.
3. **`--ring-timing`** (bse_jax.py:481): parsed; `args.ring_timing` never
   referenced (verified by regex over the file). Its consumer was
   deliberately removed in a0da0a5; the flag was missed.
4. **`--px/--py` on the `bse_jax --lanczos` route**: parsed and
   documented in STATUS.md's recommended command, but the path never
   reads them (auto mesh at bse_jax.py:229; hardcoded 1,1 at 295-300).
5. **`--eqp` / `--n-occ` on the FEAST and KPM routes of bse_jax**:
   accepted by the parser but not forwarded in the delegation argvs
   (bse_jax.py:516-537, 543-602) and not accepted by bse_feast/bse_kpm —
   a user passing `--eqp` on the default (FEAST) route silently gets
   DFT-energy windows/Ritz values.

### Conventions / footguns (not bugs)

6. Default kernel is **RPA**: `use_rpa = args.rpa or not args.bse`
   (bse_jax.py:515, 542) and `include_W=not(args.rpa or not args.bse)`
   (616). You must pass `--bse` to get D+V−W.
7. `parse_known_args` (bse_jax.py:484) silently ignores typo'd flags.
8. STATUS.md:16 says `--matvec-kind=simple` is the default; code default
   is `ring` (bse_jax.py:449).
9. `absorption_haydock --n-spinor` defaults to 2 (spinor) while
   `absorption_eigvecs` infers it from the file — a scalar-relativistic
   haydock run without the flag divides the ε₂ prefactor by 2.
10. `pseudopoles_eval` fallback Ry→eV constant 13.605693122994
    (pseudopoles_eval.py:144) differs in the 7th decimal from the
    package-wide 13.6056980659.
11. BSE re-parses cohsex.in with two private parsers
    (bse_io.py:584-600, 667-695) instead of the gw config layer — a
    parallel-parser redundancy; keys accepted by GW but silently ignored
    by BSE (everything except wfn_file/vhead/whead_0freq) give no
    warning.
12. `test_bse.py` keeps a private `find_restart_file` duplicate of
    `bse_io._find_restart_file` (test_bse.py:~280-306 vs bse_io.py:756).
13. Hardcoded machine paths / dataset-specific references:
    `test_davidson_bse --bgw-h5` default is an absolute /pscratch path
    (test_davidson_bse.py:203); the three feast sweeps embed
    `EXACT_EIGENVALUES_EV` valid only for one Si dataset
    (feast_sweep.py:549-554).
