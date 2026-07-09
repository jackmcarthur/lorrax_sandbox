# src/gw/cohsex_sigma.py — deep-read notes (344 LOC)

Audit date: 2026-07-01, lorrax_D checkout, branch main.

## Purpose

Static COHSEX self-energy stage: builds Σ_SX (screened exchange), Σ_COH
(Coulomb hole), V_H (Hartree), and Σ_X (bare exchange) in the ISDF/flat-k
sharded representation. All heavy work is done by three jit'd kernels
produced by a cached factory; the module also owns the band-space
occupation projector `Gij` and the q→0 static head attach.

Physics implemented (from module docstring, signs as written):

```
Σ_SX(k)  = -project[ FFT[ G_occ(R) * W(R) / √Nk ] ]
Σ_COH(k) = +project[ FFT[ G_RI(R)  * (W − V)(R) / (2√Nk) ] ]
V_H(k)   =  project[ V(q=0) * ρ ]
Σ_X      =  same kernel as Σ_SX with V_q substituted for W_q
```

Category: **physics: static Σ (COHSEX/X/Hartree) stage**.

## Entry points (grep over src/, tests/, tools/, scripts/)

| symbol | callers (grep evidence) |
|---|---|
| `compute_cohsex_sigma` | `src/gw/gw_jax.py:359` (main one-shot driver), `src/gw/sigma_dispatch.py:165` (COHSEX mode of `compute_sigma_xc`, used by `sc_iteration.py`) |
| `compute_v_h_sigma_x` | `src/gw/sigma_dispatch.py:173` (X_ONLY / PPM modes) |
| `build_Gij` | `src/gw/gw_jax.py:342`; also internally by both drivers when `Gij=None` |
| `get_cohsex_kernels` | **no callers found anywhere** (grepped `get_cohsex_kernels` repo-wide) |
| `_make_cohsex_kernels` (private) | `src/gw/sigma_x_bispinor.py:154-156` reaches into the private factory to reuse the `sigma_sx` kernel |

No test/tool/script imports of this module were found
(`grep -rn 'cohsex_sigma' tests tools scripts` → empty).

## Function table

### `build_Gij(meta, mesh_xy)` — lines 37–51
- Role: occupation projector G_ij = diag(1..1,0..0) per k, shape
  `(nk_tot, nb_sigma, nb_sigma)` complex128; `nocc = min(meta.nelec, meta.nb_sigma)`.
- Built with **numpy on host** then `device_put` with fully-replicated
  `P(None, None, None)`. A prominent "NOTE TO FUTURE EDITORS" (lines 40–46)
  explains the jnp→numpy conversion (commit 7781b80, 2026-04-18: the jnp
  version fired 8 standalone pjits) and forbids reverting. Intentional.
- Consumers: `gw_jax.py:342`, both drivers here, forwarded into
  `sigma_x_bispinor.compute_sigma_x_bispinor`.
- Meta fields consumed: `meta.nelec`, `meta.nb_sigma`, `meta.nk_tot`.

### `_make_cohsex_kernels(mesh_xy, kgrid, nk_tot)` — lines 63–124
- Cached factory (module-global `_cohsex_kernel_cache`, line 60), key =
  `(id(mesh_xy), tuple(kgrid))`. `nk_tot` is redundant for the key but is a
  compile-time constant closed over by the Hartree kernel.
- Builds flat-k FFT helpers from `common.fft_helpers`:
  `make_flat_k_ifftn/fftn(mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')` and
  `make_flat_k_ifftn(mesh_xy, kgrid, V_FFT5D_SPEC, norm='ortho')`.
  Specs (from `wavefunction_bundle.py`):
  `G_FFT7D_SPEC = P(None,None,None,None,'x',None,'y')`,
  `V_FFT5D_SPEC = P(None,None,None,'x','y')`.
- Line 80: `_inv_sqrt_nk = -1.0 / jnp.sqrt(float(nk_tot))` — **the minus
  sign is baked into a constant named like a positive normalization**.

Inner jit'd kernels:

#### `_convolve(G_k, V_or_W, prefactor)` — lines 82–87
- `Σ = prefactor · FFT[ G(R) · V(R) · (−1/√Nk) ]`.
- `V_R = _V_ifftn(V_or_W)[:, None, :, None, :]` — broadcasts the 5D V(R)
  slab against the 7D G(R) slab (inserting band and spinor axes).

#### `sigma_sx(wfns, Gij, W_q)` — lines 89–98
- `Σ_SX = -project[ FFT[ G_occ(R)·W(R)/√Nk ] ]`;
  `G_occ = build_G(wfns.xn(s.sigma), wfns.yr(s.sigma), Gij=Gij)`
  (greens_function_kernel). Prefactor passed = `+1.0`; net sign −1 comes
  from `_inv_sqrt_nk`. Doubles as Σ_X kernel when called with `V_q`.

#### `sigma_coh(wfns, W_q, V_q)` — lines 100–106
- `Σ_COH = +project[ FFT[ G_RI(R)·(W−V)(R)/(2√Nk) ] ]`;
  `G_ri = build_G(wfns.xn(s.full), wfns.yr(s.full))` (no Gij → resolution
  of identity over the **full** band slice). Prefactor passed = `−0.5`;
  net +0.5 after `_inv_sqrt_nk`'s sign.

#### `hartree(wfns, Gij, V_q)` — lines 108–121
- `V_H(m,n,k) = <m| V(q=0, no G0) · ρ |n>`; uses `V_q[0]` (flat-q index 0
  must be q=0). Einsums verbatim:
  - `rho`: `'kisx,kjsx,kij->x'` over `(conj(psi_yr), psi_yr, Gij)`
  - `Vrho`: `'xy,y->x'` over `(V_q[0], rho / nk_tot)`
  - result: `'kmsx,x,knsx->kmn'` over `(conj(psi_xr), Vrho, psi_xr)`
- Axis convention: k=kpt, i/j/m/n=bands, s=spinor, x/y=ISDF centroid.

### `_add_static_head(sig_sx, sig_coh, *, static_head_terms, meta, mesh_xy, do_screened)` — lines 131–142
- No-op if `static_head_terms is None`. Otherwise calls
  `head_correction.static_head_terms_to_kij(..., do_screened=...)`, then
  **additionally zeroes coh_h when `not do_screened`** (line 138-139) even
  though `do_screened` was already passed into the converter — double
  gating, likely belt-and-braces. Heads device_put with replicated
  `P(None,None,None)` and added.

### `compute_cohsex_sigma(wfns, V_q, W_q, meta, mesh_xy, *, Gij=None, do_screened=True, static_head_terms=None, compute_bare_x=True, wfns_transverse=None, bispinor_v_q_path=None, backend=None, use_ffi_io=None)` — lines 149–258
- Top-level driver. Builds Gij if None, gets kernels from factory, runs
  sigma_sx(W_q), sigma_coh(W_q,V_q), hartree(V_q) under `with mesh_xy:`,
  attaches static head, pins all outputs to fully-replicated
  `P(None,None,None)` (documented rationale lines 192–198: outputs are
  small, `nk·nb_sigma²·16B`; heavy ω-grid Σ_c stays sharded in ppm_sigma),
  `block_until_ready` on each.
- `compute_bare_x=True`: reruns `sigma_sx_k(wfns, Gij, V_q)` for Σ_X,
  attaches the X head via a second direct
  `static_head_terms_to_kij(..., do_screened=False)` call (lines 229–232,
  not via `_add_static_head`).
- Bispinor branch (lines 240–251): if both `wfns_transverse` and
  `bispinor_v_q_path` given, lazily imports
  `sigma_x_bispinor.compute_sigma_x_bispinor` and adds Σ^B to sig_x.
  References BISPINOR_DHFB_DESIGN.md §3.
- Returns dict `{sig_sx, sig_coh, sig_h, sig_x}` (sig_x possibly None).
- Docstring caveat: caller must pass V_q as W_q when `do_screened=False`
  — the kernels don't test the flag themselves.

### `compute_v_h_sigma_x(wfns, V_q, meta, mesh_xy, *, Gij=None, static_head_terms=None, wfns_transverse=None, bispinor_v_q_path=None, backend=None, use_ffi_io=None)` — lines 261–334
- V-only fast path (X_ONLY and PPM modes via `sigma_dispatch`): runs only
  `hartree` and `sigma_sx(V_q)`; saves the two W-touching convolutions
  (docstring: ~half the cohsex_sigma wall on dense band manifolds).
- Head attach (lines 308–313) and bispinor branch (315–326) are
  near-verbatim copies of the corresponding blocks in
  `compute_cohsex_sigma`.
- Returns the same dict shape with `sig_sx`/`sig_coh` = `zeros_like(sig_x)`
  placeholders so downstream `cohsex["sig_sx"]` never sees None.

### `get_cohsex_kernels(meta, mesh_xy)` — lines 337–344
- Thin public wrapper over `_make_cohsex_kernels`. Docstring says
  "Exposed for the SC-COHSEX fixed-point loop" — but grep finds **zero
  callers**; `sc_iteration.py` instead goes through
  `sigma_dispatch.compute_sigma_xc` (sc_iteration.py:73), which calls the
  dict-returning drivers per iteration. Dead-code suspect.

## Cross-module dependencies
- `gw.greens_function_kernel.build_G` (G_occ / G_RI construction)
- `gw.head_correction.static_head_terms_to_kij` (q→0 head)
- `gw.wavefunction_bundle`: `project`, `G_FFT7D_SPEC`, `V_FFT5D_SPEC`,
  plus the bundle protocol `wfns.slices`, `wfns.xn/xr/yr/yn(bands)`
- `common.fft_helpers.make_flat_k_fftn / make_flat_k_ifftn` (lazy import
  inside the factory)
- `gw.sigma_x_bispinor.compute_sigma_x_bispinor` (lazy, bispinor Σ^B)
- Inbound private reach-in: `sigma_x_bispinor.py:154` imports
  `_make_cohsex_kernels` to reuse the sigma_sx kernel.

## Flags / config keys
None read directly from LorraxConfig in this file. Callers translate:
`config.do_screened` → `do_screened`, `config.do_G0` → whether
`static_head_terms` is built (gw_jax.py:352-355), `config.backend.slab_io`
→ `backend`, plus `use_ffi_io` forwarded opaquely to the bispinor path.
Mode routing (`COHSEX` vs V-only) is decided in `sigma_dispatch` from
`ComputeMode`.

## I/O
None. Pure in-memory compute. `bispinor_v_q_path` is a filesystem path
passed through untouched to `sigma_x_bispinor` (which does the actual
V_qmunu tile reads).

## Key arrays across the boundary
- `V_q`, `W_q`: flat-q `(nq, μ, μ)` device arrays, sharding per
  `V_FFT5D_SPEC` upstream; `V_q[0]` assumed = q=0 with G0 removed.
- `Gij`: `(nk_tot, nb_sigma, nb_sigma)` complex128, host-built,
  device_put fully replicated.
- Outputs `sig_sx/sig_coh/sig_h/sig_x`: `(nk, nb_sigma, nb_sigma)`,
  forced fully-replicated `P(None,None,None)`.
- G slabs are 7D (`G_FFT7D_SPEC`), V slabs 5D; `_convolve` broadcasts
  V_R into G_R's band/spinor axes via `[:, None, :, None, :]`.

## Suspects

### Dead
- `get_cohsex_kernels` (lines 337–344): zero callers by repo-wide grep for
  the name across src/tests/tools/scripts; its stated consumer
  (SC-COHSEX loop) actually uses `sigma_dispatch.compute_sigma_xc`.

### Redundancy
- `compute_v_h_sigma_x` vs `compute_cohsex_sigma`: the head-attach block
  (308–313 vs 229–233) and the bispinor Σ^B block (315–326 vs 240–251)
  are copy-paste duplicates; the V-only entry is essentially the
  `compute_bare_x` tail of the full driver. Classic parallel old/new
  path per the sandbox's known cruft pattern.
- Two top-level Σ-static call paths coexist: `gw_jax.py:359` calls
  `compute_cohsex_sigma` directly with W_q **in all modes** (comment at
  gw_jax.py:344-350 admits PPM overwrites sig_sx/sig_coh downstream,
  wasting the W convolutions), while `sigma_dispatch.compute_sigma_xc`
  has the proper mode routing that skips W kernels. One-shot vs SC
  drivers thus disagree on routing.
- `_add_static_head` exists for SX/COH but the bare-X head is applied via
  direct `static_head_terms_to_kij` calls in both drivers — three head
  attach sites for one concept.

### Weird
- Line 80: `_inv_sqrt_nk = -1.0/√Nk` — global sign flip hidden in a
  normalization-named constant; kernel prefactors (`1.0` for SX, `-0.5`
  for COH) are only correct in combination with it. Refactor hazard.
- Cache key `(id(mesh_xy), kgrid)` (line 71): `id()` of a GC'd mesh can be
  reused by a new Mesh object → stale kernels closing over a dead mesh.
  Cache is also never evicted (unbounded across mesh rebuilds).
- `_add_static_head` double-gates `do_screened` (passes flag to
  `static_head_terms_to_kij` AND zeroes `coh_h` afterwards, lines 136–139).
- `build_Gij` "DO NOT fix back to jnp" all-caps note (lines 40–46) —
  intentional and well documented, but a refactor tripwire.
- Docstring contract (lines 170–173): when `do_screened=False` the caller
  must substitute V_q for W_q; nothing enforces it.
