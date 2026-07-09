# Refactor map: src/centroid density providers

Group: `src/centroid/charge_density.py`, `src/centroid/current_density.py`
Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`
Date: 2026-07-01. Read on login node; grep-only verification (no execution).

---

## 1. src/centroid/charge_density.py (230 LOC)

### Purpose
Provides ρ_val(r) on the QE FFT grid as the weight field for k-means ISDF
centroid selection. Two sources: (1) read QE's already-symmetrized
`charge-density.hdf5` from the `<prefix>.save` dir (preferred, symmetric);
(2) compute ρ(r) = Σ_k w_k Σ_n |ψ_nk(r)|² from IBZ wavefunctions in WFN.h5
(fallback, NOT point-group symmetrized). `get_charge_density` is the unified
dispatcher with an `'auto'` mode that walks the run-directory layout looking
for a `*.save` dir.

Category guess: **preprocessing tool: centroid-selection weight field (ρ provider)**.

### Entry points / callers (grep over src, tests, tools, scripts)
- `get_charge_density` <- `src/centroid/kmeans_cli.py:27` (import), `:234` (call, passes
  `source=args.rho_source, save_dir=args.qe_save`). Only caller found.
- `rho_from_qe_save` <- internal only (`get_charge_density:225`). No external callers.
- `rho_from_wfn_ibz` <- internal only (`get_charge_density:230`). No external callers.
  Grep `rho_from_wfn_ibz|wfn_ibz` across src/tests/tools/scripts finds only this file
  plus the kmeans_cli argparse choice string `"wfn_ibz"` (line 60) — the path is only
  reachable via `--rho-source wfn_ibz` or auto-fallback when no `.save` dir is found.

### Function table

| Function | Lines | Role |
|---|---|---|
| `rho_from_qe_save(save_dir)` | 47–72 | Read ρ_val(r) from QE `charge-density.hdf5` via `file_io.qe_save_reader.CrystalData.from_qe_save(...).load_charge_density()` (lazy import). Returns `(Nx,Ny,Nz)` float64 host numpy. Physics: ρ(r) = FFT of ρ(G) written by QE at end of SCF — band+k summed, point-group symmetric, valence only (NLCC excluded). |
| `_load_wfn_k_fftbox_ibz(wfn, n_val)` | 79–100 | Load IBZ ψ into FFT box `(nk_irr, n_val, nspinor, Nx, Ny, Nz)` via `file_io.wfn_loader.WfnLoader.load(bands=(0,n_val), k="ibz")` + `common.wfn_transforms.to_box`. No fractional-translation phase (only \|ψ\|² needed). Builds a 1-device `Mesh(np.asarray(jax.devices()[:1]).reshape(1,1), axis_names=('x','y'))`. **BUG: `jax` is never imported** in this module (only `import jax.numpy as jnp` at line 37 and `from jax.sharding import Mesh` locally) → `jax.devices()` at line 97 raises NameError. Docstring says "P5 will switch the caller to pass a WfnLoader directly so this transient construction goes away" — a planned-refactor marker. |
| `rho_from_wfn_ibz(wfn, sym, n_val=None)` | 103–143 | Physics: ρ(r) = Σ_{k∈IBZ} w_k Σ_{n≤n_val} \|ψ_nk(r)\|². Delegates arithmetic to `psp.get_DFT_mtxels.compute_valence_density(wfn_k, sym, wfn)` (lazy import), which picks up `wfn.kweights` when leading dim of wfn_k == len(kweights), and handles the ecutrho>4·ecutwfc re-embed. `n_val` defaults to `int(wfn.nelec)` — correct for nspinor=2 (one band/electron); for nspinor=1 this is 2× the occupied band count (contrast: kmeans_cli's current-density branch explicitly uses `nelec//2` for scalar). Result is host numpy float64; NOT star-averaged/symmetrized (docstring caveat). |
| `_autodetect_save_dir(start=".")` | 150–171 | Walk cwd + parents looking for `*.save/charge-density.hdf5`; probes `qe/scf` and `qe/nscf` under each ancestor; stops at ancestor named `runs` or filesystem root. Encodes the sandbox run-directory layout. |
| `get_charge_density(wfn=None, sym=None, *, source="auto", save_dir=None, n_val=None)` | 174–230 | Dispatcher. `source ∈ {auto, qe_save, wfn_ibz}`. auto: prefer qe_save when a `.save` is findable, else wfn_ibz fallback (which is currently broken, see BUG above). Prints its decision. |

### Flags consumed
No LorraxConfig / cohsex.in keys. Consumes kmeans_cli CLI args indirectly:
`--rho-source {auto,qe_save,wfn_ibz}` (kmeans_cli.py:60), `--qe-save` (passed as
`save_dir`). Environment: relies on `$PWD` layout for auto-detection.

### Cross-module deps
`file_io.WfnLoader` (as WFNReader), `common.symmetry_maps.SymMaps` (type only; forwarded
to compute_valence_density), lazy: `file_io.qe_save_reader.CrystalData`,
`file_io.wfn_loader.WfnLoader`, `common.wfn_transforms.to_box`,
`psp.get_DFT_mtxels.compute_valence_density`, `jax.sharding.Mesh`.

### I/O
- Reads `<prefix>.save/charge-density.hdf5` (QE HDF5 ρ(G)) + `data-file-schema.xml`
  via `CrystalData.from_qe_save` (qe_save_reader.py:280 `load_charge_density` returns
  (ρ_r, ρ_G)).
- Reads `WFN.h5` via `WfnLoader(wfn._filename)` on the wfn_ibz path (accesses the
  private/legacy `_filename` attr, kept for "legacy WFNReader compat", wfn_loader.py:133).
- Writes nothing.

### Key arrays crossing boundaries
- `rho_r`: (Nx,Ny,Nz) float64, host numpy — returned to kmeans_cli, then re-uploaded
  as `jnp.asarray(..., float64)` (kmeans_cli.py:251).
- `wfn_k`: (nk_irr, n_val, nspinor, Nx,Ny,Nz) complex, device (to_box output on a
  1-device mesh) — transient, handed to compute_valence_density.

### Suspects
- **dead_suspects**: none at file level (kmeans_cli imports it). But the
  `wfn_ibz`/auto-fallback path is effectively dead-on-arrival: `jax.devices()`
  NameError at line 97 (module imports only `jax.numpy as jnp`). Grep evidence:
  `grep -n '^import jax' src/centroid/charge_density.py` → only line 37
  `import jax.numpy as jnp`. No test exercises `rho_from_wfn_ibz` (grep over
  tests/ finds zero hits).
- **redundancy_suspects**:
  - Fourth ρ-builder in the codebase: this file's `rho_from_wfn_ibz` (delegating to)
    `psp.get_DFT_mtxels.compute_valence_density`, plus archived
    `psp/archive/charge_density.py:build_density_from_ibz` (with its known-broken
    `_symmetrise_density`, per psp/dev_status.md:176), plus
    `qe_save_reader.load_charge_density`. The centroid module at least delegates
    rather than duplicates.
  - Docstring (line 15) records that `centroid/get_charge_density.py` was already
    removed — this module is the consolidation survivor.
- **weird_code**:
  - line 97: `jax.devices()` with no `import jax` → NameError (see above).
    Hypothesis: refactor from an eager WFNReader-based loader to WfnLoader/to_box
    was committed without exercising the fallback path (every run has a QE .save).
  - line 139: `n_val = int(wfn.nelec)` default double-counts bands for nspinor=1
    restricted-KS wfns (compare kmeans_cli.py:227-229 comment: "nelec for FR,
    nelec/2 for scalar"). Moot while the path is broken, but a trap post-fix.
  - `_autodetect_save_dir` hard-codes sandbox layout strings (`qe/scf`, `qe/nscf`,
    ancestor named `runs`) into library code — cwd-dependent behavior inside a
    "library" function.
  - Stale cross-reference elsewhere: `src/centroid/orbit_syms.py:19` cites
    "``charge_density._symmetrise_density``" which does not exist in this file
    (it lives only in `psp/archive/charge_density.py:130` and is documented broken).

---

## 2. src/centroid/current_density.py (185 LOC)

### Purpose
Builds the Gordon-decomposed Pauli-current weight field
W_curr(r) = Σ_{n∈occ,k,i} |j^Gordon_{n,k,i}(r)|² on the FFT grid, used as the
k-means weight for selecting *transverse* (current-channel) ISDF centroids in
bispinor runs (`kmeans_cli --density-mode current`). Loops full-BZ k-points one
at a time (WfnLoader handles the IBZ→full unfold) with an all-bands-at-once
jitted kernel per k.

Category guess: **preprocessing tool: bispinor transverse-centroid weight (physics: Gordon/Pauli current)**.

### Entry points / callers
- `build_current_density(wfn, sym, n_occ, *, verbose=True)` <-
  `src/centroid/kmeans_cli.py:226` (lazy import inside the
  `args.density_mode == "current"` branch), called at `:232`. Sole caller.
  `__all__ = ["build_current_density"]` (line 185).

### Function table

| Function | Lines | Role |
|---|---|---|
| `_build_K_cart(gvecs_k, kvec_frac, bvec_dimless, alat)` | 29–31 | Cartesian (k+G) in Bohr⁻¹: `(G + k_frac) @ bvec_dimless * (2π/alat)`. Host numpy. Module-private; no other callers (grep `_build_K_cart` → this file only). |
| `build_current_density(wfn, sym, n_occ, *, verbose=True)` | 34–182 | Driver. Precomputes FFT-grid momentum closures `Kc_x/Kc_y/Kc_z` (complex128, from `jnp.fft.fftfreq` × bvec_cart) captured as jit constants; stacks `sigmas = [σ_x,σ_y,σ_z]` from `common.gamma_matrices`. Loops `ik in range(sym.nk_tot)`: loads unfolded full-BZ ψ per-k via `WfnLoader(wfn._filename).load(bands=(0,n_occ), k=[ik], sharding=None)` ("U_spinor + τ-phase + TRS-conj + G-vector rotation in one place"; per-k loop keeps host RAM bounded), zero-pads scalar wfns into 2-spinor slots (`psi_G_k_np[:, :nspinor_wfn, :]`), accumulates `_kpt_contrib`. Normalizes by `/nk_full` at the end. Progress print every 5 s (with `block_until_ready`). Returns (Nx,Ny,Nz) float64 host numpy. |
| `_kpt_contrib(psi_G_k, K_cart_g, ng_x, ng_y, ng_z)` (jit closure, lines 67–134) | 67–134 | Physics per k, all bands: paramagnetic j^para_i = Im[ψ† ∂_i ψ] with ∂_iψ = IFFT[i(k+G)_i ψ(G)]; spin density s_i(r) = ψ† σ_i ψ; spin-curl term via FFT: curl_i = IFFT[i(K×s(G))_i]; Gordon current j^Gordon_i = j^para_i + ½ curl_i; returns Σ_n Σ_i (j^Gordon_{n,i})². Memory-shaped: each ∂_iψ computed and freed sequentially; each curl component touches only two s-components ("no 5-D blowup", per module docstring — the naive ∂_j s^k intermediate blew up). Scatter into FFT box via `.at[:, :, ng_x, ng_y, ng_z].set(...)`, `ifftn/fftn` with `norm='ortho'`. |

Einsum signatures (verbatim):
- `jnp.einsum('naxyz,naxyz->nxyz', jnp.conj(psi_r), grad_x).imag` (and _y, _z) — lines 94/97/100
- `jnp.einsum('naxyz,ab,nbxyz->nxyz', jnp.conj(psi_r), sigmas[i], psi_r).real` — lines 104–109

Equation implemented (module docstring, lines 3–9):
`j^Gordon_{n,k}(r) = Im[ψ_L† ∇ ψ_L] + (1/2) ∇ × (ψ_L† σ ψ_L)`;
`W_curr(r) = Σ_{n∈occ,k,i} |j^Gordon_{n,k,i}(r)|²`.

### Flags consumed
No LorraxConfig / cohsex.in keys. Reached only via kmeans_cli
`--density-mode current` (kmeans_cli.py:113–131, 225); caller computes
`n_occ = nelec (nspinor=2) | nelec//2 (nspinor=1)`.

### Cross-module deps
`common.gamma_matrices` (σ_x, σ_y, σ_z), `file_io.wfn_loader.WfnLoader` (lazy,
in-function import), `sym` (SymMaps: uses `sym.nk_tot`, `sym.unfolded_kpts[ik]`),
`wfn` attrs: `fft_grid`, `nspinor`, `bvec`, `alat`, `_filename`.

### I/O
- Reads `WFN.h5` via `WfnLoader(wfn._filename)`: `gvecs(k="full_bz")`
  (nk_full, ngkmax, 3), `ngk_valid(k="full_bz")`, per-k `load(...)`.
- Writes nothing (kmeans_cli writes the resulting centroids with an `_current`
  filename suffix).

### Key arrays crossing boundaries
- `gvecs_full`: (nk_full, ngkmax, 3) host. `psi_k`: (1, n_occ, ns, ngkmax) host
  from loader (sharding=None). `psi_G_k_np`: (n_occ, 2, ngk_k) complex128 host,
  uploaded per-k. `Kc_x/y/z`: (Nx,Ny,Nz) complex128 device closure constants.
- `rho_curr`: (Nx,Ny,Nz) float64 device accumulator → host numpy return.
- One JIT compile per distinct ngk_k shape (documented, line 39–40); peak GPU
  ≈ O(n_occ × FFT-grid) (docstring cites CrI3 6×6×1: ~7 GB on 1× A100).

### Suspects
- **dead_suspects**: none. Grepped `build_current_density` across src/tests/tools/scripts:
  only kmeans_cli (caller) and this file. No test coverage found.
- **redundancy_suspects**:
  - `tests/archive/current_density.py` — an older standalone
    `compute_current_density(wfn_file, output_h5=...)` script (writes
    `current_density.h5`, dataset `'current_density'`, plots slices). Archived
    parallel implementation of the same physics; grep-only evidence, no live callers.
  - The FFT-box scatter (`zero_box.at[...,ng_x,ng_y,ng_z].set`) + ifftn pattern
    re-implements what `common.wfn_transforms.to_box` (used by charge_density.py)
    does — two in-house ψ(G)→ψ(r) embeddings in the same subpackage.
- **weird_code**:
  - Lines 53–64: momentum grids `Kc_*` cast to complex128 and captured as jit
    closure constants — 3 full complex FFT-grid arrays resident for the whole run;
    real dtype would halve that. Hypothesis: complex to avoid dtype promotion
    inside `1j * (...)` products.
  - Lines 155–157: scalar (nspinor=1) ψ zero-padded into spinor slot 2 so the σ
    einsums run unchanged; s_x=s_y=0, s_z=|ψ|² — the "current" weight for scalar
    wfns silently includes a ½∇×(ẑ|ψ|²) term. Intentional (spin-up embedding)
    but worth a comment; mode is advertised as bispinor-only.
  - Line 174: `/ nk_full` normalization applied once at the end (weights are
    uniform full-BZ); contrast charge path which uses wfn.kweights — consistent
    but a convention asymmetry between the two providers.
  - Line 144: private-attr access `wfn._filename` (same legacy-compat pattern as
    charge_density.py).
  - No fractional-translation phase question here — WfnLoader's full-BZ unfold
    applies τ-phase + TRS conjugation (comment lines 140–142), unlike the IBZ
    charge path which deliberately skips phases.

---

## Cross-file observations
- Both files are ρ-weight providers for `kmeans_cli` only; natural refactor target:
  a single `centroid/density_weights.py` with `scalar` and `current` providers
  behind the existing `--density-mode` switch.
- Both reach into `wfn._filename` to re-open WFN.h5 with WfnLoader while being
  handed an already-open reader — the "P5" note in charge_density.py:88-90 says
  this transient double-open is slated for removal by passing a WfnLoader directly.
- The charge fallback path (`wfn_ibz`) is broken (missing `import jax`) and
  untested; the qe_save path is what production uses.
