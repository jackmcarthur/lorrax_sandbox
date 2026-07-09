# src/gw/sigma_x_bispinor.py — deep-read notes (2026-07-01)

Repo: /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
LOC: 199. Deep-read of every function; grep coverage over src/, tests/, tools/, scripts/.

## Purpose

Computes Σ^B, the **transverse-only bispinor bare-exchange self-energy** for the
Phase-1 DHF + bare-Breit design (`BISPINOR_DHFB_DESIGN.md` §3):

    Σ^B_{αβ}(12) = -Σ_{i,j∈{1,2,3}} γ̃^i_{αγ} G^0_{γδ}(12) γ̃^j_{δβ} D^{ij}_bare(12)

with γ̃^μ ≡ γ^0 γ^μ (γ̃^0 = I_4, γ̃^i = α^i in LORRAX convention) and
D^{ij}_bare = V_qmunu_TT_{ij} (transverse projector-weighted Coulomb).
It does **not** contain its own convolution kernel: it reuses the scalar
`sigma_sx` kernel from `cohsex_sigma._make_cohsex_kernels`, folding the γ̃
vertices into ψ (psi_xn on the left, psi_yr on the right) so the unchanged
`build_G → _convolve → project` chain evaluates the bispinor formula verbatim.
The charge channel Σ^C-bare (V_00 == V_qmunu_CC) is deliberately NOT here — it
runs through the ordinary scalar `compute_cohsex_sigma(compute_bare_x=True)`
path on 4-spinor wavefunctions.

Category: **physics: Σ_X (bare exchange) stage — bispinor/Breit transverse channel orchestrator**.

## Module-level constant

| Name | Line | Value | Role |
|---|---|---|---|
| `_TRANSVERSE_INDICES` | 59 | `(1, 2, 3)` | Lorentz indices summed in Σ^B. (0,i)/(i,0) tiles vanish by Coulomb gauge and are skipped; (0,0) handled by scalar path. Consumed by the double loop at L178-179 and by `tests/test_sigma_x_bispinor.py::test_unique_transverse_pairs_correct` (L123). |

## Function table

### `_wfns_with_lorentz_vertices(wfns, mu_L, nu_L)` — L62–100 (private)

- **Role**: return a `dataclasses.replace` clone of a `Wavefunctions` bundle with
  γ̃^{μ_L} folded into `psi_xn` (spin axis 1; shape comment `(nk, s, μ_X, n)`)
  and γ̃^{ν_L} folded into `psi_yr` (spin axis 2; shape comment `(nk, n, s, μ_Y)`).
  Uses `common.gamma_matrices.gamma_perm_phase` + `gamma_apply` (gather +
  per-element phase multiply; every γ̃^μ is a monomial matrix with values ∈ {±1, ±i} —
  no dense 4×4 matmul).
- **Physics**: left-vertex γ̃^{μ_L} insertion = `psi_xn ← γ̃^{μ_L} psi_xn` (spin axis);
  right-vertex γ̃^{ν_L} = `psi_yr ← γ̃^{ν_L} psi_yr`. Correctness relies on
  γ̃^ν being **Hermitian** (docstring L77-82): `build_G` conjugates `psi_yr`, so
  `conj(γ̃^ν ψ) = (γ̃^ν)^T ψ*` contracts with `psi_yn` to give `ψ†_{yr} γ̃^ν ψ_{yn}`.
  γ̃^0 = I_4 and γ̃^i = α^i both qualify.
- **Short-circuit**: `mu_L == 0` / `nu_L == 0` return the same array objects
  (no copy) — the "PSD scalar path" no-op. `psi_xr` / `psi_yn` always pass
  through unchanged (vertex sits on the build_G side only).
- **Callers** (grep `_wfns_with_lorentz_vertices` across src/tests/tools/scripts):
  - `compute_sigma_x_bispinor` (this file, L180)
  - `tests/test_sigma_x_bispinor.py` L80, L99 (`test_wfns_replace_no_op_for_00`)
- **Boundary arrays**: psi_xn `(nk, ns=4, μ_X, n)` c128 device; psi_yr
  `(nk, n, ns=4, μ_Y)` c128 device. Sharding inherited from the input bundle
  (transverse-centroid Wfns).
- **Inline import**: `from common.gamma_matrices import gamma_perm_phase, gamma_apply` (L84).

### `compute_sigma_x_bispinor(*, wfns_transverse, Gij, bispinor_v_q_path, meta, mesh_xy, backend=None, use_ffi_io=None, print_fn=print, verbose=True) -> jax.Array` — L103–199 (public)

- **Role**: orchestrator. Opens `v_q_bispinor.h5` via
  `gw.v_q_bispinor.BispinorVqReader` (context manager, L176), loops the 9
  (i, j) ∈ {1,2,3}² transverse pairs, per pair:
  1. `wfns_ij = _wfns_with_lorentz_vertices(wfns_transverse, i, j)` (L180)
  2. `V_ij = _pad_V_to_padded(reader.get_tile(i, j))` (L181)
  3. `contrib = sigma_sx_k(wfns_ij, Gij, V_ij)` (L182) — the same jit'd scalar
     SX kernel from `cohsex_sigma._make_cohsex_kernels(mesh_xy, meta.kgrid, nk_tot)` (L154-156)
  4. accumulates `sig_x_b += contrib` (L194).
  Returns Σ^B `(nk, nb_sigma, nb_sigma)` on the same sharding as scalar Σ_X.
- **Physics** (docstring L117-121, note the docstring formula is garbled at the
  end — see weird_code):

      Σ^B[k, m, n] = -Σ_{i,j∈{1,2,3}} ⟨m,k| γ̃^i V^{i,j}_{q=k-k'} γ̃^j |n,k⟩ ...

- **Inner helper `_pad_V_to_padded(V_logical)`** — L164–172: ψ arrives at
  PADDED n_rmu (`load_centroids_band_chunked` rounds to mesh-product); V tiles
  on disk are at LOGICAL extent. Zero-pads V's last two axes up to
  `n_rmu_T_padded = wfns_transverse.psi_yr.shape[-1]` so the convolve
  broadcasts. Comment: pad rows of ψ are zero ("Phase 3a invariant") so
  zero-padding V is exact. NOTE: `BispinorVqReader.get_tile` (v_q_bispinor.py
  L845-870) itself already returns mesh-padded tiles (`n_L_padded, n_R_padded`);
  this second pad only fires when the reader's mesh-padding differs from ψ's
  padding — hence the early-return equality check at L167.
- **Per-tile diagnostic** — L184-193: `tr = float(jnp.einsum('kmm->', contrib).real) * RYD_TO_EV`
  (einsum signature VERBATIM: `'kmm->'`), wrapped in `try/except Exception → nan`,
  stored in local dict `contributions[(i,j)]` and printed
  (`Σ^B tile (μ_L=i, ν_L=j): tr Σ = ... eV`) when `verbose and jax.process_index()==0`.
  Comment says it exists "for diagnostic comparison against agent-B's MoS2
  reference values (commit 69e8863)". The `contributions` dict is **never read
  or returned** (grep: only assignment at L191).
- **Callers** (grep `compute_sigma_x_bispinor` across src/tests/tools/scripts):
  - `src/gw/cohsex_sigma.py` L241/243 — inside `compute_cohsex_sigma`'s
    `compute_bare_x` branch, gated on `wfns_transverse is not None and
    bispinor_v_q_path is not None`; result added to `sig_x`.
  - `src/gw/cohsex_sigma.py` L316/318 — identical block in
    `compute_v_h_sigma_x` (the V-only two-kernel path).
  - No test calls the orchestrator end-to-end (test file covers only the
    γ-algebra and `_wfns_with_lorentz_vertices`; docstring L4-6 of the test file
    says the e2e smoke "belongs in a later integration test").
- **Upstream plumbing**: `src/gw/gw_jax.py` L187-191 builds the arguments:
  `wfns_transverse = getattr(isdf, 'wf_bundle_transverse', None)`;
  `bispinor_v_q_path = os.path.join(tmp_dir, 'v_q_bispinor.h5')` iff
  wfns_transverse is not None. Comment there: bundle is None when
  `bispinor=False` or `centroids_file_current` unset. So the effective config
  gates are **`cfg.bispinor`** and **`cfg.paths.centroids_file_current`**
  (consumed upstream in isdf prep, not read in this module).
- **Boundary arrays**:
  - `wfns_transverse`: `gw.wavefunction_bundle.Wavefunctions` sampled at the
    **transverse** centroid set r_{μ_T}; all 9 tiles share these centroids.
  - `Gij` `(nk, nb_sigma, nb_sigma)` — band-space occupation projector, same
    as scalar Σ_X.
  - `V_ij` from `reader.get_tile(i,j)`: `(n_q_total, n_L_padded, n_R_padded)`
    c128, sharded `P(None, 'x', 'y')` on mesh_xy (per v_q_bispinor.py L856).
    Hermitian companions `(j,i)` for j>i are synthesized by the reader as
    `conj(swapaxes(V[i,j], -1, -2))` — 6 disk datasets serve 9 tiles.
  - Return: Σ^B `(nk, nb_sigma, nb_sigma)` c128, sharding = whatever
    `sigma_sx_k` emits (caller in cohsex_sigma then does its own
    with_sharding_constraint on the summed sig_x).
- **Sync points**: `contrib.block_until_ready()` per tile (L183) and
  `sig_x_b.block_until_ready()` at the end (L198) — serializes the 9 kernel
  launches (each tile's convolution completes before the next V read).
- **Inline imports**: `_make_cohsex_kernels` (L154), `RYD_TO_EV` (L186 —
  **inside the double loop**, re-executed per tile).

## Flags / config keys consumed

- Directly in this module: **none** (no LorraxConfig access; all plumbing via
  keyword args).
- Effective upstream gates (documented in-module and at the gw_jax call site):
  `cfg.bispinor`, `cfg.paths.centroids_file_current` (aka
  `centroids_file_current` in cohsex.in / the ISDF prep) — these decide whether
  `wf_bundle_transverse` exists, which decides whether Σ^B runs at all.
- `use_ffi_io` / `backend` are passed through to `BispinorVqReader` (SlabIO
  FFI vs h5py read path).

## I/O

- **Reads**: `{tmp_dir}/v_q_bispinor.h5` via `BispinorVqReader` / SlabIO.
  Datasets consumed (names from `v_q_bispinor.tile_dataset_name`):
  `V_qmunu_TT_11`, `V_qmunu_TT_22`, `V_qmunu_TT_33`, `V_qmunu_TT_12`,
  `V_qmunu_TT_13`, `V_qmunu_TT_23`, each `(n_q_total, n_rmu_T, n_rmu_T)` c128;
  the (2,1)/(3,1)/(3,2) tiles are Hermitian-filled in-memory, not read.
  (`V_qmunu_CC`, `V_qmunu_CC_g0`, and `v_qmunu_format` live in the same file
  but are NOT touched by this module.)
- **Writes**: nothing. Stdout diagnostics only (per-tile trace lines).

## Cross-module dependencies

- `gw.v_q_bispinor` — `BispinorVqReader` (tile reads, Hermitian fill,
  mesh-padding), file written by `compute_V_q_bispinor_to_h5` /
  `compute_V_q_bispinor_g_flat_to_h5`.
- `gw.cohsex_sigma` — `_make_cohsex_kernels` (borrows the jit'd `sigma_sx_k`
  kernel; also the only two call sites of this module).
- `common.gamma_matrices` — `gamma_perm_phase`, `gamma_apply`.
- `common` — `RYD_TO_EV`.
- `gw.gw_jax` — argument plumbing (`wf_bundle_transverse`, tmp_dir path).
- `jax`, `jax.sharding.Mesh`, `dataclasses`.

## Dead suspects

- `contributions` dict (L175, L191): populated per tile, never read, never
  returned. Grep in-file: only the assignment; grep repo-wide for
  `contributions[` in this context: no external consumer. Write-only diagnostic
  residue (likely from the agent-B MoS2 cross-check session).
- No dead functions: both `_wfns_with_lorentz_vertices` (called at L180 + 2
  test sites) and `compute_sigma_x_bispinor` (2 call sites in cohsex_sigma.py)
  are live. Greps run: `grep -rn "sigma_x_bispinor\|compute_sigma_x_bispinor\|_wfns_with_lorentz_vertices" src tests tools scripts`.

## Redundancy suspects

- **Duplicated Σ^B integration block in cohsex_sigma.py**: the ~12-line
  "if wfns_transverse is not None and bispinor_v_q_path is not None: import,
  call, block, add" stanza appears verbatim twice (`compute_cohsex_sigma`
  L239-250 and `compute_v_h_sigma_x` L316-327). Not in this file, but this
  module is the cause; refactor should hoist one helper.
- **Double padding logic**: `_pad_V_to_padded` (L164-172) re-implements
  mesh/extent padding that `BispinorVqReader.get_tile` already performs
  (v_q_bispinor.py L845-870, `_padded_shape_LR`); the module even has a comment
  at v_q_bispinor.py L834 acknowledging the "further padding step in
  sigma_x_bispinor". Two padding sources of truth for the same μ-axis invariant
  (ψ pads to mesh-product via load_centroids_band_chunked; V pads to mesh axes
  via the reader; this function reconciles them). Candidate for unification in
  the refactor.
- **9 kernel calls vs 6 unique tiles**: the module docstring (L31-33) advertises
  "6 unique kernel calls + 3 Hermitian-conjugate fills delivered for free" —
  but only the V *reads* are deduplicated; the orchestrator still launches all
  9 `sigma_sx_k` convolutions (Σ^B_{ji} is NOT derived from Σ^B_{ij} at the
  Σ level). Potential 33% Σ^B compute saving if Σ-level Hermiticity is exploited
  (needs verification that tile-wise Σ^B_{ji} = Σ^B_{ij}† under the kernel).

## Weird code

1. **L119-121, garbled docstring formula**: the Σ^B formula in the
   `compute_sigma_x_bispinor` docstring ends with a stray
   `⟨m', k'| ⟨m, k|` fragment — two unmatched bra factors that make the
   equation nonsensical as written. Hypothesis: copy-paste truncation of the
   full matrix-element expression (the internal-band sum over m',k' got
   mangled). Code is unaffected (kernel reuse), docs need fixing.
2. **L184-193, in-loop diagnostic with hardcoded provenance**: per-tile trace
   `jnp.einsum('kmm->', contrib)` with a bare `except Exception: tr = nan`,
   tied by comment to "agent-B's MoS2 reference values (commit 69e8863)".
   Blanket exception swallow + write-only `contributions` dict + `from common
   import RYD_TO_EV` executed inside the double loop. Hypothesis: debugging
   scaffolding from the Milestone-A validation session left in the production
   path; also forces a device→host sync per tile beyond the existing
   block_until_ready.
3. **L183/L198, per-tile `block_until_ready`**: serializes all 9 tile
   convolutions. Possibly intentional (bounds peak memory: one V tile resident
   at a time, matching the v_q_bispinor "peak GPU memory equals one tile"
   design), but it also prevents any read/compute overlap. Refactor should
   decide deliberately.
4. **Docstring-vs-behavior mismatch on "6 unique kernel calls"** (L31-33): see
   redundancy suspects — actual behavior is 9 kernel calls, 6 unique disk reads.
5. **`_pad_V_to_padded` pads both axes to the same `n_rmu_T_padded`** (L162,
   L169-171) taken from `psi_yr.shape[-1]` only; correct today because all 9
   transverse tiles share one centroid set (square tiles), but it silently
   assumes n_L == n_R padding — would break if transverse channels ever get
   per-channel centroid counts.
6. **Known physics caveat (external)**: per project memory
   (`project_bispinor_tt_noncovariance`), the transverse **IBZ-unfold** feeding
   these V^{i,j} tiles gives wrong in-plane Σ^B on CrI3 (z exact, in-plane ~23%
   off vs full-BZ truth). The bug is on the v_q_bispinor unfold side, not in
   this module, but any refactor map should mark this consumer as affected;
   pragmatic fix on record: IBZ charge + full-BZ-direct transverse.

## Test coverage

`tests/test_sigma_x_bispinor.py` (CPU-only): γ̃^0 identity perm/phase;
`gamma_apply` vs dense γ̃ matmul on both psi_xn (axis 1, einsum ref
`'bs,ksxn->kbxn'`) and psi_yr (axis 2, einsum ref `'bs,knsx->knbx'`);
bit-identity no-op for (0,0); 9-pair enumeration sanity. No end-to-end
orchestrator test (explicitly deferred in the test module docstring).
