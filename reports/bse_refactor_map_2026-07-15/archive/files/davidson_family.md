# BSE Davidson family — deep-read notes (4 files, 935 LOC)

Files covered:

| file | LOC | role |
|---|---|---|
| `src/solvers/davidson.py` | 353 | shape-agnostic block Davidson eigensolver (shared with DFT NSCF) |
| `src/bse/bse_davidson_helpers.py` | 207 | BSE-layout initial subspace + ΔE diagonal preconditioner |
| `src/bse/davidson_absorption.py` | 217 | CLI driver: BSE eigenpairs + dipole projections → BGW `eigenvalues_b{1,2,3}.dat` |
| `src/bse/bse_preconditioner.py` | 158 | older diagonal-preconditioner utilities — mostly dead, one live helper |

Audit date: 2026-07-15, lorrax_D checkout. **Note**: the prompt named branch
`agent/slate-linalg-ffi @ e18d0e5`; the checkout actually at HEAD during this
audit is `agent/ppm-fit-conditioning @ adc2197`. All line numbers below are
from that HEAD.

## Purpose

The exact-diagonalization arm of the BSE stage: converge the lowest `n_eig`
eigenpairs of the TDA BSE Hamiltonian in the ISDF pair basis, per state to a
residual tolerance (LORRAX analogue of BGW `absorption.x diagonalization`,
vs the Lanczos/Haydock spectrum-shape arm in `bse_lanczos`/`absorption_haydock`).

The operator being diagonalized (built elsewhere, in `bse_simple` /
`bse_ring_comm`) is, signs as written at `bse_simple.py:175`:

```
H·X = D_term + HX_V − HX_W                       (TDA, resonant block only)

D_term[b,c,v,k]  = (ε_c[k,c] − ε_v[k,v]) · X[b,c,v,k]          bse_simple.py:82-83
HX_V[b,c,v,k]    = (1/Nk) Σ_μν M*[k,c,v,μ] V_q0[μ,ν] Σ_{c'v'} M[k,c',v',ν] X[b,c',v',k]
HX_W[b,c,v,k]    = (1/Nk) Σ_{νs} ψ_v[k,v,s,ν] Σ_μ ψ_c*[k,c,t,μ] (FFT⁻¹W FFT ∘ T)[b,μ,ν,t,s,k]
M[k,c,v,μ]       = Σ_s ψ_c*[k,c,s,μ] ψ_v[k,v,s,μ]              (spin-traced pair amplitude)
```

This family contributes:

1. `solvers.davidson.davidson` — physics-free block Davidson (after QE
   `cegterg.f90`): Gram projection `Hc[m,n] = Σ_… V*[m,…] HV[n,…]`,
   generalized eigh via Cholesky, Ritz reconstruction
   `X[n,…] = Σ_m C[m,n] V[m,…]`, residual `R = HX − λX`, CGS2
   re-orthonormalisation, fixed-block expansion, restart at `m_max`.
2. `bse_davidson_helpers` — the BSE plugs: initial subspace = unit vectors at
   the lowest `(c,v,k)` transitions sorted by `ΔE[k,c,v] = ε_c[k,c] − ε_v[k,v]`
   plus a random Gaussian tail; preconditioner
   `P = R / (ΔE[c,v,k] − λ + ε_shift)`, per-state renormalised
   (BSE analogue of QE's `g_psi`).
3. `davidson_absorption` — end-to-end CLI: load restart bundle → optional
   BGW-eqp QP shift → simple matvec wrapped in `lax.scan` over the batch axis
   → Davidson → per-state dipole projections
   `proj[S,α] = Σ_{cvk} A*_S[c,v,k] d_α[c,v,k]` → BGW-format
   `eigenvalues_b{1,2,3}.dat` (eV, `|⟨0|r̂|S⟩|²` columns).
4. `bse_preconditioner` — an earlier, unsharded take on the H-diagonal
   (`diag(H) = ΔE + V_diag − W_diag`, shifted inverse `1/(z − diag H)` for
   FEAST-style linear solves). Superseded by
   `bse_feast.build_preconditioner_diagonal_sharded`; only `energy_diff_cv_k`
   is still called.

Category: **pipeline stage (BSE eigensolver) on top of shared solver infra**
(`solvers/davidson.py` also drives the DFT NSCF in `psp/run_nscf.py` — it is
NOT BSE-private).

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox runs/skills/scripts)

| symbol | callers (grep evidence) |
|---|---|
| `python -m bse.davidson_absorption` (`main`) | documented usage in its own docstring (line 14); production evidence `runs/Si/C_bse_davidson_profile_2026-04-29/manifest.yaml:27` + `davidson_absorption.out` ("Converged 100 eigvals in 18 iter / 209 s on 4xA100"). No src importers. |
| `solvers.davidson.davidson` | `bse/davidson_absorption.py:31`, `bse/test_davidson_bse.py:38`, `bse/bse_lanczos.py:189` (Davidson branch of `solve_bse_sharded`, reached from `bse_jax --solver davidson`, `bse_jax.py:462`), `psp/run_nscf.py:46` (DFT NSCF); re-exported `solvers/__init__.py:2`. |
| `solvers.davidson.warmup_davidson_jit` | same four sites: `davidson_absorption.py:170`, `test_davidson_bse.py:243`, `bse_lanczos.py:216`, `psp/run_nscf.py:197`. |
| `solvers.davidson._to_host` (private) | reach-in import `bse/test_davidson_bse.py:295`. |
| `init_bse_subspace`, `bse_diagonal_precond` | `bse_lanczos.py:190,204,207`, `test_davidson_bse.py:43,226,236`, `davidson_absorption.py:33,156,165`. |
| `bse_preconditioner.energy_diff_cv_k` | `bse_feast.py:22,108,829`; `bse_jax.py:40,90`; `bse_serial.py:9,34,56` (and via `bse_serial.apply_D` in XLA dumps under `runs/Si/04_si_4x4x4_bse/.../xla_dump`). |
| `bse_preconditioner.{BSEPreconditionerTerms, _pair_amplitude, compute_v_diagonal, compute_w_diagonal, extract_w_q0, build_preconditioner_terms, build_shifted_preconditioner}` | **NONE FOUND** — repo-wide grep of src/tests/tools/scripts/docs plus sandbox runs/skills/scripts finds zero callers; only a docstring mention at `bse_davidson_helpers.py:156`. |

No pytest coverage of any of the four files (`grep -rln davidson tests/ tools/ scripts/` → only `docs/architecture/codebase.md`). The only "test" is
`bse/test_davidson_bse.py`, a `python -m` smoke/comparison script with no asserts.

## Function tables

### `src/solvers/davidson.py`

| function | lines | notes |
|---|---|---|
| `_to_host(arr)` | 45–64 | numpy pass-through; single-process `device_get`; multi-process `process_allgather(tiled=False)` → take row 0. Mirrors `file_io._slab_io_allgather._to_host` (`file_io/_slab_io_allgather.py:36`). |
| `_generalized_eigh(A, B)` | 71–84 | `A v = λ B v` via Cholesky of `B + 1e-12·I`, `C = L⁻¹ A L⁻ᴴ`, symmetrize, `eigh`, back-solve `Lᴴ x = V`. Small replicated (m,m) only. |
| `_ritz_and_residuals(V, HV, n_eig)` | 87–124 | jit, `static_argnames=('n_eig',)`. `Hc[m,n]=Σ_… V*[m,…]HV[n,…]` (`'m...,n...->mn'`), symmetrize Hc/Sc (line 101–102 — **Hermitian H assumed**), Ritz `X[n,…]=Σ_m C[m,n]V[m,…]` (`'mn,m...->n...'` — trailing sharding inherited from V), `R = HX − λX`, `‖R‖` over all trailing axes. Recompiles per subspace size m — hence the warmup. |
| `_default_precond(R, λ)` | 127–132 | identity: per-state normalised R. |
| `_orthonormalise_batch(P)` | 135–147 | jit; Cholesky of `PᴴP + 1e-12·I`, `P ← conj(L⁻¹)·P` (`'mi,i...->m...'`). Used once on V0. |
| `_ortho_expand(V, P)` | 150–178 | jit; CGS2 against V (×2: `P ← P − Σ_m (V*·P)[m,n] V[m,…]`) then Cholesky self-orthonormalisation. Comment: without it eigenvalues blow up to ~1e40 by iter ~50. |
| `warmup_davidson_jit(n_eig, trailing_shape, m_max, *, dtype, sharding)` | 185–219 | pre-compiles `_ritz_and_residuals` at m ∈ {n_eig, 2n_eig, …, m_max} with dummy buffers under the production sharding (compile-cache key match). |
| `davidson(apply_H, *, n_eig, precond_fn, init_fn, X0, m_max=4·n_eig, max_iter=100, tol=1e-8, verbose)` | 226–352 | loop: `_ritz_and_residuals` → `precond_fn(R, Λ)` → host convergence check `‖R‖ < tol·max(1,|λ|)` counting only the **leading contiguous** converged states (311–318) → `_ortho_expand` → `apply_H(P)` → concat → restart to Ritz vectors when `m > m_max` (344–347). Returns `(np eigenvalues, jax.Array eigenvectors)` preserving trailing sharding. |

### `src/bse/bse_davidson_helpers.py`

| function | lines | notes |
|---|---|---|
| `_gather_to_host(arr)` | 41–58 | verbatim copy of `solvers.davidson._to_host`. |
| `init_bse_subspace(eps_c, eps_v, n_eig, *, n_random=5, mesh, sharding, seed=0, dtype=c128)` | 61–138 | host numpy: `ΔE[k,c,v] = eps_c[k,:,None] − eps_v[k,None,:]` (line 109); flatten (`f = k·nc·nv + c·nv + v`, unravel `(nk,nc,nv)` → `(k_idx,c_idx,v_idx)`, line 113–115 — index math checked, correct); keep `flat > 1e-12` finite entries; place `V[i, c_i, v_i, k_i] = 1` for the `n_eig − n_random` lowest; append `n_random` seeded complex Gaussians, unit-normalised. Upload via `make_array_from_callback` under the given sharding (canonically `P(None,"x","y",None)`). NOT orthonormalised here — `davidson` does it (davidson.py:292). |
| `bse_diagonal_precond(eps_c, eps_v, *, epsilon_shift=1e-3, sharding)` | 145–203 | returns `precond_fn(R, Λ)`. Inner jit `_impl` recomputes `ΔE[c,v,k] = eps_c.T[:,None,:] − eps_v.T[None,:,:]` (line 185) at **call time from arguments** (multi-process: no sharded closure), constrains to `sharding`, `P = R / (ΔE − λ + ε)` with `|denom| < 1e-12 → 1e-12` clamp, per-state renormalise. `epsilon_shift` in Ry (1e-3 ≈ 13.6 meV). |

### `src/bse/davidson_absorption.py`

| function | lines | notes |
|---|---|---|
| `_gather_to_host(arr)` | 43–53 | third verbatim copy of `_to_host`. |
| `main(argv=None)` | 56–212 | argparse → `_find_restart_file(input)` → `create_mesh_2d()` (px = largest factor ≤ √n_dev, `bse_ring_comm.py:31-43`) → `load_bse_data_from_restart_sharded` (bse_io.py:358) → optional EQP branch (93–111): re-read `enk_full`, `apply_eqp_corrections`, re-slice `val_idx = arange(n_occ_eff − n_val_eff, n_occ_eff)`, `cond_idx = arange(n_occ_eff, n_occ_eff + n_cond_eff)`, re-pad eps to mesh multiples (zeros, `bse_io.py:139-146`) — explicitly n_occ-based, NOT nearest-energy matching. Matvec: `build_bse_simple_matvec` + precomputed `W_R = ifft₃(W_q)` (123–126); `matvec_scan` (142–149) wraps the m-batch in `lax.scan` over m=1 slices because the simple matvec's workspace scales ~m^1.5 (comment 134–137: 4.5 GB at m=10 → 88 GB at m=50); psi/eps/W_R/V_q0 passed as jit **arguments** (multi-host closure rule, 138–141). Then `init_bse_subspace` (seed=42), `bse_diagonal_precond` (`P("x","y",None)`), `warmup_davidson_jit(m_max=4·n_eig)`, `davidson(tol, max_iter, verbose=rank0)`. Post: Ry→eV (`ryd2ev = 13.6056980659`, line 192), dipole slice + `proj[S,a] = Σ_{cvk} A*_S d_a` (einsum `"Scvk,acvk->Sa"`, line 203), `write_eigenvalues_dat(..., n_spin=1, n_spinor=2, vol_supercell=V_cell·nk)` per polarisation (206–211), rank0 only. |

### `src/bse/bse_preconditioner.py`

| function | lines | notes |
|---|---|---|
| `BSEPreconditionerTerms` | 22–26 | NamedTuple (delta_E, V_diag, W_diag), each `(nc,nv,nk)`. Dead. |
| `energy_diff_cv_k(eps_c, eps_v)` | 29–41 | `ΔE[c,v,k] = eps_c[k,c] − eps_v[k,v]` via `eps_c.T[:,None,:] − eps_v.T[None,:,:]`. **The one live function** (bse_feast/bse_jax/bse_serial). |
| `_pair_amplitude(psi_c, psi_v)` | 44–46 | `M[k,c,v,μ] = Σ_s ψ_c*[k,c,s,μ] ψ_v[k,v,s,μ]`. Dead; duplicates `bse_serial.compute_pair_amplitude` (bse_serial.py:27). |
| `compute_v_diagonal(psi_c, psi_v, V_q0, nk)` | 49–72 | `V_diag[c,v,k] = (1/Nk) Σ_{μν} M*[k,c,v,μ] V_q0[μ,ν] M[k,c,v,ν]` (`"kcvM,MN,kcvN->cvk"` — index math checked, matches docstring). Dead. |
| `compute_w_diagonal(psi_c, psi_v, W_q0, nk)` | 75–100 | **einsum bug, see Suspects** — code disagrees with its own docstring. Dead. |
| `extract_w_q0(W_q)` | 103–105 | `W_q[:, :, 0, 0, 0]` from `(μ,ν,qx,qy,qz)`. Dead (bse_feast inlines the same slice at bse_feast.py:98). |
| `build_preconditioner_terms(...)` | 108–127 | bundles the three diagonals. Dead. |
| `build_shifted_preconditioner(..., z, return_inverse=True)` | 130–158 | `diag(H) = ΔE + V_diag − W_diag`; returns `1/(z − diag H)` or `(z − diag H)`. Dead — `bse_feast._get_gmres_solver` builds `m_inv = 1/(z − diag_h)` inline (bse_feast.py:133-135) from the sharded `build_preconditioner_diagonal_sharded` instead. |

## Flags / CLI args consumed

Only `davidson_absorption` reads flags; the other three files are pure APIs.

| flag | meaning | default |
|---|---|---|
| `-i/--input` | cohsex.in — locates the gw_jax restart h5 (`_find_restart_file`), and its `wfn_file` key resolves n_occ via WFN.h5 ifmax | required |
| `--n-val` | valence-band count in BSE window (slice `arange(n_occ−n_val, n_occ)`) | required |
| `--n-cond` | conduction-band count (`arange(n_occ, n_occ+n_cond)`) | required |
| `--n-occ` | occupied-band count override | None → WFN ifmax (`resolve_n_occ` → `WfnLoader.nelec` = `max(ifmax)`, wfn_loader.py:162) |
| `--eqp` | BGW eqp.dat; QP-corrects `enk_full` before band slicing | None |
| `--dipole` | dipole h5 from `psp.get_dipole_mtxels`; `--skip-vnl` file for BGW `use_momentum` match | `dipole_p_only.h5` |
| `--V-cell` | unit-cell volume bohr³; header `vol = V_cell·nk` | required |
| `--n-eig` | eigenpairs to converge | 20 |
| `--n-random-init` | random tail of initial subspace | 5 |
| `--max-iter` | Davidson cap | 80 |
| `--tol` | relative residual `‖R‖ < tol·max(1,|λ|)` | 1e-7 |
| `--out-prefix` | output `.dat` prefix | `eigenvalues_davidson` |

Indirect flag routing into this family: `bse_jax --solver {lanczos,davidson}`
(bse_jax.py:461-469) → `solve_bse_sharded(solver_kind=...)`; its
`davidson_n_random_init=5` / `davidson_eps_shift_Ry=1e-3` parameters
(bse_lanczos.py:113-114) have **no CLI flags** — bse_jax never passes them
(bse_jax.py:287 forwards only `solver_kind`), so they are API-only knobs.
Config keys read from cohsex.in on the load path used here: `wfn_file`,
`vhead`, `whead_0freq` (bse_io.py:668-696). Env: `LORRAX_NGPU` is consumed by
the `lxrun` wrapper, not by these modules.

## Sharding / PartitionSpec assumptions

- Trial-vector layout `X: (m, nc_pad, nv_pad, nk)` sharded
  `sh.X = P(None, "x", "y", None)` — c on x, v on y (`make_bse_shardings`,
  bse_ring_comm.py:46-64). Other specs used here: `sh.W = P("x","y",None,None,None)`
  (μ,ν sharded; FFT axes replicated), `sh.V = P("x","y")`,
  `sh.psi_x/psi_y = P(None,None,None,"x"/"y")`, `sh.eps = P(None,None)`
  (replicated), preconditioner ΔE `P("x","y",None)`.
- `solvers/davidson.py` is deliberately sharding-blind: every contraction is an
  ellipsis einsum. `'m...,n...->mn'` sums ALL trailing axes → (m,m) replicated;
  `'mn,m...->n...'` leaves trailing axes untouched → output inherits V's spec
  (comment 108–111). It never reshapes/flattens/gathers trailing axes and
  inserts no collectives of its own.
- Compile-cache discipline: `warmup_davidson_jit` places dummies under the
  production sharding so cache keys match (185–219); `bse_lanczos` comments
  ~2 s per missed compile otherwise.
- Multi-process rules (module docstring bse_davidson_helpers 14–26): never
  close a jit over a sharded array — `bse_diagonal_precond._impl` and
  `davidson_absorption.matvec_scan` take eps/ψ/W_R/V_q0 as call-time args;
  host fetches go through `process_allgather(tiled=False)[0]`.
- `create_mesh_2d` factorises the device count as px·py with px the largest
  factor ≤ √n — a prime device count degenerates to 1×n.

## Host vs device residency

- `init_bse_subspace` builds V0 **on host** as dense numpy
  `(n_eig, nc, nv, nk)` complex128, then per-shard upload via
  `make_array_from_callback` (130–135). eps tensors gathered to host first
  (tiny, ~kB).
- `bse_diagonal_precond`: ΔE `(nc,nv,nk)` rebuilt **on device inside the jit
  each call** from eps args; nothing big lives in the closure.
- `davidson`: V/HV device-resident, grow to `(m_max, trailing)`
  (m_max = 4·n_eig by default — with n_eig=100 on Si 8×8×64 that is
  400·4096·16 B ≈ 26 MB; small here, but scales linearly with problem dim);
  per-iteration host fetch of `res`, `Λ` only. Eigenvectors returned as
  jax.Array with sharding intact.
- `davidson_absorption`: `W_R = ifft₃(W_q)` materialised once on device
  under `sh.W` (125–126); final eigvecs host-gathered on all ranks
  (186–187) for the numpy dipole einsum; `.dat` written by rank0 only.

## TDA vs full BSE

TDA-only, structurally: the matvecs this family diagonalises
(`bse_simple`/`bse_ring_comm`) implement only the resonant block
`H = D + V − W`, and `solvers.davidson` *assumes Hermitian H* — it
symmetrizes `Hc` (davidson.py:101) and solves with `eigh`. The full-BSE
(coupling-block) effective Hamiltonian is non-Hermitian; there is no path
here. (`bse_feast.build_preconditioner_diagonal_sharded` has a
`use_tda=False` branch stacking `[diag_h, −diag_h]`, but that serves the
FEAST contour solver, not Davidson.)

## Spin / nspinor

- ψ arrays carry an explicit spinor axis: `(nk, nb, nspinor, μ)`; all pair
  amplitudes are spin-traced (`Σ_s`, e.g. `"kcsN,kvsN->kcvN"`), so shapes work
  for nspinor 1 or 2 alike. No nspin=2 (collinear) handling anywhere.
- `davidson_absorption` hardcodes `n_spin=1, n_spinor=2` into the
  `eigenvalues_*.dat` header (line 210) — see Suspects; the header is consumed
  numerically by `eigvals_to_eps2.py:48,92` (prefactor
  `16π²/(V·ns·nspinor)`).
- `--n-val` help text says "Kramers pairs (SOC)" — but the code slices
  individual spinor bands (see Suspects).

## Coupling to gw/ and isdf/

- **gw/**: via `bse_io.load_bse_data_from_restart_sharded` — reads the gw_jax
  restart h5 (`V_qmunu`, `W0_qmunu` gated on `W0_ready`, `psi_full_y`,
  `enk_full`, `G0_mu_nu`, `vhead`/`whead`) and injects the q=0 head with
  `gw.head_correction.apply_q0_head_rank1_sharded` (bse_io.py:509-513).
  `common.fft_helpers.make_sharded_ifftn_3d` is the same helper gw_jax uses.
- **isdf/**: no direct import; the ISDF coupling is entirely through the
  restart tensors (μ,ν are ISDF centroid indices fitted upstream).
- **psp/**: `solvers.davidson` is shared infra — `psp/run_nscf.py:46` drives
  the plane-wave NSCF with it (`trailing = (nspinor, ngkmax)`), with
  `psp/dft_precond.py` providing `precond_fn`/`init_fn`. Any change to the
  solver's contract must be checked against BOTH consumers.

## Suspects

### Dead

- `bse_preconditioner.py` minus `energy_diff_cv_k`: `BSEPreconditionerTerms`
  (22–26), `_pair_amplitude` (44–46), `compute_v_diagonal` (49–72),
  `compute_w_diagonal` (75–100), `extract_w_q0` (103–105),
  `build_preconditioner_terms` (108–127), `build_shifted_preconditioner`
  (130–158): zero callers by repo-wide + sandbox grep. Superseded by the
  sharded re-implementation `bse_feast.build_preconditioner_diagonal_sharded`
  (bse_feast.py:75-117; identical math, verified per-element for V) plus the
  inline `1/(z − diag_h)` in `bse_feast._get_gmres_solver` (bse_feast.py:133-135).
  Only reference is a "see …build_preconditioner_terms if more accuracy is
  needed" docstring pointer (bse_davidson_helpers.py:156) — which points at
  the buggy dead copy instead of the live sharded one.

### Bugs

- **`compute_w_diagonal` einsum contracts ρ_v on the wrong index**
  (bse_preconditioner.py:99). Code:
  `jnp.einsum("kcm,mn,kvm->cvk", rho_c, W_q0, rho_v)` — per element
  `out[c,v,k] = Σ_m Σ_n ρ_c[k,c,m]·W[m,n]·ρ_v[k,v,m]`
  `= Σ_m ρ_c[k,c,m]·ρ_v[k,v,m]·(Σ_n W[m,n])`,
  i.e. both densities land on μ and W is summed over ν unweighted. Docstring
  (line 83) and physics require `Σ_{μν} ρ_c[μ] W[μ,ν] ρ_v[ν]` — ρ_v must be
  subscript `kvn`. The live bse_feast version does it right
  (`"MN,kvN->kvM"` then `"kcm,kvm->kcv"`, bse_feast.py:101-102). Currently
  unreachable (dead code), but any resurrection of
  `build_preconditioner_terms`/`build_shifted_preconditioner` inherits an
  O(1)-wrong W diagonal (W_q0 is dense in μν, so the two expressions differ
  at leading order).
- **`davidson_absorption` dipole/eigenvector valence-axis misalignment under
  band padding** (davidson_absorption.py:200-202). It calls
  `slice_dipole_to_bse_window(..., n_val=nv_pad, n_cond=nc_pad)`, which
  slices dipole valence bands `n_occ−nv_pad … n_occ−1` (absorption_common.py:97-99),
  so dipole slot v=j ↔ band `n_occ−nv_pad+j`. The eigenvector's valence axis
  came from `val_idx = arange(n_occ−n_val, n_occ)` **padded at the END**
  (bse_io.py:139-146 pads `(0, pad)`; bse_io.py:450-452), so eigvec slot
  v=j ↔ band `n_occ−n_val+j` for j < n_val. Whenever `grid_y ∤ n_val`
  (nv_pad > n_val), `proj[S,a] = Σ A*_S d_a` pairs every amplitude with the
  dipole of a band shifted by `nv_pad − n_val` — e.g. 2×2 mesh with
  `--n-val 5` → nv_pad=6 → all oscillator strengths use the wrong valence
  band (eigenvalues remain correct). The correct slice-then-pad helper
  already exists and is used by the Haydock route
  (`build_dipole_vector_bse`, absorption_common.py:105-116;
  absorption_haydock.py:198). Validated runs used n_val=8 on 1×1/2×2 meshes
  (divisible), so this has never fired.
- **Hardcoded `n_spinor=2` in the `.dat` header** (davidson_absorption.py:210:
  `write_eigenvalues_dat(..., n_spin=1, n_spinor=2, ...)`).
  `eigvals_to_eps2.read_bgw_eigvals` parses that header line
  (eigvals_to_eps2.py:48) and feeds `pref = 16π²/(V·ns·nspinor)`
  (eigvals_to_eps2.py:92), so for an nspinor=1 (non-SOC) system the derived
  ε₂ is exactly 2× too small. The module was written against the Si SOC
  setup; nothing plumbs the actual WFN nspinor through.
- **Zero-padded bands are exact eigenpairs of H and only conditionally
  excluded** (bse_davidson_helpers.py:112 `finite = … & (flat > 1e-12)`).
  Padding is `mode="constant"` zeros for both ψ and ε (bse_io.py:139-146,
  450-453; same in the EQP branches, davidson_absorption.py:104-107). A unit
  vector at a padded slot has V- and W-terms exactly zero (ψ=0 ⇒ M=0 ⇒
  S=T=0), so it is an exact eigenvector with eigenvalue
  `λ_pad = ε_c[k,c] − 0` (padded v) or `0 − ε_v[k,v]` (padded c). The
  `>1e-12` filter keeps such slots out of the *initial guess* only under the
  QE positive-reference assumption (valence energies > 0 ⇒ padded-c ΔE < 0;
  padded-v ΔE = ε_c, usually above the exciton window). Nothing keeps them
  out of the *converged spectrum*: on a system whose energy reference makes
  any ε_c[k,c] (or −ε_v) fall below/inside the requested window and a mesh
  where `grid_x ∤ n_cond` or `grid_y ∤ n_val`, Davidson dutifully returns the
  spurious padded eigenvalues among the lowest n_eig. A large-sentinel pad
  for ε (or masking ΔE at padded slots in the D term) would close this.

### Redundancy

- `_gather_to_host`/`_to_host` exists in **four** copies:
  `solvers/davidson.py:45-64`, `bse_davidson_helpers.py:41-58`,
  `davidson_absorption.py:43-53`, and the original
  `file_io/_slab_io_allgather.py:36` they all cite. One source of truth,
  three verbatim clones.
- `ΔE[c,v,k] = ε_c[k,c] − ε_v[k,v]` is computed at three independent sites:
  `bse_preconditioner.energy_diff_cv_k:41` (the canonical one),
  `bse_davidson_helpers.py:185` (inline reimplementation inside
  `bse_diagonal_precond._impl` — same `eps.T` broadcast, could just call the
  helper), `bse_simple.py:82` (4D batched variant).
- Two divergent EQP-override blocks: `davidson_absorption.py:93-111` uses
  `resolve_n_occ` (WFN ifmax); `test_davidson_bse.py:76-93` uses the
  `mean_enk < 0` heuristic that `resolve_n_occ`'s docstring (bse_io.py:620-624)
  explicitly calls "silently broken" (unreachable by default there since
  `--n-occ` defaults to 4, but a copy-paste trap). A third near-copy lives at
  `bse_jax.py:241-262`. All three re-slice + re-pad eps by hand.

### Weird / conventions (not bugs)

- `--n-val` help text (davidson_absorption.py:59-60): "Kramers pairs (SOC) or
  SP bands (non-SOC). For Si SOC 8x8: --n-val 4" — contradicts the actual
  slicing, which counts individual spinor bands
  (`val_indices = arange(n_occ − n_val, n_occ)`, bse_io.py:433); on an SOC
  system `--n-val 4` yields a 4×4 spinor-band window (2 Kramers pairs), not
  8×8. The module's own usage example (line 15) propagates this. The 8×8
  production run evidently used `--n-val 8`.
- BGW-compat boundaries (per `bse/STATUS.md` "Index ordering"): everything in
  this family is LORRAX-internal — v=0 = deepest valence in the window,
  energies in Ry. BGW conventions enter only at the edges:
  `write_eigenvalues_dat` output in eV (`ryd2ev` at davidson_absorption.py:192),
  and `test_davidson_bse._load_bgw_eigvecs` flips the valence axis on read
  (test_davidson_bse.py:158). No valence flip is needed (or done) in the
  Davidson internals — correct.
- `davidson`'s verbose print uses `float(Lambda[0])` on the device array
  (davidson.py:323) two lines after carefully `_to_host`-ing Λ for the
  convergence check — works for fully-replicated arrays but contradicts the
  module's own multi-process caution; redundant device sync.
- `tol` is a *relative* tolerance (`‖R‖ < tol·max(1,|λ|)`, davidson.py:311)
  and convergence counts only the leading contiguous run of states (313–318)
  — a converged state above an unconverged one doesn't count.
- `davidson` returns `eigenvalues=None` if `max_iter=0`; `n_conv` in the
  final WARNING (line 351) is stale from the last iteration.
- `warmup_davidson_jit` warms `m = m_max` twice when `n_eig ∤ m_max`
  (`m_eff = min(m, m_max)` clamp, line 213-214) — harmless duplicate compile
  key.
- `davidson_absorption.py:199` passes the windowed, padded `eps_v` as
  `resolve_n_occ`'s `enk_full` argument — that arg is only consulted with a
  `fermi_energy` hint (bse_io.py:654-659), never given here, so it's inert
  but misleading (reads as if n_occ were derivable from a window that no
  longer contains the Fermi level).
