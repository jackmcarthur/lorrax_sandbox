# src/common/symmetry_maps.py — deep-read notes (gw_refactor_map_2026-07-01)

LOC: 1429. Category: **symmetry machinery** (BZ/IBZ tables + IBZ→full-BZ unfold kernels for ψ, V_q/W_q, bispinor TT tiles).

Header note (lines 1–8): the wfn-unfold pipeline (`get_gvecs_kfull`, `get_cnk_fullzone[_batch]`) was moved to `file_io.wfn_loader.WfnLoader` in "P5". This module retains sym-table construction (`kpoint_map`, `R_grid`, `unfolded_kpts`) plus kfull-symmap / q-IBZ helpers for the GW driver.

No LorraxConfig flags / cohsex.in keys are consumed anywhere in this file. No file I/O — pure in-memory; reads WFN metadata off a `WfnLoader` object (`kpoints, kgrid, shift, ntran, sym_matrices, translations, avec, nkpts, atom_crys, atom_types`).

## Module-level functions

### `find_irreducible_bz_points(full_kgrid_int, sym_mats_k, *, irr_kgrid_int=None)` — lines 19–108
IBZ reduction over integer-kgrid points. Two branches:
- `irr_kgrid_int=None` (q-side): derive IBZ as lex-smallest orbit representatives. `images = np.einsum('sij,qj->sqi', Smk, full) % kg` (VERBATIM einsum), keys via `((q0*kg1+q1)*kg2+q2)`.
- `irr_kgrid_int` given (k-side, anchored to `wfn.kpoints`): explicit double loop; comment at 70–72 says the outer iteration over `ikbar` **without break ⇒ HIGHEST ikbar with any match wins**, deliberately matching `find_symmetry_ops_simple` for bit-equality with prior code.

Returns `(irr_idx, sym_idx, irr_kgrid_int_out)` all int32. `sym_idx` values may be TRS-augmented (`>= ntran`).
Callers: `SymMaps.__init__` (line 968, q-side); `tests/test_q_ibz_and_centroid_perm.py:16,32`.

### `slice_q_full_to_ibz(arr_full, q_irr_full_idx, *, out_sharding=None)` — lines 111–158
Pure row gather `(n_q_full, …) → (n_q_ibz, …)` along axis 0; optional `with_sharding_constraint` (typically `P(None,'x','y')`) to stabilize the JIT cache key. Companion of `unfold_v_q` in the other direction. No phase, no permute, no conj.
Callers: `src/gw/gw_jax.py:251,253` (slice V_q and chi0 to IBZ before the W-solve `(1 − v χ)⁻¹ v`); `src/common/isdf_fitting.py:2126` (slice C_q before `factor_c_q` Cholesky/LU).

### `unfold_v_q(V_q_ibz, *, irr_idx, sym_idx, sym_perm, L_table, q_irr_frac, mesh_xy, n_sym_spatial)` — lines 161–333
IBZ→full-BZ unfold of a (n_q, n_rmu, n_rmu) ISDF-metric object (V_q or W_q). Physics (docstring, lines 178–180):

    V_full[q, μ', ν'] = exp(2π i q_irr · (L_{s,μ'} − L_{s,ν'})) · V_ibz[i(q), α_s(μ'), α_s(ν')]

with `i(q)=irr_idx[q]`, `s(q)=sym_idx[q]`, `α_s(μ)=sym_perm[s,μ]` (FORWARD source-map from `compute_centroid_sym_perm`), `L_{s,μ}=L_table[s,μ]` the real-space lattice wrap of `y_μ = mtrx·(x_μ − τ) = x_{α(μ)} + L_μ`. TRS rows (`sym_idx >= n_sym_spatial`): `V_full[TRS-q, π_s(μ), π_s(ν)] = conj(V_ibz[i(q), μ, ν])` (lines 199–201).

Notable structure:
- Trivial-IBZ short-circuit (244–255): if irr_idx is identity and sym_idx all zero, return input unchanged — explicitly a **workaround for an XLA HLO verifier dtype mismatch** (s64 broadcast vs s32 operand on 2×2 mesh) in the sharded take_along_axis path.
- Explicit ValueError if `sym_idx` exceeds `sym_perm` rows (259–271) — this is the guard added after the TRS-blind silent-OOB-clip bug (MEMORY: trs-blind-sym-bug).
- Comment 273–281 records the history: prior code used `inv_perm = argsort(sym_perm)` — no-op for involutive ops (MoS2, Si cubic) but wrong for order-3 CrI3 C3 ("the silent 4 eV gap on hex systems"). Forward permutation verified vs `reports/trs_sym_audit_2026-05-14/test_production_unfold_v_q.py`.
- μ-axis padding: identity rows appended to `fwd_perm`, zero rows to `L_table`, to match `V_q_ibz`'s padded μ-extent (283–310).
Callers: `src/gw/gw_jax.py:278` (W_q IBZ→full after solve), `src/gw/v_q_g_flat.py:451` (V_acc unfold in the flat-G V_q pipeline), `src/gw/compute_vcoul.py:1092`.

Key arrays: `V_q_ibz` (n_q_ibz, n_rmu_pad, n_rmu_pad) complex128 device, sharded `P(None,'x','y')`; sym tables all small host numpy, baked into jit closure as constants.

### `_get_unfold_v_q_jit(...)` — lines 339–473 (+ `_UNFOLD_V_Q_JIT_CACHE` at 336)
Content-keyed jit cache (key = shape + `tobytes()` of every table + `id(mesh_xy)`). Docstring claims runtime-arg form was ~2× slower per call than closure-baked constants; V_q and W_q share one compiled module when tables match.

Inner `_do_unfold` / `_kernel` (392–470): `shard_map` over mesh `('x','y')`, in/out specs `P(None,'x','y')`, `check_rep=False`. Memory contract (comment 374–380): never exceed 1× single tile per rank; uses `lax.all_to_all` pairs to permute μ (on 'x') then ν (on 'y') volume-preservingly. Requires `n_rmu_padded % (Px·Py) == 0` (raises otherwise, 384–389). Gathers use `jnp.take_along_axis(..., mode='promise_in_bounds')`. Umklapp phase: `qL = jnp.einsum('qi,qmi->qm', q_per_q, L_per_q)` (VERBATIM, line 449), `phase = exp(2πi·qL)`, applied as `phase_mu[:, :, None] * V * conj(phase_nu)[:, None, :]` with per-rank `dynamic_slice_in_dim` via `lax.axis_index`. TRS conj applied via `jnp.where(trs_mask, conj(V), V)`.

### `unfold_v_q_bispinor_lorentz(V_tt_per_channel, *, sym_idx, R_proper_table, mesh_xy)` — lines 476–566
3-vector Lorentz mixing of the bispinor TT block (μ_L, ν_L ∈ {1,2,3}²) AFTER scalar `unfold_v_q`. Rule (derivation `reports/bispinor_ibz_2026-05-16/derivation.md` §A5):

    V^{i,j}_mixed[q,μ,ν] = Σ_{α,β} R_proper[s(q), i-1, α-1] · R_proper[s(q), j-1, β-1] · V^{α,β}_unfolded[q,μ,ν]

TRS rows reuse the spatial R_proper: the σ-flip sign factorizes as (−1)·(−1)=+1 per stored tile, absorbed by the scalar conj-wrap (§A4). Input: dict of 9 tiles (i,j) ∈ {1,2,3}², each (n_q_full, μ, ν) c128 `P(None,'x','y')`; caller synthesizes Hermitian-redundant tiles via `conj(swapaxes(V[i,j],-1,-2))`. Raises if any tile missing or R table shape wrong.
Caller: `src/gw/v_q_bispinor.py:704` (only).
NOTE dead local: line 548 `R_dev = jnp.asarray(R_per_q)` is assigned and never used (the jit builder re-does `jnp.asarray` at line 589).

### `_get_unfold_v_q_lorentz_jit(*, V_shape, R_per_q_arr, mesh_xy)` — lines 569–611 (+ cache at 569)
Same content-keyed cache pattern. Contraction (VERBATIM, lines 605–608):

    jnp.einsum('qai,qbj,abqmn->ijqmn', R_per_q_j, R_per_q_j, V_in)

Comment 595–604 documents a convention subtlety: live `R_per_q` is `R_LORRAX = A.T·mtrx·inv(A.T)` (with det-flip), which is the **transpose** of the derivation's `R_deriv = A.T·inv(mtrx)·inv(A.T)` for orthogonal mtrx, hence `qai` (col index i) rather than `qia`. In/out shardings `P(None,None,None,'x','y')`.

### `_I_SIGMA_Y` — line 615
`np.array([[0,1],[-1,0]])` c128, the iσ_y factor in the TRS convention T = iσ_y K.

### `unfold_psi(cnk_kbar, *, sym_idx, g_kbar, sym_mats_k, translations, U_spinor_spatial)` — lines 618–726
Host-side (pure numpy) ψ unfold at one full-BZ k from its IBZ rep. The bispinor TRS rule lives HERE and only here (PR3+). Math (docstring 632–644):
- spatial: `ψ_full(G_rot) = exp(-i (S·G_kbar)·τ) · U_spinor(S) · ψ_kbar(G_kbar)`
- TRS: `ψ_full = (iσ_y · conj(U_spinor(S))) · exp(+i (S·G_kbar)·τ) · conj(ψ_kbar)`

Implementation trick: `sym_mats_k[TRS row] = -S`, so computing the phase from the TRS row automatically yields `conj(phase_spatial)` — no sign branch; conj applied to cnk BEFORE the phase multiply (order matters, comment 708–713). `has_tau` skip when τ≈0. Spinor rotation applied as `np.einsum("jk,nkl->njl", U_eff, cnk)` (VERBATIM, line 725); no-op for ns=1.
Arrays: `cnk_kbar` (nb, ns, ngk) complex host; `g_kbar` (ngk,3) int; returns (nb, ns, ngk) on the IBZ G-axis (caller's WfnLoader handles G-rebuild/umklapp).
Callers: `src/file_io/wfn_loader.py:1012` (the single production consumer); `tests/test_unfold_psi_trs.py:128,166,189,200` (also imports `_I_SIGMA_Y`).
Docstring quirk: documents an `n_sym_spatial` parameter (662–664) that is NOT in the signature — it is derived internally as `sym_mats_k.shape[0] // 2` (line 688). Stale docstring.

## class SymMaps

### `__init__(self, wfn)` — lines 730–978
Builds all sym tables from a WFNReader. Attributes:
- `sym_matrices` (ntran,3,3) int = BGW `mtrx` (acts on G columns: G' = mtrx@G); `sym_mats_k` = `mtrx.T` per row, then TRS-augmented: `concatenate([sym_mats_k, -sym_mats_k])` → (2·ntran,3,3) (lines 842–856).
- `translations` (ntran,3) f64 = BGW `tnp[:ntran]` (legacy files pad to 48).
- `kpoint_map`, `unfolded_kpts` via `create_kpoint_symmetry_map` (validated in [0, nkpts)); `irr_idx_k`, `sym_idx_k` via `find_symmetry_ops_simple`.
- `nk_tot`, `nk_red`; `kirr_fullids` (nk_red,) = first full-BZ index of each IBZ k, with silent identity **fallback** if an IBZ k has no full-BZ match (lines 874–881).
- `R_grid` = rint(sym_matrices); `Rinv_grid` = rint(inv); `R_cart` via `syms_crystal_to_cartesian` (2·ntran rows); `U_spinor` = `get_spinor_rotations(R_cart[:ntran])` — **spatial half only**; comment 893–901: before 2026-05-14 the array was 2·ntran long with the TRS half computed wrong via `get_spinor_rotations(-S)`'s det<0 flip; restricting to ntran "makes the bug unreachable by construction" (trs_sym_audit Site #6).
- `R_proper` (2·ntran,3,3) f64: proper (det=+1) part of R_cart spatial half, duplicated for the TRS half (932–937). Long comment (904–931) documents: (a) it satisfies `U_spinor† σ^i U_spinor = Σ_j R_proper^{j,i} σ^j`; (b) it differs from the offline fixture `reports/bispinor_ibz_2026-05-16/cri3_R_proper.npz` by a transpose on every row (fixture follows derivation text, live code follows the σ-sandwich identity).
- `kq_map` (nk_full, nk_red) via `get_kminusq_map`; `kqfull_map` (nk_full, nk_full) via `get_kminusqfull_map`.
- `kvecs_asints` (nk_full,3) integer kgrid coords (meshgrid order, NOT wfn shift-aware in the full branch — cf. trivial branch which subtracts shift_frac, lines 787–794).
- `all_unfolded_qpts` / `all_unfolded_qpt_ids`: unique k'−k integer differences (can be outside 1BZ) + per-(k,k') id; full-branch built via per-q boolean-mask loop `(qpt_vecs == q).all(axis=2)` (959–961) = O(n_q · nk²), while the trivial branch uses `np.unique(..., return_inverse=True)` — two parallel implementations.
- q-IBZ block (964–978): `irr_idx_q, sym_idx_q, q_irr_kgrid_int = find_irreducible_bz_points(kvecs_asints, sym_mats_k)`; `q_irr_full_idx` = sorted first-occurrence rows. Comment notes `is_trs = sym_idx_q >= ntran` is implicit, not stored.

**Trivial branch** (ntran ≤ 1, lines 755–834): fully parallel construction of every attribute with different (direct modular-arithmetic) algorithms, including a `lookup` cube indexed by integer coords, `kqfull_map` via mod arithmetic, `kq_map = kqfull_map.copy()`, identity q-IBZ. ~80 lines duplicating the general path's outputs.

### `get_qpt_id_from_kkp(self, kidx, kpidx)` — lines 981–984
Linear search of `all_unfolded_qpts` for k'−k. **Zero callers** (grep `get_qpt_id_from_kkp` across src/tests/tools/scripts: only the definition). DEAD.

### `_wrap_to_bz(kpts)` (988–993), `_periodic_delta(points, target)` (995–999), `_generate_uniform_full_kpoints(wfn)` (1001–1012)
Static/internal helpers. `_wrap_to_bz` snaps `wrapped > 0.99999` → 0.0 (magic threshold; `_get_umklapp_vector` uses a different threshold 0.9999 at line 1394).

### `create_kpoint_symmetry_map(self, wfn)` — lines 1014–1065
Triple loop (nk_full × nsym × nk_irr) matching `sym_mat @ k_full` to wrapped `wfn.kpoints` at atol 1e-6. On failure: **nearest-IBZ identity fallback with only a warning** (1050–1063) — masks incomplete WFN symmetry data instead of raising.

### `find_symmetry_ops_simple(self, wfn, kpoint_map, full_kpts)` — lines 1067–1090
Returns (irr_idx_k, sym_idx_k). `kpoint_map` arg immediately `del`-ed ("kept in signature for compatibility"). Einsum `'ijk,lk->lij'` (VERBATIM, line 1073) applies all syms to IBZ k. **No break** in the ikbar loop ⇒ highest matching ikbar wins (same quirk `find_irreducible_bz_points` deliberately replicates). No error if some ikfull never matches (silently keeps index 0).

### `validate_atomic_symmetries(self, wfn, tol=1e-6)` — lines 1092–1123
Checks each spatial sym maps the atom basis onto itself; uses `rot = inv(mtrx)` and `tau = wfn.translations[s] / (2π)` (line 1100 — translations here are treated as radians/2π-scaled, whereas `unfold_psi` uses `translations` directly in `exp(-i G·τ)`; the two conventions are consistent only if wfn.translations is stored in 2π·fractional units, worth confirming during refactor). Greedy same-species matching with `available.remove`. Callers: `src/centroid/orbit_syms.py:48`, `src/common/symmetry_test.py:21`.

### `validate_kgrid_unfolding(self, wfn, tol=1e-6)` — lines 1125–1154
Checks S·k_irr + kg0 reproduces every full k. Caller: `src/common/symmetry_test.py:22` only (debug utility).

### `syms_crystal_to_cartesian(self, wfn)` — lines 1156–1218
`R_cart = A.T @ mtrx @ inv(A.T)` via `np.einsum('ij,njk,kl->nil', A_T, mtrx, A_T_inv)` (VERBATIM, line 1214), rounded to 10 decimals, then `concatenate([R_spatial, -R_spatial])`. Long docstring documents the 2026-05-14 fix: pre-fix used `sym_mats_k` (= mtrx.T) → wrong U_spinor whose error CANCELLED for involutive/cubic groups but gave 6 eV (CrI3) / 160 eV (Si) Σ_X failures; the original smoking-gun comment was "# NOT SURE IF THESE SHOULD BE SYM_MATS_K OR SYM_MATS TODO". Referenced by `tests/test_R_proper_cri3.py:115`.

### `get_spinor_rotations(self, wfn, sym_matrices_cart)` — lines 1220–1304
Markley/Shepperd quaternion extraction → SU(2): det<0 ⇒ R→−R (improper→proper, matches BGW `Common/spinor_symmetries.f90`); build symmetric 4×4 Q, largest-eigenvalue eigenvector = quaternion, θ=2·arccos(q0), spinor = cos(θ/2)·I − i·sin(θ/2)·(n·σ). `wfn` arg unused in the body. Docstring anticipates future 4×4 bispinor rotations. Only internal caller (`__init__` line 903).

### `get_kminusq_map(self, wfn, full_kpts)` — lines 1306–1349
`kq_map[ik_full, iq_red]` = index of k−q in full grid; O(nk_full·nk_red·nk_full) periodic nearest-match with 1e-4 tolerance, raises on miss. Consumers via the attribute: `src/common/kq_mapping.py:51` (`kminq_idx_for_iq`, the canonical accessor), `src/psp/tests/test_sternheimer_jvp.py:110`.

### `get_kminusqfull_map(self, wfn, full_kpts)` — lines 1351–1381
Copy-paste of `get_kminusq_map` with iq over the FULL grid. The resulting `kqfull_map` attribute has **zero external readers** (grep `kqfull_map` outside this file: no hits). DEAD + redundant.

### `_get_umklapp_vector(self, wfn, nk, sym_idx, kbar_idx, sym_krep)` — lines 1383–1406
BGW kg0: `k_full = S·k_irred + kg0` (matches Common/gmap.f90). TRS branch (sym_idx ≥ ntran) wraps S·kbar into zone with threshold 0.9999 and returns `q_inzone − q_full` instead of rint against unfolded_kpts. Callers: `src/file_io/wfn_loader.py:377`, `validate_kgrid_unfolding` (internal); referenced in `src/file_io/read_bgw_vcoul.py:37` docs.

### `find_qpoint_index(self, q_ext, tol=1e-6)` — lines 1408–1428
Nearest-match of a fractional q in `unfolded_kpts`; oddly uses `jnp` (jnp.abs/sum/min/argmin) for a host scalar lookup and returns a traced-typed index. Caller: `src/gw/vcoul.py:166`.

## Consumers of SymMaps as an object (attribute-level)
`SymMaps(wfn)` constructed in: `gw/gw_jax.py:137` (main GW driver), `gw/kin_ion_io.py:101`, `centroid/kmeans_cli.py:214`, `centroid/charge_density.py`, `centroid/pivoted_cholesky.py`, `bandstructure/htransform.py:430`, `bse/bse_io.py:715`, `psp/run_sternheimer.py:1095`, `common/symmetry_test.py:19`, and lazily inside `file_io/wfn_loader.py` / `common/psi_G_store.py`. Heaviest attribute consumers (grep counts): `file_io/wfn_loader.py` (23 hits), `psp/run_sternheimer.py`, `psp/get_DFT_mtxels.py`, `common/isdf_fitting.py`, `gw/v_q_g_flat.py`, `gw/compute_vcoul.py`, `file_io/zeta_loader.py`, `centroid/orbit_syms.py`.

## Dead suspects
| item | evidence |
|---|---|
| `SymMaps.get_qpt_id_from_kkp` (981) | grep `get_qpt_id_from_kkp` over src/tests/tools/scripts → only the definition |
| `SymMaps.get_kminusqfull_map` / `kqfull_map` attr (1351) | grep `kqfull_map` over src/tests/tools/scripts → zero hits outside this file |
| `all_unfolded_qpts` / `all_unfolded_qpt_ids` attrs (950–961, 817–826) | grep `all_unfolded_qpt` → zero external hits; only internal user is the dead `get_qpt_id_from_kkp` |
| `validate_kgrid_unfolding` (1125) | single caller `src/common/symmetry_test.py` (standalone debug utility) |
| local `R_dev` (548) | assigned in `unfold_v_q_bispinor_lorentz`, never read |

## Redundancy suspects
- `get_kminusq_map` vs `get_kminusqfull_map` (1306/1351): line-for-line copy-paste differing only in the q list; the latter is also dead.
- Trivial ntran≤1 branch of `__init__` (755–834) re-implements every table with a second algorithm (modular lookup cube) in parallel to the general path.
- `find_irreducible_bz_points` anchored branch (66–106) vs `find_symmetry_ops_simple` (1067–1090): two implementations of "match IBZ point + sym op onto full grid", the former written to preserve bit-equality with the latter's no-break quirk.
- `unfold_v_q` here vs `gw/v_q_tile.py:_unfold_v_q_ij_ibz_to_full` (and the referenced `_unfold_v_q_ibz_to_full` in `gw/v_q_g_flat.py` comments): parallel IBZ→full unfold kernels living in gw/ that mirror this module's logic (v_q_tile.py comments repeatedly cite "same logic as `_unfold_v_q_ibz_to_full`", "mirroring _unfold_v_q_ibz_to_full"). Direct target for the MEMORY "unified sym action" consolidation.
- `all_unfolded_qpt_ids` built two ways (np.unique return_inverse in trivial branch, O(n_q·nk²) mask loop in full branch).

## Weird code
- 70–72 / 1082–1084: deliberate no-break ⇒ HIGHEST ikbar wins, preserved "for bit-equality with prior code".
- 244–255: trivial-IBZ short-circuit in `unfold_v_q` is a workaround for an XLA HLO verifier s32/s64 dtype mismatch on 2×2 meshes.
- 273–281: history note — prior `argsort(sym_perm)` inverse-permute was the silent 4 eV hex-system bug; forward permute is empirical, verified vs trs_sym_audit test.
- 918–925: live `R_proper` is the TRANSPOSE of the offline derivation fixture per row; compensated by the `'qai,qbj,...'` (column-index) einsum in the Lorentz mixer.
- 992 vs 1394: two different snap thresholds (0.99999 vs 0.9999) for zone-wrapping.
- 1050–1063: nearest-IBZ identity fallback in `create_kpoint_symmetry_map` only warns on unmatched k-points (masks bad WFN sym data).
- 874–881: `kirr_fullids` silent identity fallback if an IBZ k never appears in irr_idx_k.
- 1100: `tau = wfn.translations / (2π)` in atom validation, while `unfold_psi` (700–704) uses `translations` un-scaled in the G·τ phase — unit convention split across the two consumers of the same array.
- 662–664: `unfold_psi` docstring documents an `n_sym_spatial` parameter absent from the signature (derived internally).
- 1408–1428: `find_qpoint_index` uses jnp ops for a host scalar search.
- 312–318, 343–349: jit tables closure-baked as constants with `tobytes()` cache keys; keyed also on `id(mesh_xy)` — a re-created (but equal) mesh silently recompiles.
