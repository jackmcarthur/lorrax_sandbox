# Team 3 — ζ-fit numerics + sharding audit (`agent/env-var-cleanup`, tip `488e870`)

Scope: the ζ-fit machinery itself (solver, monolithic `shard_map`, einsum spec,
trace hoist, μ-sharded landing, Lorentz metric stabilization, gamma-contract
variants, inner-jit drops, dispatch tightening). Per-element IBZ unfold of ζ
and V_q construction are out of scope.

Audited tree: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.
Hot file: `src/common/isdf_fitting.py`.

---

## 1. Executive verdict

The audited stack of commits preserves ζ-fit math elementwise, with the
following caveats:

- Solver dispatch is **correct in normal (`solver_kind='auto'`) usage**:
  charge → Cholesky family; transverse (`vertex_mu_L ∈ {1,2,3}`) → LU family.
  An explicit override of `solver_kind='cusolvermp_cholesky'` with
  `vertex_mu_L != 0` would still silently route an indefinite C_q to
  Cholesky (LOW-severity hazard; production caller uses `'auto'`).
- The monolithic `shard_map` (5a31803, 192c31c, d775254) **drops all replicated
  intermediates** in the C_q / Z_q pipeline. Per-rank tensors stay local to
  the rank; the rank-5 pair density is never materialised as a global XLA
  value. Slot count drops from 5 → 3 as advertised.
- The `karmb` einsum (91ca4a0) is bit-identical to the old `kabmr` einsum
  modulo a static axis permutation; verified numerically.
- `cct_trace_per_q` hoist (0abeb9a) is a pure no-op semantically. Trace is
  computed once per channel on the full-BZ `L_q`, sliced by `q_irr_full_idx`
  inside the kernel — exactly the elements the inline einsum would have
  produced.
- ζ producer–consumer agree on `P(None, ('x','y'), None)`; the μ-sharded
  landing fixes the unsharded FFT-box hazard called out in `b4ba9b8`.
- The "Lorentz metric stabilization" (1d0ba98) **was reverted** by the later
  `ce28d50` unification commit. Current HEAD applies γ̃ on both P_l and P_r,
  building an indefinite-by-construction CCT^μ and routing it to LU. This is
  the right semantics for the current Phase-3a contract but worth flagging.
- The `gamma_double_contract` mode variants (23fec11) are bit-identical
  under fp64 (JAX_ENABLE_X64=1). They diverge by O(1e-3) under fp32, but
  the production stack runs fp64 everywhere ζ-fit lives.
- Inner-`@jit` drops (6767a0d) safely removed compile-pollution; the
  remaining inner `@jit` in the legacy `solve_zeta` Z-reshard path
  (`isdf_fitting.py:1097`) is a minor still-extant instance of the same
  pattern. Doesn't affect correctness; small compile-cache impact.

No silent-wrong-answer defects found in the in-scope commits. Two cleanup
items + one open question listed in §10.

---

## 2. ζ-fit per-element formula

ψ_n,k(r) is decomposed onto centroid amplitudes ζ_μ(r) such that the pair
product fits as

    ψ_n,k(r_μ) ψ_m,k+q(r) ≈ Σ_ν ζ_ν(r) · ψ_n,k(r_μ) ψ_m,k+q(r_ν)

where the right side is built from on-centroid pair products. The least-
squares normal equation per q is

    C_q ζ_q = Z_q,    C_q[μ,ν] = (Π_q Π_q†)[μ,ν],    Z_q[μ,r] = (Π_q · pair)[μ,r]

where Π_q[μ; n,m,k] is the pair-density tensor at centroid μ. In the LORRAX
implementation the q-dependence comes out of an FFT_k chain:

    P_l_k[α,β,μ,r] = Σ_n ψ*_l,n,k[α,μ] · ψ_l,n,k[β,r]   (P_l: left bands)
    P_r_k[α,β,μ,r] = Σ_n ψ*_r,n,k[α,μ] · ψ_r,n,k[β,r]   (P_r: right bands)
    C_q[μ,ν]       = Σ_k IFFT(P_l)*_R · IFFT(P_r)_R    + γ̃ contract on (α,β)
    Z_q[μ,r]       = same expression with the second-axis being r_chunk

For the charge channel γ̃^0 = I_4, the (α,β) reduction collapses to a Frobenius
sum `Σ_{α,β} P_l*·P_r`. For transverse γ̃^i ∈ {γ̃^1, γ̃^2, γ̃^3} the reduction
becomes

    C_q^μ_L[μ,ν] = Σ_k Σ_{α,β,α',β'} γ̃^μ_L[α,α'] γ̃^μ_L[β,β'] · P_l*[α,β,k,μ,X] · P_r[α',β',k,X,ν]

(γ̃^i ⊗ γ̃^i has eigenvalues {+1, -1} from α^i alone, so the bilinear form is
Hermitian indefinite by construction.)

For numerical stability the transverse path adds the ridge
`ε · |tr(L)|/n_rmu · I` (ε = 1e-12) before LU, lifting TRS-paired near-zero
modes safely above LU's pivoting floor without perturbing well-conditioned
modes (`isdf_fitting.py:946-951`, `985-1004`).

---

## 3. Solver dispatch correctness

### 3.1 Dispatch routing (PD vs indefinite)

| caller                          | `vertex_mu_L`             | path                                                       |
|---------------------------------|---------------------------|------------------------------------------------------------|
| `factor_c_q` (line 705)         | `!= 0` → early return     | C_q passed unfactored to `solve_zeta`                       |
| `factor_c_q` (line 708)         | `== 0` only               | `_resolve_solver_kind_charge` → `cusolvermp_cholesky` or `sharded_cholesky` |
| `solve_zeta` (line 889)         | actual `vertex_mu_L`      | `_resolve_solver_kind` → charge/transverse resolver         |

`_resolve_solver_kind` is the single arbitration point:
- `vertex_mu_L != 0`  → `_resolve_solver_kind_transverse` → `cusolvermp_lu` (true 2D) or `lu`
- `vertex_mu_L == 0`  → `_resolve_solver_kind_charge`    → `cusolvermp_cholesky` (true 2D) or `sharded_cholesky`

The transverse branch returns ONLY `lu` / `cusolvermp_lu` — never any
Cholesky variant. The Cholesky branch in `solve_zeta` (line 891) cannot
trigger for `vertex_mu_L != 0` under `solver_kind='auto'`. ✓

### 3.2 PD/indefinite math correctness

- `vertex_mu_L = 0`: γ̃^0 = I_4 → C_q = Σ_k P_l*·P_r (Frobenius open-spin),
  a Gram-style matrix that is PSD when L=R or near-Hermitian when L⊆R.
  Cholesky valid.
- `vertex_mu_L ∈ {1,2,3}`: γ̃^i = α^i with eigenvalues ±1. The double
  contraction `γ̃^i ⊗ γ̃^i` is Hermitian but indefinite by tensor-product
  of two ±1 spectra. Cholesky would NaN; LU correct.

The dispatch matches the math.

### 3.3 Override hazard (LOW severity)

`_resolve_solver_kind` (lines 607-611):

```python
if solver_kind != 'auto':
    return solver_kind
```

If a caller passes `solver_kind='cusolvermp_cholesky'` explicitly AND
`vertex_mu_L != 0`, the override bypasses the vertex check and routes an
indefinite C_q to Cholesky. The production caller in `fit_zeta_to_h5`
(line 1693-1696) resolves once via `_resolve_solver_kind` and threads
`_resolved_solver_kind` through, so the override is gated by caller
discipline. No production path triggers this. Worth a one-line guard:

```python
if solver_kind != 'auto':
    if int(vertex_mu_L) != 0 and solver_kind in ('cusolvermp_cholesky', 'sharded_cholesky'):
        raise ValueError(...)
    return solver_kind
```

---

## 4. Monolithic `shard_map` intermediate inventory

Functions: `c_q_from_psi_sm._local` (lines 301-351), `z_q_from_psi_sm._local`
(lines 417-450). Inputs sharded as `P(None,'x',None,None)` (psi_l_X, psi_r_X)
and `P(None,None,None,'y')` (psi_l_Y, psi_r_Y). Outputs at
`P(None,'x','y')`. Inside the `shard_map` body all data is **per-rank-local
slabs** — no cross-rank sharding annotation, by `shard_map` semantics.

| intermediate | shape (per rank)                                  | replicated? | citation |
|--------------|---------------------------------------------------|-------------|-----------|
| `psi_l_X_`   | `(nk, n_rmu/p_x, nb_l, ns)`                       | no, x-sliced | isdf_fitting.py:296-302 |
| `psi_l_Y_`   | `(nk, nb_l, ns, n_col/p_y)`                       | no, y-sliced | isdf_fitting.py:297 |
| `psi_r_X_`, `psi_r_Y_` | analogous                                | no          | 297 |
| `P_l` (rank-5) | `(nk, ns_l, col/p_y, μ/p_x, ns_r)`              | no, per-rank | 318-319 |
| `P_l_3d`     | bitcast of `P_l` to `(nkx,nky,nkz, ns_l, col, μ, ns_r)` | no, alias | 324 |
| `P_l_R`      | IFFT of `P_l_3d`, same shape                       | no          | 326 |
| `P_l_R_conj` | conj of `P_l_R`, same shape                        | no          | 327 |
| `P_r`, `P_r_3d`, `P_r_R` | analogous                              | no          | 320-331 |
| `C_R`        | rank-5 `(kx,ky,kz, col/p_y, μ/p_x)` reduced over spin | no    | 335-342 |
| `C_q_3d`     | FFT of `C_R`, same rank-5                          | no          | 344 |
| Output       | reshape+transpose to `(nk, μ/p_x, col/p_y)`        | no, per `out_spec=P(None,'x','y')` | 349-351 |

Verdict: **zero replicated intermediates**. All buffers are slabs along
mesh axes 'x' and/or 'y'. The `del` calls between IFFT steps drop the
pre-IFFT buffers eagerly so XLA's BufferAssignment can alias the post-IFFT
output into the pre-IFFT slot. Commit `d775254` measured 3 concurrent
rank-5 slots peak (P_l_R_conj + P_r_R + scratch), matching the planner
default `pair_density_slots_charge = pair_density_slots_transverse = 3`
in `gflat_memory_model.py:244-245`. ✓

Comparison to legacy chain (`git show d775254^:src/common/isdf_fitting.py`,
not loaded here for brevity but documented in `d775254` commit body):

- Legacy `c_q_from_pair` was a sequence of three jits (pair_density →
  ifft_conj/contract → fft); each materialised the rank-5 pair density
  globally between calls. 5 concurrent rank-5 slots in XLA's
  BufferAssignment (P_l + P_r + P_l_R + P_r_R + scratch).
- Monolithic shard_map fuses all four ops inside ONE shard_map body, so
  XLA scopes the lifetimes to the body and can alias. 3 concurrent slots.

Math elementwise identical: legacy did `pair_density(ψ_X, ψ_Y) → IFFT_k →
γ̃ contract → FFT_k`; monolithic does the same chain but inlined. No
operation reordering. The only difference is XLA's BufferAssignment scope.

---

## 5. `karmb` vs `kabmr` einsum — explicit per-element formulas

Old (pre-91ca4a0):

    P_old[k, a, b, m, r] = Σ_n psi_X[k, m, n, a] · psi_Y[k, n, b, r]      ('kmna,knbr->kabmr')

New (91ca4a0):

    P_new[k, a, r, m, b] = Σ_n psi_X[k, m, n, a] · psi_Y[k, n, b, r]      ('kmna,knbr->karmb')

Per-element equality: `P_new[k, a, r, m, b] == P_old[k, a, b, m, r]`. Sum
expression identical; only the output axis ordering differs. Verified
numerically (random c128, 0.0 elementwise diff after axis permutation —
`output[k,a,r,m,b] == transpose(output_kabmr, (0,1,4,3,2))[k,a,r,m,b]`).

Downstream uses match the new ordering:
- Reshape `(nk, ns, col, μ, ns) → (kx, ky, kz, ns_l, col, μ, ns_r)` is a
  pure bitcast under the new spec (consecutive memory layout). The old
  spec required a transpose copy of the rank-5 (~4 GiB at MoS2 3×3
  bispinor scale) to feed the IFFT.
- `gamma_double_contract` is invoked with `spin_axes=(3, 6)` matching the
  new layout where ns_l is axis 3 and ns_r is axis 6 of the rank-7 form.
  In `_gamma_double_contract_take` (gamma_matrices.py:147-160), the two
  `gamma_apply` calls operate on axes (a_axis, b_axis) = (3, 6); each
  applies γ̃ as `Y[..., β, ...] = phase[β]·X[..., perm[β], ...]` along the
  given axis. Reducing axes (3, 6) via `jnp.sum(P_l*·P_r_p, axis=(3,6))`
  contracts ns_l and ns_r — exactly the original two spin axes.
- The final `jnp.transpose(C_q_3d.reshape(nk, col, μ), (0, 2, 1))`
  produces `(nk, μ, col)` matching `out_spec=P(None,'x','y')` where 'x'
  shards μ and 'y' shards col.

No counterexample.

---

## 6. `cct_trace_per_q` hoist correctness (0abeb9a)

Pre-hoist (`solve_zeta`, was line 947 in pre-0abeb9a):
```python
trace_per_q = jnp.einsum('qii->q', L_q)
```
This was inside the per-r-chunk jit, firing an all-reduce across the
(μ_X, ν_Y) sharding on every r-chunk.

Post-hoist:
1. `fit_zeta_to_h5:1716-1721` computes `cct_trace_per_q = jnp.einsum('qii->q', L_q)`
   ONCE per channel where `L_q.shape = (nq_full, n_rmu, n_rmu)`. Charge
   channel passes None.
2. Threaded through `fit_one_rchunk:1356` → `_kernel:1240` (added as
   `cct_trace_per_q` kwarg).
3. Inside the kernel (lines 1309-1318): when `q_irr_idx_j is not None`
   (IBZ cascade active), the trace is sliced `cct_trace_per_q[q_irr_full_idx]`
   to match the L_q gather. Otherwise passed through unchanged.
4. `solve_zeta:946-948` consumes the precomputed value:
   ```python
   trace_per_q = (cct_trace_per_q if cct_trace_per_q is not None
                  else jnp.einsum('qii->q', L_q))
   ```

Identity: `einsum('qii->q', L_q)[q_irr_full_idx]` equals
`einsum('qii->q', L_q[q_irr_full_idx])` because the einsum is purely
elementwise on the q axis (sum over `i` of `L[q, i, i]` is a per-q reduction
with no inter-q coupling). ✓

Charge channel: trace not used in Cholesky path. The placeholder zeros at
`fit_one_rchunk:1411-1413` keep the jit signature uniform across channels
but the value is never read. ✓

Bit-identity claim (`<1 neV eqp0 diff`) consistent with the structural
identity above.

---

## 7. μ-sharded ζ producer ↔ consumer trace (b4ba9b8)

### Producer (`solve_zeta`)

The output `out_sharding` is `P(None, ('x','y'), None)` — q replicated,
μ flat-sharded over `('x','y')` mesh product, r replicated within
each rank.

Three branches all land here:
1. `cusolvermp_cholesky` (line 891-920): potrs naturally outputs
   `P(None, 'x', 'y')`. `_reshard_zeta_mu_X_r_Y_to_mu_XY` does a single
   all-to-all moving 'y' from r-axis to μ-axis. Result: `P(None, ('x','y'), None)`.
2. `cusolvermp_lu` (line 922-957): same staging as branch 1.
3. legacy `sharded_cholesky` / `lu` (line 1108-1141): solve naturally
   lands at `P(None, None, ('x','y'))` (r flat-sharded). `_reshard_zeta_r_XY_to_mu_XY`
   does a two-step reshard `(q_, μ_, r_XY) → (q_, μ_X, r_Y) → (q_, μ_XY, r_)`,
   each step a single-axis all-to-all so SPMD's planner doesn't fall back
   to Involuntary Full Rematerialization.

### Consumer (`accumulate_rchunk_to_gflat`)

`wfn_transforms.py:610-616`:
```python
in_spec  = P(None, ('x', 'y'), None)
out_spec = P(None, ('x', 'y'), None)
```
The shard_map body operates per-rank on `(n_q, n_rmu_padded/p_prod, r_len)`
slabs. The FFT runs on a per-rank-local `(cs, nx, ny, nz)` box with no
resharding (axes are fully on the rank).

### Per-element trace

Take ζ[q, μ, r]:
1. Computed by potrs on rank `(μ // (n_rmu/px) % px, r // (n_zchunk/py) % py)`
   — sharding `P(None, 'x', 'y')`.
2. After `_reshard_zeta_mu_X_r_Y_to_mu_XY`: 'y' moves from r-axis to μ-axis,
   so ζ[q,μ,r] is now on rank `μ // (n_rmu/p_xy)` (flat product index over
   ('x','y')) with r replicated. The element's VALUE is unchanged — only its
   physical residency moves. Net cost: one all-to-all on 'y' between data
   axes 1 and 2.
3. Consumer's shard_map sees ζ[q, μ_local, r] for μ_local in its slab.
   The FFT box is per-rank-local on (cs, nx, ny, nz); no reshard.

Producer and consumer agree on the same `PartitionSpec(None, ('x','y'),
None)` — verified at the type level via `out_spec` in `solve_zeta` and
`in_spec` in `accumulate_rchunk_to_gflat`.

The b4ba9b8 fix eliminated 18 calls/run of a 1.62 GiB all-gather at
`wfn_transforms.py:627` (pre-fix) and 187 MiB all-gather on `gflat_acc`
at `wfn_transforms.py:641` (pre-fix); these came from the consumer
deriving `_mu_spec = None` when ζ's μ-axis was replicated. ✓

---

## 8. Lorentz metric stabilization (1d0ba98)

**Status: reverted by later work.** Reading 1d0ba98 in isolation:

The commit moved Lorentz transverse channels from a γ̃-folded indefinite
CCT path to a "scalar Cholesky metric + γ̃ on the RHS Z only" path. Quote
from the commit body:

> The ISDF interpolation metric must stay the scalar, positive-definite
> pair-density overlap. Lorentz vertices are applied only to the RHS ZCT
> inside fit_one_rchunk. Building C_q from γ̃^i current densities makes
> the system indefinite and can catastrophically amplify null current
> modes (observed for CrI3).

The fix was therefore **semantic, not numerical regularization**. It
changed the system matrix from indefinite γ̃-folded CCT to the same
scalar PD metric for all channels.

**However**, this change was undone by the later `ce28d50` ("isdf: unify
ζ-fit on open-spin path; γ̃ identity short-circuits", 2026-05-09). The
unification commit puts γ̃ on BOTH P_l and P_r, restoring the indefinite
CCT^μ for transverse channels, but compensates by adding the proper
ridge `ε·|tr(L)|/n_rmu` and pivoted-LU path in `solve_zeta` (lines
985-1004). The current HEAD's behavior:

- charge channel: C_q PSD → Cholesky path (line 705 early-return is *not*
  taken; `factor_c_q` runs Cholesky).
- transverse: C_q indefinite (γ̃^i ⊗ γ̃^i has ±1 spectrum) → `factor_c_q`
  early-returns C_q unfactored (line 705-706); `solve_zeta` does pivoted
  LU with ridge.

The original instability 1d0ba98 was protecting against (CrI3 σ^B
blowup from amplified null current modes) is now mitigated by the LU
ridge instead. The deferred-but-stronger fix would be the proper
Gram K_q form per `project_bispinor_isdf` memory item 8 — that's
mentioned in the bispinor design doc but not in the current HEAD.

Net: the "Lorentz metric stabilization" header from 1d0ba98 is no longer
in effect. The current numerics rely on the ridge term instead. The
gamma_matrices.py portion of 1d0ba98 (static-size `_to_sparse`, line 45-48)
DID survive, and is a safe import-time fix.

**Severity**: not a defect in the audited commits — they don't introduce
the indefinite path, they preserve a previously-stable mechanism (ridge LU).
But it's worth knowing that the term "stabilization" in this commit set
refers to the inherited ridge + LU choice, not to anything new.

---

## 9. `gamma_double_contract` variants equivalence (23fec11)

Three implementations behind `LORRAX_GAMMA_CONTRACT_MODE` (default `take`):

- `_gamma_double_contract_take`: `gamma_apply(P_r, perm_L, phase_L, axis=a_axis)` then
  `gamma_apply(..., perm_R, phase_R, axis=b_axis)` then `jnp.sum(P_l_conj * P_r_p, axis=(a,b))`.
- `_gamma_double_contract_einsum`: `jnp.einsum('kabmr,aA,bB,kABmr->kmr', P_l_conj, γ̃_L, γ̃_R, P_r, optimize='optimal')`.
- `_gamma_double_contract_scan`: `lax.scan` over (a, b) pairs with rank-3 `dynamic_index_in_dim` slices.

Verified numerically on random c128 with **JAX_ENABLE_X64=1** (the
production-stack default — `gw_jax.py` sets x64 globally):

```
Pl, Pr  ∈  c128[3,4,4,5,7], random
For (mu_L, mu_R) ∈ {None,0,1,2,3}² (25 cases):
  max |take − einsum| = 0.0
  max |take − scan|   = 0.0
```

All three are bit-identical at fp64. ✓

**Caveat surfaced during audit**: under fp32 (JAX_ENABLE_X64=0 default),
the `_gamma_double_contract_einsum` path drifts from `take`/`scan` by
~5e-3 absolute error for transverse channels. This is a generic fp32
accumulation-order artifact (not a math bug), but it means
`LORRAX_GAMMA_CONTRACT_MODE=einsum` would silently produce wrong answers
if anyone ran without x64. Production runs all enable x64 — `gw_jax.py`
does this unconditionally on startup — so the practical exposure is
near zero. Worth a one-line assert in `set_gamma_contract_mode` that
x64 is enabled before allowing `einsum` mode.

---

## 10. Specific issues

| # | severity | file:line | what | fix or test |
|---|----------|-----------|------|-------------|
| 1 | low      | `isdf_fitting.py:607-611` | Explicit override `solver_kind='cusolvermp_cholesky'` can route an indefinite C_q (μ_L≠0) to Cholesky. Production caller uses `'auto'` so no current exposure. | One-line guard in `_resolve_solver_kind` raising ValueError when an explicit Cholesky kind is combined with `vertex_mu_L != 0`. |
| 2 | low      | `isdf_fitting.py:1097-1100` | Inner `@partial(jax.jit, donate_argnums=(0,))` for the legacy Z reshard is the same pattern that `6767a0d` removed from the cuSolverMp branches. Closure-defined per call → fresh cache entry, recompiles per channel. Only hits when `_already_resharded` is False (i.e., first call with `P(None,'x','y')` Z_q). | Convert to bare `with_sharding_constraint` calls — same pattern as `_reshard_zeta_mu_X_r_Y_to_mu_XY`. |
| 3 | low      | `gamma_matrices.py:302-313` | `set_gamma_contract_mode('einsum')` silently produces wrong answers (O(1e-3) drift) under fp32. Production runs x64; impact gated by configuration. | Add assertion in `set_gamma_contract_mode` that `jax.config.read('jax_enable_x64')` is True before accepting 'einsum'. |
| 4 | informational | n/a | The "Lorentz metric stabilization" header in `1d0ba98` was undone by the later `ce28d50` unification. Current HEAD relies on the LU ridge `ε·|tr(L)|/n_rmu` for transverse-channel stability, not on a scalar Cholesky metric. Not a defect; just inaccurate-looking-now commit log. | No code change. Note in PR/release notes. |
| 5 | informational | `isdf_fitting.py:1614-1620, 1660-1671` | `band_range_left = (b0, b3)`, `band_range_right = (b1, b4)` in production cohsex calls (`gw_init.py:550-551`). The resulting CCT `Σ_k P_l*·P_r` is a cross-product over different band ranges and is generally not Hermitian. Cholesky reads only the lower triangle so it "works" but the math is questionable. NOT introduced by any audited commit (pre-existing). | Out of audit scope. Worth documenting whether the production fit actually requires `band_range_left == band_range_right` semantically and whether a symmetrization step is missing. |

---

## 11. Open questions

1. **Asymmetric L/R bands**: with `band_range_left = (b0, b3)` and
   `band_range_right = (b1, b4)` in production cohsex, is the resulting
   `Σ_k P_l*·P_r` CCT guaranteed Hermitian/PSD, or is it only "Cholesky
   tolerates it because we feed the lower triangle"? `c_q_from_psi_sm`
   does not symmetrize (compare `gram_q0_from_pair` at line 200 which
   does). Either the math has a subtle symmetry I missed or this is a
   pre-existing pre-bispinor latent question that would surface if the
   triangular Cholesky behavior changed. Not introduced by any audited
   commit — flagging only.

2. **`cct_trace_per_q` placeholder shape (`isdf_fitting.py:1411-1413`)**:
   the charge channel passes `jnp.zeros((meta.nk_tot,))` with size
   `nk_tot`. After the IBZ slice `cct_trace_per_q[q_irr_full_idx]` the
   kernel sees a `(n_q_disk,)` array. When `q_irr_full_idx is None` and
   `nk_tot != nq` (degenerate kgrid?), is the trace shape compatible
   with `L_q`'s leading axis? In practice `nk_tot == nq` for cohsex
   (full BZ), so no exposure, but a defensive `assert nk_tot == nq` at
   the call site would catch a future shape regression.

3. **CCT^μ proper Gram K_q form** (deferred from `project_bispinor_isdf`
   item 8): the more rigorous approach for transverse channels is to use
   the proper Gram K_q (literal Gram of band-pair vectors), which is PSD
   for all four channels and admits Cholesky without ridge. The current
   code does NOT implement this — it relies on the indefinite γ̃-folded
   CCT + LU ridge. Item 4 in §10 is a pointer at this; not a defect, but
   worth tracking.

---

End of report. ~2700 words, in-range.
