# Agent 2 — Design sketch (parallel draft, pre Agent 1)

**Status**: parallel draft written before Agent 1's scope landed. May be
revised once Agent 1 enumerates additional sites.

This document spells out the per-element math for the TRS-augmented
sym-table at three candidate design points (A / B / C). The goal is to
nail down the centroid-permutation and V_q-conjugation rules under
`s ≥ ntran` so the patch is unambiguous.

---

## 0. Convention recap (current code, `symmetry_maps.py:116-130`)

```
sym_matrices          shape (ntran, 3, 3) — BGW mtrx; acts on G-vec
sym_mats_k[:ntran]    = sym_matrices.transpose(0,2,1) — acts on k
sym_mats_k[ntran:]    = -sym_mats_k[:ntran]            — TRS rows
translations          shape (ntran, 3)   — BGW tnp; τ_frac = tnp/(2π)
```

The TRS-augmented half of `sym_mats_k` has no companion entry in
`sym_matrices` or `translations`. Consumers that want `S r + τ` for
sym index `s ≥ ntran` have to know that:

- the *spatial* part is `S[s mod ntran]`,
- the *time-reversal* tag is `s >= ntran`,
- the τ-vector is `translations[s mod ntran]`,
- the wavefunction action is `K ∘ {S | τ}`, where `K` is complex-conjugation.

Today the consumers do NOT know this — they index `sym.sym_matrices[s]`
or `sym_perm[s]` and silently clip when `s ≥ ntran`. That's the bug.

---

## 1. Math: TRS action on ζ and V_q

### 1.1 ζ under {S | τ} (spatial only, current eq. 1)

From `reports/zeta_ibz_2026-05-11/report.md`:

```
ζ_{Sq, π_s(μ)}(G_new) = e^{-i(Sq + G_new)·τ_s} · ζ_{q, μ}(S^{-1} G_new)     (eq. 1)
```

### 1.2 ζ under TRS only (S = −1, τ = 0, K)

Bloch + TRS: `u_{-q,n}(r) = u*_{q,n}(r)`. The pair density `ρ_μ(r, r')`
that ζ represents (`ζ_μ(r) ≡ Σ_n ψ_n(r_μ)* ψ_n(r)` schematically) then
satisfies `ρ_μ^{-q}(r) = ρ_μ^q*(r)` (the role of bra/ket swaps under K,
which conjugates both factors). In G-space:

```
ζ_{-q, μ}(G) = ζ*_{q, μ}(-G)                                                 (eq. 1-TRS)
```

The centroid index `μ` is unchanged (TRS keeps `r_μ` fixed).

### 1.3 ζ under {K · S | τ} (TRS-augmented spatial)

Combine 1.1 and 1.2. Let `S` be a spatial op with translation `τ`, and
let the full op be `K ∘ {S | τ}`. Then `(KS)q = -Sq`, and:

```
ζ_{-Sq, π_s(μ)}(G_new) = e^{+i(Sq + G_new)·τ_s} · ζ*_{q, μ}(-S^{-1} G_new)
                       = e^{+i(-Sq + (-G_new))·(-τ_s)} · ζ*_{q, μ}(S^{-1}(-G_new))
```

Equivalently — using the row-`s` view where `S' = -S` and `τ' = -τ` are
the "k-action" matrix and the "phase generator":

```
ζ_{S'q, π_{s mod ntran}(μ)}(G_new)
   = e^{-i(S'q + G_new)·τ'} · ζ*_{q, μ}((S')^{-1} G_new)                    (eq. 1-TS)
```

The structural form is identical to 1.1 with three changes:

1. `S` → `S' = -S` (already encoded in `sym_mats_k[s]` for `s ≥ ntran`).
2. `τ` → `τ' = -τ` (the τ-vector flips sign under K).
3. ζ → ζ* (extra conjugation on ζ).

This is the precise modification we need to thread through every
"unfold IBZ → full BZ" consumer.

### 1.4 V_q under {K · S | τ}

`V_{q,μν} = Σ_G ζ*_{q,μ}(G) · v(q+G) · ζ_{q,ν}(G)`. Substituting eq.
1-TS for both ζ legs:

```
V_{S'q, π_s(μ), π_s(ν)}
  = Σ_{G_new} ζ_{S'q, π_s(μ)}*(G_new) · v(S'q+G_new) · ζ_{S'q, π_s(ν)}(G_new)
```

Insert eq. 1-TS on both ζ's. The two τ-phases (one with +i, one with
−i because of the outer `*` on the bra) cancel. The `v(S'q + G_new) =
v(q + (S')^{-1} G_new)` because v is even in K (Coulomb is real). The
double-conjugation gives one residual conjugation:

```
ζ*_{S'q, π_s(μ)}(G_new) · ζ_{S'q, π_s(ν)}(G_new)
  = [e^{-i(S'q+G_new)·τ'} · ζ*_{q,μ}((S')^{-1}G_new)]^* · [e^{-i(S'q+G_new)·τ'} · ζ*_{q,ν}((S')^{-1}G_new)]
  = e^{+i(S'q+G_new)·τ'} · ζ_{q,μ}((S')^{-1}G_new) · e^{-i(S'q+G_new)·τ'} · ζ*_{q,ν}((S')^{-1}G_new)
  = ζ_{q,μ}((S')^{-1}G_new) · ζ*_{q,ν}((S')^{-1}G_new)
```

Renaming `G' = (S')^{-1} G_new`:

```
V_{S'q, π_s(μ), π_s(ν)} = Σ_{G'} ζ_{q,μ}(G') · v(q+G') · ζ*_{q,ν}(G')
                        = V*_{q, ν, μ}
                        = V_{q, μ, ν}*           (if V is Hermitian in μν)
```

`V_q` IS Hermitian in μν (it's `ζ† · diag(v) · ζ`), so `V*_{ν,μ} =
V_{μ,ν}`. Either way:

**TRS rule for V_q-unfold**:

```
V_{full}^{q, π_s(μ), π_s(ν)} = conj( V_{ibz}^{i(q), μ, ν} )   if s ≥ ntran
V_{full}^{q, π_s(μ), π_s(ν)} =       V_{ibz}^{i(q), μ, ν}     if s <  ntran   (current code)
```

where `π_s` is the **spatial-half** permutation: `π_s = π_{s mod ntran}`.

### 1.5 g0 under {K · S | τ}

g0 is a single ζ-leg at G=0 (head-Coulomb correction support). Under
1-TS:

```
g0_{full}^{q, π_s(μ)} = conj( g0_{ibz}^{i(q), μ} ) · e^{+i(0+0)·τ'}
                      = conj( g0_{ibz}^{i(q), μ} )         if s ≥ ntran
```

At Γ (the only place g0 is actually consumed today), `s = 0 < ntran`,
so the only change for `_unfold_g0_ibz_to_full` is: when `s ≥ ntran`
apply conj on top of the centroid permutation.

### 1.6 Bispinor extension (not in this patch, but design must not preclude)

Bispinor TRS is `T = i σ_y K`. On a (2-component) spinor ψ:

```
T ψ = i σ_y · K ψ = (i σ_y) · ψ*
```

So the bispinor ζ_μ (which carries an extra spinor index `a`) transforms
as `ζ_{Tq, μ, a} = (i σ_y)^a_b · ζ*_{q, μ, b}`. The V_q bilinear is
`Σ_{a,G} ζ*_{q,μ,a}(G) v(q+G) ζ_{q,ν,a}(G)`, summed over the spinor
index. Under T:

```
ζ*_{Tq,μ,a} ζ_{Tq,ν,a}  =  [i σ_y]^a_b [i σ_y]^a_c · ζ_{q,μ,b} ζ*_{q,ν,c}
                        =  δ_{bc} · ζ_{q,μ,b} ζ*_{q,ν,b}     (σ_y σ_y = 1)
                        =  ζ*_{q,ν,b} ζ_{q,μ,b}             (no spinor extras)
```

So the bispinor-V_q-unfold rule under TRS is structurally identical to
the scalar case — `(σ_y)(σ_y) = 1`. **The scalar abstraction generalises
cleanly to bispinor V_q.** Bispinor ψ-unfold separately requires
`(i σ_y) · ψ*` (which the WfnLoader already handles in part — see
`wfn_loader.py:_eager_build` and the phdf5 kernel; though both currently
apply only `conj`, not `σ_y · conj`, so bispinor TRS-augmented k-points
are also a known latent bug, out of scope for this initiative).

---

## 2. Design option A — Extended sym table with `is_trs` tag

Introduce a `SymTable` dataclass that the codebase passes around in place
of bare `sym_matrices` / `sym_mats_k`:

```python
@dataclass(frozen=True)
class SymTable:
    mtrx:    np.ndarray   # (n_op, 3, 3) — spatial G-action ("S" in BGW)
    mtrx_k:  np.ndarray   # (n_op, 3, 3) — spatial k-action = S.T
    tau:     np.ndarray   # (n_op, 3)    — fractional translation (BGW sign: tnp/(2π))
    is_trs:  np.ndarray   # (n_op,) bool
    # Convenience: spatial_idx[i] = i mod ntran (the row of mtrx whose
    # action on G-vec coincides with row i of mtrx_k when ignoring K).
    # Provided as a derived property:
    @property
    def spatial_idx(self) -> np.ndarray: ...   # int32
```

The current `sym_mats_k`-of-shape-`(2 ntran, 3, 3)` is then equivalent
to `SymTable(mtrx=stacked, mtrx_k=stacked_k, tau=stacked_tau,
is_trs=[False]*ntran + [True]*ntran)`. Wherever the codebase reaches
into `sym.sym_mats_k[s]`, it instead pulls `sym.symtable.mtrx_k[s]` (and
where needed `sym.symtable.tau[s]`, `sym.symtable.is_trs[s]`,
`sym.symtable.spatial_idx[s]`).

**`_unfold_v_q_ibz_to_full` under option A**:

```python
def _unfold_v_q_ibz_to_full(V_q_ibz, *, full_to_irr_idx, full_to_irr_sym,
                            sym_perm, is_trs, mesh_xy):
    inv_perm = argsort(sym_perm, axis=-1)              # (ntran, n_rmu)
    perm_q = inv_perm[sym_full[:] % ntran]             # use spatial half!
    V_at_irr = V_ibz[full_to_irr_idx]                   # (n_q_full, μ, μ)
    V_perm_mu = take_along_axis(V_at_irr, perm_q[:, :, None], axis=1)
    V_perm_full = take_along_axis(V_perm_mu, perm_q[:, None, :], axis=2)
    # TRS conjugation:
    trs_mask = is_trs[full_to_irr_sym]                  # (n_q_full,)
    V_full = where(trs_mask[:, None, None], conj(V_perm_full), V_perm_full)
    return V_full
```

Key changes vs. today:

- `sym_perm` stays shape `(ntran, n_rmu)` — built from spatial-only sym,
  no TRS rows. (Compatible with `compute_centroid_sym_perm` today,
  which already operates on `n_tran = sym_matrices.shape[0]` rows.)
- The unfold uses `full_to_irr_sym % ntran` to index into `sym_perm`,
  not the raw value — fixing the silent OOB clip.
- A `is_trs` boolean per full-q triggers `conj` on the unfolded V_q.

**Pro**: First-class typed object; future-proof (bispinor extension
adds an `apply_to_spinor` method on `SymTable` that uses `is_trs +
spatial_idx`).

**Con**: Larger diff; every caller of `sym.sym_mats_k` migrates to the
new accessor; touches `find_irreducible_qpoints`, `find_symmetry_ops_simple`,
`get_kfull_symmap`, the wfn unfold path, etc.

---

## 3. Design option B — Spatial-only everywhere

Replace `sym_mats_k` (length `2 ntran`) with the spatial-only half in
`find_irreducible_qpoints` and similar consumers. The k-fold and q-fold
then produce a larger IBZ (no TRS reduction), but everything is
unambiguous because there are no `s ≥ ntran` indices.

**`_unfold_v_q_ibz_to_full` under option B**:

```python
def _unfold_v_q_ibz_to_full(V_q_ibz, *, full_to_irr_idx, full_to_irr_sym,
                            sym_perm, mesh_xy):
    # full_to_irr_sym now ranges in [0, ntran); no TRS rows.
    inv_perm = argsort(sym_perm, axis=-1)
    perm_q = inv_perm[full_to_irr_sym]
    ...
```

(Same body as today, but the indexing into `sym_perm` is now in range.)

**Pro**: Smallest behavioural change; no `conj` plumbing.

**Con**: Loses the TRS reduction in the q-IBZ wedge. For systems without
inversion (where TRS is genuinely needed to reach the smallest wedge),
this halves the speedup of the IBZ cascade. MoS2 3×3 has 9 q's; current
TRS-augmented wedge has 3 q's; spatial-only wedge has more. Performance
regression on every non-centrosymmetric calculation forever.

**Verdict**: Punts the bug fix's value-add. Don't pick this.

---

## 4. Design option C — Hybrid, narrow patch

Keep TRS in q/k-fold, but make `compute_centroid_sym_perm` produce a
`(2 ntran, n_rmu)` table where rows `[ntran:]` duplicate rows `[:ntran]`
(centroid permutation is unchanged by TRS, because r_μ → r_μ under K).
Then the unfold helper applies `conj` when `s ≥ ntran`:

```python
def compute_centroid_sym_perm(..., extend_trs=False) -> np.ndarray:
    spatial_perm = ...     # (ntran, n_rmu) as today
    if extend_trs:
        # TRS keeps r fixed; same permutation.
        return np.concatenate([spatial_perm, spatial_perm], axis=0)
    return spatial_perm
```

```python
def _unfold_v_q_ibz_to_full(V_q_ibz, *, full_to_irr_idx, full_to_irr_sym,
                            sym_perm, ntran, mesh_xy):
    # sym_perm shape: (2 ntran, n_rmu) when extend_trs=True
    inv_perm = argsort(sym_perm, axis=-1)
    perm_q = inv_perm[full_to_irr_sym]                  # (n_q_full, n_rmu)
    V_at_irr = V_ibz[full_to_irr_idx]
    V_perm_mu = take_along_axis(V_at_irr, perm_q[:, :, None], axis=1)
    V_perm_full = take_along_axis(V_perm_mu, perm_q[:, None, :], axis=2)
    trs_mask = full_to_irr_sym >= ntran
    V_full = where(trs_mask[:, None, None], conj(V_perm_full), V_perm_full)
    return V_full
```

**Pro**: Tiniest diff. Only `_unfold_v_q_ibz_to_full`,
`_unfold_v_q_ij_ibz_to_full`, `_unfold_g0_ibz_to_full`, and
`compute_centroid_sym_perm` get touched. No global rename.

**Con**: The `ntran` parameter has to be threaded through every call
site; the `extend_trs=True` flag is a magic state that consumers must
opt into. Doesn't generalise — bispinor wfn-unfold under TRS still has
a separate fix later (with σ_y).

---

## 5. Recommendation (pending Agent 1)

**Pick option A** (typed `SymTable`) **for these reasons**:

1. Centralises the spatial-half / TRS-half distinction in one place. No
   risk of future consumer reading `sym_mats_k[s]` and re-introducing
   the silent OOB clip.
2. The bispinor `apply_to_spinor` method is a clean extension point —
   when we later add bispinor wfn unfold (`(iσ_y)·conj` for `is_trs`
   rows), it's one method on `SymTable`, not a parallel patch in three
   modules.
3. `is_trs + spatial_idx` is what the underlying physics is doing
   anyway; the bare `2 ntran`-row matrix is an encoding shortcut whose
   ambiguity caused the bug.
4. Per-site change count is bounded: scope is `SymMaps.__init__`
   building the table, plus the ~8 callsites Agent 1 will enumerate.
   Not a rewrite.

Cost: a few hours of mechanical refactor. Benefit: bug class
eliminated, design extensible.

**Fallback if Agent 1 finds many surprise call sites**: revert to
option C (narrow patch). Don't pick B.

---

## 6. Per-site patch shape (will be filled after Agent 1 reports)

Placeholder list, sorted by file:

1. `src/common/symmetry_maps.py` — build `SymTable` in `SymMaps.__init__`;
   keep `sym_mats_k` as a derived property for now (deprecate later).
2. `src/gw/v_q_tile.py:_unfold_v_q_ibz_to_full` — accept `is_trs`,
   `spatial_idx`, apply conj.
3. `src/gw/v_q_tile.py:_unfold_v_q_ij_ibz_to_full` — same. (Cartesian
   tensor mixing `R_q` uses the spatial half regardless of TRS, since
   the Coulomb tensor is even in K.)
4. `src/gw/v_q_tile.py:_unfold_g0_ibz_to_full` — same.
5. `src/gw/v_q_g_flat.py:_resolve_ibz_q_list` — produce `sym_perm` for
   spatial rows only (already true today; just pass `ntran` /
   `is_trs` downstream).
6. `src/centroid/orbit_syms.py:compute_centroid_sym_perm` — unchanged
   (already spatial-only).
7. Anywhere else Agent 1 finds — TBD.

Tests:

- New unit test `tests/test_trs_unfold_helpers.py`: build a synthetic
  4×4×1 toy with one spatial op and one TRS-augmented op, hand-compute
  the expected V_full, check the helper matches.

---

End of pre-Agent-1 sketch. The math in §1 is the load-bearing content;
the option-A recommendation in §5 is provisional pending the scope
report.
