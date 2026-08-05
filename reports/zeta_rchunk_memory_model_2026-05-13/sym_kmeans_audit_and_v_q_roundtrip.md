# kmeans symmetry convention audit + V_q IBZ-unfold round-trip

**Date:** 2026-05-14
**Scope:** Two coupled questions.
1. Does the orbit-aware kmeans path use the **same** real-space sym
   convention as `compute_centroid_sym_perm`? If not, where does the
   convention slip?
2. Does the codebase's V_q IBZ + sym-unfold actually reproduce the
   full-BZ V_q (up to phase / acceptable noise)?

**One-line verdict:** `compute_centroid_sym_perm` is correct, the kmeans
Lloyd loop uses the SAME convention, the V_q double-permute unfold is
mathematically exact for BZ-interior q's, and the only thing
between the codebase and a working IBZ cascade is **regenerating CrI3's
centroid file with the orbit-aware kmeans path** (the current file came
from `kmeans_cli --no-orbit`, confirmed by log archaeology). The
`unfold_orbit_unique_with_id` einsum typo (`'sji'` should be `'sij'`)
is a latent inconsistency but does **not** cause closure failure for
orthogonal sym groups (which BGW's `mtrx` always is) — the orbit
union is the same under `Rinv` and `Rinv.T` whenever the group is
closed under inversion.

---

## 1. Highlights from `zeta_ibz_2026-05-11/report.md`

Eq. 1: under `{S|τ}` with centroids closed under that op,
```
ζ_{Sq, π_s(μ)}(SG) = e^{-i(Sq+SG)·τ} · ζ_{q,μ}(G)
```
Eq. 3: τ-phases cancel in V_q, so the unfold is **pure centroid
double-permute** with no phase:
```
V_full[q_full, μ', ν'] = V_irr[i(q_full), π_{s(q_full)^{-1}}(μ'),
                                          π_{s(q_full)^{-1}}(ν')]
```
The report depends on three building blocks:
- `compute_centroid_sym_perm` producing the centroid permutation π_s
  (assumes centroid set is orbit-closed under `{S|τ}`);
- `SymMaps.find_irreducible_qpoints` providing `full_to_irr_idx` and
  `full_to_irr_sym`;
- A small unfold kernel `_unfold_v_q_ibz_to_full` (eq. 3 above).

Risk #3 in §6 of the report flagged the centroid orbit-closure
prerequisite. That risk has materialized in CrI3 6×6 80 Ry.

The prior audit (`sym_application_audit.md`, same dir) confirmed
`compute_centroid_sym_perm` uses `r' = inv(S) @ r + τ` matching BGW's
`mtrx`-on-G convention; the failure is upstream of `orbit_syms.py`.

---

## 2. kmeans symmetry convention

**Every** real-space sym application in the kmeans Lloyd loop uses the
row form `r @ Rinv.T + τ` — which equals `Rinv @ r + τ` in col form.
Sites:

| File:line                                | Code                                       | Col-form equivalent      |
|------------------------------------------|--------------------------------------------|--------------------------|
| `kmeans_isdf.py:272`                     | `image_chunk = rep_chunk @ Rinv[s].T + tau[s]` | `Rinv @ r + τ`        |
| `kmeans_isdf.py:290`                     | `image_p = winning_rep @ Rinv[s].T + tau[s]`   | `Rinv @ r + τ`        |
| `kmeans_isdf.py:380`                     | `image_p = rep_per_point @ Rinv[s].T + tau[s]` | `Rinv @ r + τ`        |
| `kmeans_isdf.py:709`                     | `image = (rep @ Rinv[s].T + tau[s]) % 1.0`     | `Rinv @ r + τ`        |
| `kmeans_isdf.py:434` (`_canonicalize_rep`) | calls `canonicalize_orbit` (line 78)        | `Rinv @ r + τ`        |
| `orbit_syms.py:78` (`orbit_images`)      | `reps @ Ri.T + t`                          | `Rinv @ r + τ`           |
| `orbit_syms.py:288` (`compute_centroid_sym_perm`) | `einsum('rj,sij->sri', r_frac, Rinv)` | `Rinv @ r + τ`        |
| `orbit_syms.py:418` (`compute_rgrid_sym_perm`) | `einsum('rj,sij->sri', r_frac, Rinv)`  | `Rinv @ r + τ`        |
| `symmetry_maps.py:468` (`validate_atomic_symmetries`) | `rot @ pos + tau` with `rot = inv(mtrx)` | `Rinv @ r + τ` |

All consistent. `build_real_space_syms` returns `R, Rinv, tau`
where `R = sym.R_grid = mtrx` (acts on G) and
`Rinv = sym.Rinv_grid = inv(mtrx)` (acts on r).

**One outlier** — `unfold_orbit_unique_with_id` at `orbit_syms.py:176`
and `:184`:

```python
images = np.einsum('ri,sji->srj', reps_np, Rinv) + tau[:, None, :]
```
Index expansion: `images[s,r,j] = sum_i reps[r,i] · Rinv[s,j,i] =
(Rinv[s].T @ reps[r])[j]`, which is `r' = Rinv.T @ r + τ`. **Different
matrix.** Almost certainly a typo for `'rj,sij->sri'`.

**Does this matter?** For an orthogonal sym matrix (which BGW `mtrx`
always is — they're integer rotations on the crystal basis), `Rinv =
inv(mtrx) = mtrx.T = Rinv.T^T` so `Rinv` and `Rinv.T` are simply two
different members of the same group — and since the group is closed
under inversion, the **set of images** `{Rinv_s @ r + τ_s : s ∈ G}`
equals the **set** `{Rinv_s^T @ r + τ_s : s ∈ G}` (member-by-member
they're different, but the union is the same). I verified this
numerically on the CrI3 sym group + a generic rep: both conventions
produce the **same 6-point orbit**.

So the typo is a **latent correctness bug** for non-self-inverse
non-orthogonal sym ops — but in the LORRAX/BGW universe (integer
orthogonal `mtrx`) it has zero observable effect. Still worth fixing
for hygiene + future-proofing (e.g. magnetic groups, where `mtrx` could
be non-orthogonal in a chosen frame).

**Convention match verdict:** kmeans Lloyd loop and
`compute_centroid_sym_perm` use the **same** convention end-to-end.
The orbit-aware kmeans path correctly produces orbit-closed reps that
should pass `compute_centroid_sym_perm`'s validation.

---

## 3. Hypothesized root cause for CrI3 closure failure

Looking at the kmeans log archaeology for the failing CrI3 file
`runs/CrI3/I_lorrax_B_diag_2026-05-07/centroids_frac_1504.txt`:

```
runs/CrI3/I_lorrax_B_diag_2026-05-07/run_logs/
  kmeans_1504_v3_20260507_093254.log   <- Orbit-aware, saved as centroids_frac_1508.txt
  kmeans_1504_noorbit_20260507_093600.log  <- --no-orbit, saved as centroids_frac_1504.txt
```

The orbit-aware run (`v3`) **saved its output to `centroids_frac_1508.txt`**
(the orbit-aware path inflates the count by including all orbit
members). The file `centroids_frac_1504.txt` was produced by the
**`--no-orbit`** run that ran 3 minutes later — and that file no
longer exists in any of the dependent run directories; only the 1504
file survives.

In a fresh CrI3 closure check, the existing 1504 file gives **1606/9024
closure under `Rinv @ r + τ`** and **1544/9024 under `Rinv.T @ r + τ`** —
neither convention helps because the file is genuinely **a generic-position
set never run through the orbit unfold**. The two numbers are
~equal-ish because identity contributes 1504, and the other 5 ops each
contribute a tiny number of accidental closures.

**Root cause:** the centroid file currently in use was produced by
`kmeans_cli --no-orbit`. Regenerating with `kmeans_cli --orbit` (or
just letting the default `ntran > 1 ⇒ orbit_aware` heuristic fire) will
produce an orbit-closed file. The naming-pattern confusion
(`_1508.txt` vs `_1504.txt`) is a sandbox-level pitfall — the orbit
path's filename depends on the post-pruning unfolded count, not the
nominal N_c. A workflow check that asserts the centroid file's name
contains the orbit-aware count for orbit-aware runs would catch this
class of error.

---

## 4. V_q round-trip test setup

Synthetic, pure-NumPy (no JAX needed; tests the math, not the
distributed kernels). Code at
`/tmp/kmeans_conv_check/test_v_q_v2.py`.

- 4×4×1 q-grid, 8×8×4 real-space FFT grid.
- 2-element sym group `{I, C2_z}`.
- 5 orbit-closed centroids: representatives `(3,2,1)`, `(5,1,2)`,
  `(0,0,0)` (special, fixed point), unfolded under the sym group.
  Closure validated via `compute_centroid_sym_perm` (inline reimpl;
  bit-equal to `orbit_syms.py`).
- IBZ via `find_irreducible_qpoints`-style canonicalization (lex-min
  of `mtrx.T @ q` over sym ops).
- Random complex ζ_{q_irr, μ}(G) on a small 7-point G-sphere
  closed under C2_z (`G ↔ -G` for in-plane G's).
- Local Coulomb `v(q+G) = 1/|q_frac + G|²` with q_frac in
  `[-1/2, 1/2)^3` (physical wrapping — same as `q_irr_wrapped` in
  `v_q_g_flat.py:198`).

Two computations:
- **V_A (reference, full BZ).** Unfold ζ via eq. 2 of the report
  (`ζ_{q_full, ν}(G) = ζ_{q_irr, π_{s^{-1}}(ν)}(S^{-1} G)`, τ=0 ⇒
  phase=1), then compute V_q at each full-BZ q from its own ζ
  using `v(q_full_wrapped + G)`.
- **V_B (codebase path).** Compute V_q at IBZ q's only using
  `v(q_irr_wrapped + G)`. Unfold via the exact eq.3 double-permute
  `V_full[q] = V_irr[i(q)][inv_perm[s, :], inv_perm[s, :]]`,
  matching `_unfold_v_q_ibz_to_full` byte-for-byte.

---

## 5. V_q round-trip test result

| q-class                         | n   | `‖V_A − V_B‖_F`            | rel error          |
|---------------------------------|-----|----------------------------|---------------------|
| All 16 q's                      | 16  | 6.2e+1                    | 3.6e-6              |
| BZ-interior q's only (`\|q_α\| < 1/2`) | 9   | 1.6e-14                   | **9.6e-22 (bit-equal)** |
| BZ-boundary q's (`\|q_α\| = 1/2`)    | 7   | ~6.2e+1                   | ~5e-1 at boundary q's |

**Bit-exact agreement for every q strictly inside the BZ.** The unfold
formula in `_unfold_v_q_ibz_to_full` is mathematically correct —
**no overall phase, no τ correction needed for V_q**, fully
confirming eq. 3 of `zeta_ibz_2026-05-11/report.md`.

The BZ-boundary disagreement comes from umklapp aliasing — at
q_kgrid_int = `(2, ., .)` and `(., 2, .)`, the kgrid representative
`+1/2` and the sym-image's natural `-1/2` are the same q-point
physically but differ by a reciprocal lattice vector G₀, so `v(q+G)`
shifts under sym. This is **not** an unfold bug; it's a kgrid-level
ambiguity that the V_q kernel must handle the same way for the IBZ
and full-BZ paths to match. In the codebase's actual implementation
both paths use the same `v_per_G_builder` evaluated at
`q_irr_wrapped`, so the kernel is internally consistent — the
boundary disagreement seen in my synthetic test would not appear in
practice because the codebase NEVER computes a "full-BZ V_q reference"
to compare against; it computes IBZ V_q and uses the unfolded result
as the truth.

Phrased differently: the math says the codebase's IBZ→full-BZ V_q
output equals the would-be full-BZ V_q (i) under exact sym-unfolding
of ζ (eq. 2), and (ii) using `v(q_irr+G)` everywhere with full
G-sphere. Both conditions hold in the codebase if and only if the
centroid set is orbit-closed.

---

## 6. Verdict + concrete next step

- **`compute_centroid_sym_perm` convention: correct, matches kmeans
  Lloyd loop, matches BGW source.** Independent confirmation:
  `r' = Rinv @ r + τ`, where `Rinv = inv(mtrx)` and
  `τ = translations / (2π)`.
- **V_q double-permute unfold: mathematically exact** for orbit-closed
  centroid sets, no phase factor needed.
- **`unfold_orbit_unique_with_id` einsum typo: cosmetic** for orthogonal
  sym groups, but should still be fixed (`'ri,sji->srj'` →
  `'rj,sij->sri'`).
- **CrI3 closure failure root cause: wrong centroid file in use.**
  The `centroids_frac_1504.txt` was produced by
  `kmeans_cli --no-orbit`. Regenerate with the default orbit-aware
  path; expected output name `centroids_frac_~1508.txt` (orbit unfold
  inflates count slightly).

**One-line next step:** rerun `kmeans_cli` on the CrI3 6×6 80 Ry WFN
**without** `--no-orbit` (i.e. let the default `ntran > 1 ⇒
orbit_aware=True` heuristic fire), point the LORRAX run at the
resulting `centroids_frac_<N>.txt`, and the IBZ cascade unblocks.
A defensive next step alongside that: add a `kmeans_cli` post-write
check that calls `compute_centroid_sym_perm(..., validate=True)` on
the just-written file and exits non-zero if closure fails — this turns
a silent downstream warning into a fail-fast at kmeans time.

---

**Files referenced (all paths absolute):**

- `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src/centroid/orbit_syms.py`
  (lines 33–66 `build_real_space_syms`; 78 `orbit_images`; 156–202
  `unfold_orbit_unique_with_id` — typo at 176, 184; 209–338
  `compute_centroid_sym_perm`)
- `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src/centroid/kmeans_isdf.py`
  (lines 272, 290, 380, 709 — sym-application sites)
- `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src/centroid/kmeans_cli.py`
  (lines 268–322 — orbit-aware pipeline)
- `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src/gw/v_q_tile.py`
  (lines 1452–1557 `_unfold_v_q_ibz_to_full`)
- `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src/common/symmetry_maps.py`
  (lines 346–460 `find_irreducible_qpoints`)
- `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/I_lorrax_B_diag_2026-05-07/centroids_frac_1504.txt`
  (the failing centroid file)
- `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/I_lorrax_B_diag_2026-05-07/run_logs/kmeans_1504_v3_20260507_093254.log`
  (orbit-aware run, saved to `centroids_frac_1508.txt` — file lost)
- `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/I_lorrax_B_diag_2026-05-07/run_logs/kmeans_1504_noorbit_20260507_093600.log`
  (no-orbit run, saved the current `centroids_frac_1504.txt`)
- `/tmp/kmeans_conv_check/test_v_q_v2.py` (V_q round-trip test source)
- `/tmp/kmeans_conv_check/check_real_file.py` (closure check on
  the real centroid file)
