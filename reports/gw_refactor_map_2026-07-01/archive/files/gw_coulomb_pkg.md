# gw/coulomb package — refactor map notes

Files (all relative to `sources/lorrax_D`):
`src/gw/coulomb/__init__.py` (28 LOC), `base.py` (162), `bulk_3d.py` (73), `slab_2d.py` (107), `box_0d.py` (53). Total 423 LOC.

## Package-level headline findings

1. **The package's flagship API `v_qG` is dead.** Grep for `v_qG` across the entire
   repo (`grep -rn "v_qG" --include="*.py" .`) finds hits ONLY inside
   `src/gw/coulomb/` plus an unrelated local variable in
   `misc/archived_tests/cohsex_noisdf.py`. No production code calls
   `Bulk3D.v_qG`, `Slab2D.v_qG`, or `Box0D.v_qG`. The production V(q+G) body
   kernel lives in `src/gw/compute_vcoul.py` (`make_v_munu_chunked_kernel`
   with its own `sys_dim` 0/2/3 branches at lines 269–388, and the standalone
   `compute_v_q_per_G` at lines 734–836 with identical 8π/|q+G|² and
   Ismail-Beigi slab formulas). The docstring "Driver pattern" in
   `__init__.py` describing `kernel.v_qG(...)` describes an aspiration, not
   the code as wired.
2. **The only live entry into this package** is
   `gw.vcoul.compute_q0_averages` (vcoul.py:175–196), a thin wrapper that
   does `get_kernel(meta.sys_dim).q0_average(...)`. Its callers:
   `src/gw/head_correction.py:131,156` (`from_epshead` and `from_s_tensor`
   HeadSample builders), `scripts/checks/sigma_direct_check.py:70,172,201`,
   `scripts/checks/w_from_eps0_0d_check.py:59,124`. All call sites use
   default `nsamples/method/qmc_reps` — the sampling knobs are never
   overridden from any config.
3. **This is the third copy of per-dimension Coulomb branching** in the
   codebase: (a) this package, (b) `gw/compute_vcoul.py` sys_dim branches,
   (c) `common/coulomb_sphere.py` sys_dim-aware sphere narrowing. A refactor
   should pick one home.

---

## src/gw/coulomb/__init__.py (28 LOC)

Re-exports `CoulombKernel, SysDim, get_kernel, Bulk3D, Slab2D, Box0D`.
Docstring documents the two-method driver pattern (`v_qG`, `q0_average`) and
notes the `D_munu` projector runs outside the package. No logic.

Consumers of the package import: only `src/gw/vcoul.py:190`
(`from .coulomb import get_kernel`). No tests import `gw.coulomb`
(tests/test_per_q_sphere.py imports `common.coulomb_sphere`, a different module).

Caution for refactorers: `gw.aot_memory_model.core` defines its OWN
unrelated `get_kernel` (kernel registry) and `SysDims` (dataclass) —
name collision, no code relation.

## src/gw/coulomb/base.py (162 LOC)

Dispatcher + shared mini-BZ sampler.

| function | lines | role |
|---|---|---|
| `SysDim(int, Enum)` | 27–36 | BULK_3D=3, SLAB_2D=2, BOX_0D=0. int-subclassed so legacy `cohsex.in` ints (0/2/3) parse via `int(...)` and `meta.sys_dim == 3` comparisons still work. |
| `CoulombKernel(Protocol)` | 39–72 | Interface: `v_qG(wfn, qvec_wrapped, comps_qG) -> (nG,) complex128` with q+G=0 zeroed; `q0_average(wfn, meta, *, S_cart, epshead, nsamples=2**18, method="sobol", qmc_reps=10) -> (vc0_mean, wcoul0)`. Docstring claims BGW volume convention `v_q(G)·(1/Ω_cell)` applied so `ζ Vc(G) ζ†` comes out in Ry. |
| `get_kernel(sys_dim)` | 75–99 | Maps None→BULK_3D, int/SysDim→instance via lazy per-dim import. Raises ValueError on other values. Caller: `gw/vcoul.py:190`. |
| `sample_minibz_qpoints(wfn, meta, *, nsamples=2**18, method="sobol", qmc_reps=10)` | 107–162 | Sobol-QMC (scipy.stats.qmc, seed=rep, `random_base2(m)` with m=floor(log2(nsamples))) or uniform-fallback sampling of the mini-BZ Voronoi cell. Pipeline per batch: `randcart=(bvec.T @ U.T).T` → `wrap_points_to_voronoi(randcart, bvec, nmax=1)` (imported from `..vcoul`, comment at line 122–124 records a prior `shifts @ bvec.T` vs `shifts @ bvec` bug in a local reimplementation) → scale by `randlims = bvec.T @ (diag(1/(nkx,nky,nkz)) @ inv(bvec.T))` to shrink to the mini-BZ. For `sys_dim==2` sets `rq[:,2]=0`. Returns list of `(nsamples,3)` float64 cartesian q batches (one per Sobol rep). Callers: `bulk_3d.q0_average`, `slab_2d.q0_average` only. |

Flags consumed: `meta.sys_dim` (cohsex.in `sys_dim`, default 2 per
docs/docs_gwjax/COHSEX_INPUT.md:77 — note gw_jax.py:150 sets it dynamically
on meta; `Meta` has no sys_dim field, see gw_init.py:789–793 comment),
`meta.nkx/nky/nkz`. Arrays: `wfn.blat`, `wfn.bvec` (3×3 host float64);
outputs are small device arrays (≤ 2**18 × 3).

Weird:
- line 150–152: bare `except Exception:` around the whole Sobol path — any
  error (not just missing scipy) silently degrades to uniform sampling and
  silently bumps `nsamples` to 2,500,000.
- line 155: uniform fallback uses fixed `PRNGKey(0)` — deterministic but a
  magic seed.
- `qmc_reps=10` × `2**18` samples default = 2.6M q-points, eager per-rep
  Python loop (also flagged in src/gw/PERFORMANCE.md:145 for the sibling
  code in vcoul.py).

## src/gw/coulomb/bulk_3d.py (73 LOC)

| function | lines | role / equation |
|---|---|---|
| `Bulk3D.v_qG` | 15–26 | v(q+G) = 8π/\|q+G\|², then `v /= wfn.cell_volume`, zeroed where \|q+G\|²<1e-12. Pure numpy → `jnp.complex128 (nG,)`. **DEAD — zero callers (see headline finding 1).** |
| `Bulk3D._vq_isotropic` | 28–30 | v(q)=8π/q², einsum `"ij,ij->i"` (verbatim). NOTE: no 1/Ω here. |
| `Bulk3D.q0_average` | 32–73 | vc0_mean = ⟨8π/q²⟩ over mini-BZ samples, mean over reps. If `S_cart` given (dipole-derived screening tensor S(ω), from `common.chi_from_dipole.compute_S_omega` via head_correction): wcoul0 = ⟨ v(q)/(1 − v(q)·qᵀSq) ⟩ with `qSq = jnp.einsum('qi,ij,qj->q', rq, S, rq)` (verbatim). Else epshead fallback: Ismail-Beigi gamma model, γ = (1/ε₀₀ − 1)/(q₀²·v(q₀)) at hardcoded q₀_crys=(0.001,0,0), then wcoul0 = ⟨ v(q)/(1 + v(q)·q²·γ) ⟩ over **the last batch only** (`batches[-1]`, line 68). Returns complex128 scalars. Caller: `gw.vcoul.compute_q0_averages` → `gw.head_correction`. |

Weird:
- **Volume-convention split**: `v_qG` divides by `cell_volume`; `q0_average`
  does NOT (vc0/wcoul0 are bare 8π/q² averages). The Protocol docstring in
  base.py claims 1/Ω is applied to "outputs". Hypothesis: head consumers
  (head_correction / Σ head term) carry the 1/Ω·(1/nk) mini-BZ factor
  themselves; since v_qG is dead the inconsistency is latent, but any
  refactor resurrecting v_qG must re-audit this.
- epshead fallback uses only `batches[-1]` for wcoul0 while vc0_mean
  averages all reps — asymmetric estimator (probably historical
  bit-compat).
- magic constants: 1e-12 zero-threshold, q0=0.001 crystal.

## src/gw/coulomb/slab_2d.py (107 LOC)

Ismail-Beigi slab truncation along c: v_2D(q+G) = (8π/\|q+G\|²)·(1 − e^{−zc·\|q∥+G∥\|} cos((qz+Gz)·zc)), zc = π/b_z with `b_z = bvec[2,2]` (blat-scaled).

| function | lines | role |
|---|---|---|
| `Slab2D.v_qG` | 21–38 | Formula above on the per-q G list, `/cell_volume`, zeroed at q+G=0. **DEAD — zero callers.** |
| `Slab2D._vq_2d` | 40–48 | 4π/q² × 2·(1 − e^{−zc·kxy}) form (the "2·" pulls 4π→8π). **DEAD — grep `_vq_2d` across repo: only its own definition, `q0_average` uses the local closure `_vq_sobol` instead.** |
| `Slab2D.q0_average` | 50–107 | `_vq_sobol(rq)` closure (lines 70–77): 8π/q² × (1 − e^{−zc·kxy} cos(rq[:,2]·zc)); since sampler forces qz=0 the cosine ≡ 1, kept "for bit-identity with prior runs" per comment. vc0_mean = rep-mean of batch means. S_cart branch identical in structure to Bulk3D (`einsum('qi,ij,qj->q', rq, S, rq)` verbatim). epshead fallback: 2D gamma model with vc_q0 = (1 − e^{−q₀·zc})/q₀² at q₀_crys=(0.001,0,0); wq built on `batches[-1]`; **wcoul0 = 8π·mean(wq)** — the 8π is factored out of vc_q here, unlike the 3D file where it's inside v(q). |

Weird:
- `zc = π/bvec[2,2]` assumes the c axis is orthogonal to the plane and
  aligned with z (only element [2,2] used) — silent wrong answer for
  monoclinic slabs.
- Dead `_vq_2d` next to live `_vq_sobol` closure computing the same thing
  in a different algebraic form (4π×2 vs 8π×cos) — exactly the
  "fetch_X_dyn next to fetch_X" cruft pattern.
- comment block lines 64–69 is doubled/rambling ("In practice: Sobol uses
  the explicit formula in the historical code; we replicate it") — history
  note, not spec.
- epshead-fallback guards (`jnp.where(q0len > 0, ...)`, lines 96–100)
  guard against a q0 that is hardcoded nonzero two lines earlier.

## src/gw/coulomb/box_0d.py (53 LOC)

0D Wigner-Seitz cell-box truncation; numerical work delegated to
`gw.compute_vcoul_0d.compute_vcoul_box` (real-space 1/r on dense grid,
WS minimum-image truncation, FFT; cf. BGW Common/trunc_cell_box.f90).

| function | lines | role |
|---|---|---|
| `Box0D.v_qG` | 24–30 | `compute_vcoul_box(wfn.bdot, wfn.fft_grid, comps_qG) / cell_volume` → complex128. Ignores `qvec_wrapped` (box truncation is q=0 only). **DEAD — zero callers.** |
| `Box0D.q0_average` | 32–53 | vc0 = truncated v(G=0) from the same FFT (`g0=[[0,0,0]]`), `/cell_volume`; returns `(vc0, vc0)` — BGW convention wcoul0 = vc0 for box truncation (comment cites BGW Common/vcoul_generator.f90:717). Explicitly `del`s all sampling kwargs. Caller: `gw.vcoul.compute_q0_averages`; exercised by `scripts/checks/w_from_eps0_0d_check.py`. |

Weird:
- Unlike Bulk3D/Slab2D, `q0_average` here DOES divide by cell_volume —
  a third convention within the same Protocol (0D: /Ω in both methods;
  3D/2D: /Ω in v_qG only). Consumers must currently know which dim they
  are in; refactor should unify.
- `v_qG` recomputes the full dense-grid FFT per call with no cache
  (moot while dead).

## I/O

None in this package. No files read or written. Upstream inputs it depends
on transitively: `eps0mat.h5` epshead (via head_correction → EPSReader) and
`dipole.h5` S-tensor (via head_correction → chi_from_dipole) — both read
OUTSIDE this package and passed in as arrays.

## Flags consumed

- `sys_dim` (cohsex.in; set on `meta` dynamically by gw_jax.py:150; Meta
  dataclass has no such field — see gw_init.py:789 workaround comment).
- `meta.nkx, meta.nky, meta.nkz` (mini-BZ size).
- `nsamples / method / qmc_reps` exist as kwargs but no config key ever
  sets them (all production callers use defaults).
- Indirectly: `wcoul0_source` (head_correction chooses epshead vs S_cart
  path — outside this package).

## Suspect summary

Dead (grep evidence: `grep -rn "v_qG" --include="*.py" .` from repo root;
`grep -rn "_vq_2d"` repo-wide):
- `Bulk3D.v_qG`, `Slab2D.v_qG`, `Box0D.v_qG` — zero callers.
- `Slab2D._vq_2d` — zero callers.
- Consequently `CoulombKernel.v_qG` protocol slot and the `__init__.py`
  "driver pattern" docstring describe unwired API.

Redundancy:
- v_qG formulas duplicate `gw/compute_vcoul.py` (`_sqrt_v_phase` kernels
  lines 305–388 and `compute_v_q_per_G` lines 734–836) — the live V_q
  path never routes through this package.
- `Slab2D._vq_2d` vs `_vq_sobol` closure (same physics, two algebraic
  forms, one dead).
- Third sys_dim branch set in `common/coulomb_sphere.py`.
- `gw/vcoul.py` also still contains `compute_wcoul0_with_S` (line 198)
  and commented-out sampler code (line 127) overlapping this package's
  q0_average — outside assigned scope but relevant to the same
  consolidation.
