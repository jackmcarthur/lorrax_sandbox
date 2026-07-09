# Group: src/gw/experimental/ — staged (not-yet-production) GW features

Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`

---

## File: `src/gw/experimental/head_wing_schur.py` (274 LOC)

### Purpose

Sharded head/wing/body Schur (bordered Sherman–Morrison) decomposition of the
screened interaction W in the ISDF centroid basis. It splits the q→0 head
channel out of the dielectric solve so the rank-1 head subtract/rebuild are
zero-communication outer products under the production 2-D device mesh
(`V_FFT5D_SPEC` convention, expressed here in flat-q form), while the
unavoidable Schur reductions collapse to one small all-reduce per q.
The module docstring is explicit that this is **purely the algebraic kernel**:
it does NOT replace production W yet; a separate driver patch is supposed to
plumb it into the static COHSEX path after `tests/test_head_wing_schur.py`
confirms correctness vs a dense reference and that the HLO has no collectives
in the outer-product steps.

Stated provenance: "Mirror of agent-A's bordered Sherman-Morrison
reconstruction (`sources/lorrax_A/src/psp/finite_q_head_interp.py`), recast in
the production V_FFT5D_SPEC sharding."

**Category guess:** physics: chi0/W stage — staged/experimental head-correction
kernel (sharded linear algebra for W reconstruction).

### Math being implemented

Block the dielectric problem into head (the single q→0 Coulomb-divergent
channel spanned by g0) and body (everything else):

1. `V_body[q,μ,ν] = V_q[q,μ,ν] − v_head[q] · conj(g0[q,μ]) · g0[q,ν]`
2. `W_body0 = (1 − pref·V_body·χ_body)^{-1} · V_body`  (existing solver)
3. Schur complement of the head channel:
   - `A_wing[q,μ]  = Σ_ν  W_body0[q,μ,ν] · χ_wing[q,ν]`
   - `A_wing'[q,ν] = Σ_μ  χ_wing'[q,μ] · W_body0[q,μ,ν]`
   - `χ_head_eff[q] = χ_head[q] + Σ_μν χ_wing'[q,μ]·W_body0[q,μ,ν]·χ_wing[q,ν]`
4. `W_head[q] = v_head[q] / (1 − v_head[q]·χ_head_eff[q])`
5. `W[q,μ,ν] = W_body0[q,μ,ν] + W_head[q]·(conj(g0)+A_wing)[q,μ]·(g0+A_wing')[q,ν]`

Setting χ_wing = χ_wing' = 0 before the rebuild reproduces BGW's `fixwings`
q=0 behavior — stated as "the planned use case" (docstring of
`reconstruct_W_via_schur_sharded`, lines 259–266).

### Sharding specs (module constants, lines 51–65)

| Constant | Value | Meaning |
|---|---|---|
| `V_FLATQ_SPEC` | `P(None, 'x', 'y')` | (nq, μ_X, μ_Y) — same as W_q flat-q form |
| `G0_X_SPEC` | `P(None, 'x')` | (nq, μ) sharded on x, replicated on y |
| `G0_Y_SPEC` | `P(None, 'y')` | (nq, μ) sharded on y, replicated on x |
| `SCALAR_Q_SPEC` | `P(None,)` | (nq,) replicated |
| `_RESHARD_MID_SPEC` | `P('x', None, 'y')` | intermediate for two-stage reshard, avoids XLA "Involuntary full rematerialization" |

Caller convention: every per-q vector (g0, χ_wing, χ_wing') is provided in
**two pre-replicated copies** — one sharded on x, one on y — so outer products
need no all-gather. All arrays are device-resident jax.Arrays; no host caches.

### Function table

| Function | Lines | Role | Physics / einsum | Callers (grep evidence) |
|---|---|---|---|---|
| `_reshard_W_to_flatq(W_q, mesh_xy)` | 68–81 | Two-stage `with_sharding_constraint` (`P('x',None,'y')` then `V_FLATQ_SPEC`) to reshard the body-solve output without XLA full remat | — | `solve_W_body0_sharded` (line 143) only |
| `extract_V_body_sharded(V_q, g0_x, g0_y, v_head_q, *, mesh_xy)` | 88–106 | Step 1: rank-1 head subtract, zero comm. `@partial(jax.jit, static_argnames=('mesh_xy',))` | `V_body[q,μ,ν] = V_q[q,μ,ν] − v_head[q]·conj(g0_x[q,μ])·g0_y[q,ν]` (broadcasted outer product, no einsum) | `reconstruct_W_via_schur_sharded` (line 267); `tests/test_head_wing_schur.py:65,271` |
| `solve_W_body0_sharded(V_body_q, chi_body_q, *, mesh_xy, pref=1.0+0j)` | 117–143 | Step 2: thin wrapper delegating to `w_isdf._get_w_solve_fn` (q-parallel shard_map LU), then forces output to V_FLATQ_SPEC via `_reshard_W_to_flatq` | `W_body0 = (1 − pref·V_body·χ_body)^{-1}·V_body`; `pref=2.0+0j` = spinor factor convention in production gw_jax, `1.0+0j` for synthetic tests | `reconstruct_W_via_schur_sharded` (line 268); `tests/test_head_wing_schur.py:66` |
| `schur_reductions_sharded(W_body0_q, chi_wing_x, chi_wing_y, chi_wingp_x, chi_wingp_y, chi_head_q, *, mesh_xy)` | 150–196 | Step 3: three reductions → (χ_head_eff, A_wing_x, A_wingp_y). One psum per contracted axis (all-reduce on y, on x, and on full mesh). jitted, static mesh. NO defensive reshard — caller contract requires V_FLATQ_SPEC input | einsums VERBATIM: `'qmn,qn->qm'` (A_wing_x, contracts ν/y); `'qm,qmn->qn'` (A_wingp_y, contracts μ/x); `'qm,qmn,qn->q'` (χ_head correction). `χ_head_eff = χ_head + χ_wing'·W_body0·χ_wing` | `reconstruct_W_via_schur_sharded` (line 269); `tests/test_head_wing_schur.py:67,305,318` |
| `W_head_scalar_per_q(v_head_q, chi_head_eff_q)` | 199–201 | Scalar head screening per q, replicated; plain function (not jitted) | `W_head[q] = v_head[q] / (1 − v_head[q]·χ_head_eff[q])` | `reconstruct_W_via_schur_sharded` (line 272); `tests/test_head_wing_schur.py:68` |
| `assemble_W_sharded(W_body0_q, g0_x, g0_y, A_wing_x, A_wingp_y, W_head_scalar_q, *, mesh_xy)` | 208–237 | Step 4: rank-1 rebuild, zero comm; jitted, static mesh; trusts sharding labels (no defensive reshard) | `W[q,μ,ν] = W_body0 + W_head[q]·(conj(g0_x)+A_wing_x)[q,μ]·(g0_y+A_wing')[q,ν]` (broadcasted outer product) | `reconstruct_W_via_schur_sharded` (line 273); `tests/test_head_wing_schur.py:69,285,295` |
| `reconstruct_W_via_schur_sharded(V_q, chi_body_q, g0_x, g0_y, chi_wing_x, chi_wing_y, chi_wingp_x, chi_wingp_y, chi_head_q, v_head_q, *, mesh_xy, pref=1.0+0j)` | 244–274 | End-to-end convenience driver chaining steps 1–4. Equivalent to production `solve_w(V_q, χ_q)` modulo the ability to override the wing convention (χ_wing=0 → BGW fixwings mimic) | Full pipeline above | `tests/test_head_wing_schur.py:70,205` ONLY — no production caller |

Note: `chi_wing_x` is accepted by `schur_reductions_sharded` and
`reconstruct_W_via_schur_sharded` but is **never contracted** inside
`schur_reductions_sharded` (only `chi_wing_y`, `chi_wingp_x` are used; the
x/y-copy convention means each vector is used from exactly one of its two
replicas). Similarly `chi_wingp_y` is passed but unused. This is consistent
with the "two pre-replicated copies" API but means half the wing arguments
per call are dead weight in this particular function.

### Cross-module deps

- `gw.w_isdf._get_w_solve_fn` (lazy import inside `solve_W_body0_sharded`,
  line 130: `from ..w_isdf import _get_w_solve_fn`) — reuses the production
  q-parallel shard_map LU solver at `src/gw/w_isdf.py:205`.
- `jax`, `jax.numpy`, `jax.sharding` (Mesh, NamedSharding, PartitionSpec),
  `jax.experimental.shard_map` (imported, unused — see weird_code).

### Entry points / callers (grep across src, tests, tools, scripts)

Grepped `head_wing_schur|extract_V_body_sharded|solve_W_body0_sharded|schur_reductions_sharded|assemble_W_sharded|reconstruct_W_via_schur_sharded|W_head_scalar_per_q|gw\.experimental` over
`src/`, `tests/`, `tools/`, `scripts/`:

- **Only caller anywhere: `tests/test_head_wing_schur.py`** (imports all six
  public functions at lines 64–70; correctness test at line 205; HLO
  collective-count checks at lines 271, 295, 318; smoke `python
  tests/test_head_wing_schur.py` entry at line 342).
- Zero imports from `src/gw/gw_jax.py`, `src/gw/w_isdf.py`, or any tool/script.
  The promised "separate driver patch" that plumbs this into static COHSEX has
  not landed in lorrax_D.

### Flags consumed

None. No `LorraxConfig` / `cohsex.in` keys are read; everything comes in as
arrays + `mesh_xy`. The `pref` convention (2.0 spinor factor) is passed by the
would-be caller, not read from config.

### I/O

None. Pure in-memory JAX kernel; no files read or written.

### dead_suspects

- `reconstruct_W_via_schur_sharded` and the entire module: zero production
  callers. Grep over src/tests/tools/scripts finds imports only in
  `tests/test_head_wing_schur.py` (plus egg-info SOURCES.txt listing).
  Per the `experimental/__init__.py` charter this is "staged, intended for
  promotion", not abandoned — but for the refactor map it is code with no
  production consumer. Whether it survives the refactor depends on whether the
  static-COHSEX head-correction driver patch is still planned.

### redundancy_suspects

- Declared mirror of `sources/lorrax_A/src/psp/finite_q_head_interp.py`
  (module docstring, lines 3–5): a parallel bordered-Sherman-Morrison
  implementation exists in the agent-A checkout. Cross-checkout duplication;
  the refactor should pick one canonical home.
- `solve_W_body0_sharded` is a thin wrapper around `w_isdf._get_w_solve_fn`
  whose only added value is the two-stage reshard; if promotion happens, the
  reshard fix arguably belongs inside `_get_w_solve_fn` itself (whose trailing
  `with_sharding_constraint(W, rep_3d)` "XLA may or may not honour", per the
  comment at lines 136–142) rather than as a parallel wrapper — exactly the
  "parallel old/new path" pattern the sandbox rules forbid.
- Functional overlap: `reconstruct_W_via_schur_sharded` computes the same W as
  production `solve_w(V_q, χ_q)` ("Equivalent to ``solve_w(V_q, χ_q)`` modulo
  the user's ability to override the wing convention", line 261) — two ways to
  build W will coexist if promoted as-is.

### weird_code

- **Unused import** `from jax.experimental.shard_map import shard_map`
  (line 41): `shard_map` never appears in the module body (all sharding is via
  `with_sharding_constraint` / delegation to w_isdf). Hypothesis: leftover
  from an earlier draft that used shard_map directly, or copied from the
  lorrax_A mirror.
- **Conjugation asymmetry** as convention: left vector is `conj(g0_x)`, right
  is bare `g0_y` (lines 103, 232) — i.e. rank-1 term is `g0* ⊗ g0`, Hermitian
  head channel. Matches the docstrings; flagged only so a reviewer doesn't
  "fix" one side.
- **Trust-the-label contract**: `schur_reductions_sharded` and
  `assemble_W_sharded` deliberately skip defensive resharding ("charges a
  16-all-to-all reshard cost on every call", lines 171–176, 227–231). Silent
  wrong-sharding inputs would produce a performance cliff or SPMD remat, not
  an error. Refactor should either keep the contract documented or add a
  debug-mode assert.
- **Magic reshard route** `_RESHARD_MID_SPEC = P('x', None, 'y')` (line 65):
  hand-tuned to make GSPMD plan two single-axis all_to_alls instead of
  "Involuntary full rematerialization". Same trick as in
  `w_isdf._get_w_solve_fn`; XLA-version-sensitive workaround duplicated in two
  places.
- **Half-unused wing arguments**: `chi_wing_x` and `chi_wingp_y` parameters of
  `schur_reductions_sharded` (lines 153, 156) are never referenced in the
  function body; only the y-copy of χ_wing and x-copy of χ_wing' contract.
  Hypothesis: kept for API symmetry with the "provide both replicas"
  convention, but as written they are dead parameters at this call level.
- **Stale docstring in the test**: `tests/test_head_wing_schur.py:1` says
  "Sharded test for ``gw.head_wing_schur``" while the module actually lives at
  `gw.experimental.head_wing_schur` — evidence the module was moved into
  `experimental/` after the test was written.
- `pref: complex = 1.0+0j` default differs from the production spinor
  convention `2.0+0j` (documented at lines 125–129); a promoted caller must
  remember to pass 2.0.

---

## File: `src/gw/experimental/__init__.py` (7 LOC)

### Purpose

Package marker for `gw.experimental` with a one-paragraph charter docstring:
modules here are "staged GW features that aren't yet wired into the production
``gw.gw_jax.main`` flow but pass their own targeted tests… real, tested, and
intended for promotion — not abandoned code. Move them up to ``gw/`` proper
when the production driver consumes them." No code, no imports, no `__all__`.

**Category guess:** infrastructure: package marker / staging-area charter.

### Functions

None (docstring only, lines 1–7).

### Entry points / callers

Imported implicitly whenever `gw.experimental.head_wing_schur` is imported —
i.e., only by `tests/test_head_wing_schur.py:64`. Listed in
`src/lorrax.egg-info/SOURCES.txt:213`.

### Flags / I/O / deps

None.

### dead_suspects

None per se (required for package importability), but the package currently
contains exactly one module with zero production callers; if
`head_wing_schur.py` is promoted or deleted during the refactor, the
`experimental/` package becomes empty and should go too.

### redundancy / weird_code

None.
