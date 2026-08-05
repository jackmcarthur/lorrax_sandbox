# Round 1 — Agent 3: Tighten the config/options seam (kill the PPMSigmaRuntimeOptions mirror)

**Lens**: the CONFIG/DATA seam. `gw_driver_helpers.PPMSigmaRuntimeOptions` is a drifted
parallel copy of `config.ppm`/`config.debug`, consumed via a `getattr(..., default)`
grab-bag in `ppm_sigma.compute_sigma_c_ppm_omega_grid`. The invalid_mode "3 modes
fighting" bug lived at exactly this seam.

**Question being answered**: what is the single-source-of-truth contract for
config → kernel? Kill the mirror and pass `config.ppm`/`config.debug` directly, or
slim to a validated resolved-bundle? What does jit-friendliness require, and how do
the two class-(c) carriers (`ppm_invalid_mode`, `sigma_at_dft_energies`) stay wired?

All paths below are relative to `sources/lorrax_D/src/gw/`.

---

## 1. The problem — a field census of the mirror (post-2A: 15 fields)

`PPMSigmaRuntimeOptions` (`gw_driver_helpers.py:16-34`) is built once by
`build_ppm_sigma_runtime_options` (`gw_driver_helpers.py:230-269`) from `config.ppm` +
`config.debug` + `input_dir`, threaded through `ppm_pipeline` into `ppm_sigma`, and
*also* stored in `PPMOutputs.ppm_options` (`ppm_pipeline.py:53`) so downstream code can
get the ω-grid back out. Exhaustive read census:

| # | field | reads | verdict |
|---|-------|-------|---------|
| 1 | `omega_grid_ry` | `ppm_sigma.py:1416`, `ppm_pipeline.py:141`, `sigma_dispatch.py:247`, `gw_jax.py:454` | **live — the only genuinely DERIVED array** |
| 2 | `omega_grid_ev` | `ppm_pipeline.py:201,262,273`, `sigma_dispatch.py:227,246`, `gw_jax.py:451` | **live — derived** |
| 3 | `sigma_regularization_ry` | `ppm_sigma.py:1435` (getattr, magic default `0.018374661…`) | live — trivial eV→Ry of `ppm.regularization_ev` |
| 4 | `sigma_edge_factor` | `ppm_sigma.py:1436` (getattr) | live — verbatim `ppm.window_edge_factor` |
| 5 | `sigma_omega_batch_size` | `ppm_sigma.py:1437` (getattr), `ppm_pipeline.py:276` | live — verbatim (+`max(1,·)`) |
| 6 | `sigma_omega_accumulation` | `ppm_sigma.py:1438` (getattr) | live — verbatim `ppm.omega_accumulation` |
| 7 | `sigma_kij_h5_path` | `ppm_sigma.py:1439` (getattr) | **live — genuinely RESOLVED** (`config.paths.sigma_kij_h5_file` + `input_dir`, `gw_driver_helpers.py:267`) |
| 8 | `fermi_reference` | `ppm_sigma.py:1440` (getattr) | live — verbatim `ppm.fermi_reference` |
| 9 | `ppm_invalid_mode` | `ppm_sigma.py:1441` (getattr) | live — **class-c carrier**, verbatim `ppm.invalid_mode` |
| 10 | `omega_p_ry` | none — `ppm_pipeline.py:335-338` reads `config.ppm.omega_p` directly | **dead mirror** |
| 11 | `ppm_fallback` | none — `ppm_pipeline.py:344` reads `config.ppm.fallback_omega` directly | **dead mirror** |
| 12 | `sigma_freq_debug_output` | none — `gw_jax.py:817` reads `config.debug` directly | **dead mirror** |
| 13 | `sigma_freq_debug_file` | none — `gw_jax.py:920` reads `config.debug` directly | **dead mirror** (its `_resolve_input_path` at helper :268 is dead too — gw_jax uses the *unresolved* config string; a latent cwd-vs-input_dir inconsistency the collapse should settle) |
| 14 | `sigma_at_dft_extrapolate` | none — `gw_jax.py:673` reads `config.ppm` directly | **dead mirror** |
| 15 | `sigma_at_dft_energies` | none anywhere | **dead mirror of an orphaned class-c flag** (config carrier `gw_config.py:572` also unread) |

So: 6/15 fields are dead copies of values the code already reads straight off
`config`, 7/15 are verbatim (or trivially unit-converted) relays of `config.ppm`
scalars, and only **2 are genuinely resolved values**: the derived ω-grid and the
input_dir-resolved h5 path. The mirror earns its 60 lines for exactly two values.

### 1a. The consumption side is a loose contract by construction

`ppm_sigma.py:1435-1441` reads every scalar via `getattr(ppm_options, 'x', default)`.
Consequences, concretely:

- A typo'd or *removed* field silently falls back to a hardcoded default. This is
  the invalid_mode mechanism: the config default said one thing, the fit baked
  another, and the kernel's `getattr(..., 'zero')` default meant that whether the
  config value even *arrived* was unobservable — three layers each with their own
  default, innermost silently winning (SIGMA_PPM_MAP.md §1, §2C).
- The magic literal at `:1435` (`0.018374661087827496` = 0.25 eV in Ry) is a
  *fourth* copy of `_DEFAULTS["ppm_sigma_regularization"]` — defaults duplicated at
  the read site drift independently of `gw_config._DEFAULTS`.
- Same smell one line up: `getattr(ppm, 'valid_mask_q', None)` at `ppm_sigma.py:1415`
  is pure defensive cruft — `PPMBuildResult.valid_mask_q` is a declared field
  (`ppm_sigma.py:138`) and `fit_ppm` always sets it (`:733`). Delete the getattr.

### 1b. Bonus finding: TWO parallel ω-grid derivations, one dead, numerically different

`LorraxConfig.omega_grid_ry` / `omega_grid_ev` **properties already exist**
(`gw_config.py:758-775`) — and have **zero readers** (the only `.omega_grid_*` reads
are off `ppm_options`/`SigmaResult`). Worse, they use a different formula:

- properties: `np.arange(min, max + 0.5·step, step)` — computed **independently** in
  eV and in Ry (float-step arange; the two can disagree in length near the endpoint,
  and the Ry one accumulates different rounding than the eV one).
- live builder: `n = floor((max−min)/step + 0.5) + 1`, grid built in eV, Ry derived
  by division (`gw_driver_helpers.py:244-250`) — length-stable, ev/ry always congruent.

Single source of truth demands ONE derivation. Keep the builder's formula, move it
into the config property, delete the arange version.

---

## 2. The two options, weighed

### Option 1 — kill the mirror, pass `config.ppm` (+ derived grid + resolved path) directly

Kernel signature carries `ppm_cfg: PPMConfig` and reads `ppm_cfg.window_edge_factor`
etc. by direct attribute access.

**Pros**
- Zero new types; deletes the dataclass + builder (~60 L). Trivially single-source:
  there is no second object to drift. Matches the standing "no new API layers" and
  "no parallel paths" rules.
- Direct attribute access on a frozen dataclass **raises `AttributeError` on a
  stale/typo'd name** — the exact failure mode `getattr(..., default)` silenced.
- `PPMConfig` is all-scalars frozen (`gw_config.py:541-572`) ⇒ **hashable** ⇒ usable
  as a jit `static_argnums` arg if we ever want that (see §4). The current mirror is
  frozen-but-unhashable because it embeds `np.ndarray` fields — structurally
  disqualified from static-arg use, which tells you it was never the right shape.
- Validation can live in one place for ALL consumers: `PPMConfig` construction.

**Cons**
- The kernel signature no longer *declares* which of PPMConfig's 18 fields it reads —
  you must read the body to know the dependency set. (Mitigated: reads fail loudly,
  and the read block is 6 contiguous lines at the top of the driver.)
- `ppm_sigma` (math module) grows an import of a `gw_config` type. Layering-wise
  acceptable — `PPMConfig` is a leaf frozen dataclass, and `ppm_sigma` already
  imports the sibling `SigmaQuadratureConfig` from `minimax_config` — but it must
  import *only* `PPMConfig`, never `LorraxConfig`.
- Unit tests calling the kernel must build an 18-field `PPMConfig`. Mitigated by
  `dataclasses.replace` on a parsed-defaults instance; no helper class needed.

### Option 2 — slim validated resolved-bundle (explicit fields, no getattr)

Shrink the mirror to the 8-9 actually-read fields + the grid, keep one `resolve_*`
constructor that validates.

**Pros**: the kernel's dependency set is the dataclass definition; no `gw_config`
import in `ppm_sigma`; tests construct a 9-field bundle.

**Cons — and these are decisive**:
- It is **still a mirror**. The drift mechanism that produced 6 dead fields is the
  copy step itself, not the field count. Next flag added to `config.ppm` gets added
  to the bundle "for later" and rots identically.
- It keeps a parallel name-space (`sigma_edge_factor` vs `window_edge_factor`,
  `ppm_fallback` vs `fallback_omega` — the renames are themselves drift surface).
- Mixing the grid arrays into the frozen bundle keeps it unhashable (the current
  defect); keeping the grid out means the bundle is *just* PPMConfig-minus-10-fields,
  i.e. a worse PPMConfig.
- "No new API layers" / "no redundancy in refactors" both push against.

### Verdict

**Option 1, with two amendments** that capture Option 2's real virtues (validation,
explicitness) without the copy:

1. **Validation moves to config construction** — `PPMConfig.__post_init__` — not a
   resolver, not the kernel.
2. **The two genuinely-derived values travel as explicit kernel arguments**
   (`omega_grid_ry: np.ndarray`, `sigma_kij_h5_path: str | None`), because they are
   *data*, not config: one is an array (must not ride inside a hashable static), the
   other needs `input_dir` (a driver concern the math module must not know about).

---

## 3. The concrete proposal

### 3a. The contract (single source of truth, stated)

> **`config.ppm` is the only home for Σ_PPM scalar knobs. Values are validated once,
> at `PPMConfig` construction. Derived values (ω-grid) are `LorraxConfig` properties
> with exactly one formula. Resolved values (paths) are resolved at the driver seam
> (`ppm_pipeline`) where `input_dir` lives, and passed as explicit args. The kernel
> reads scalars by direct attribute access — `getattr(x, 'field', default)` on a
> config-like object is banned in `gw/`. No object may copy a `config` field it does
> not itself derive or resolve.**

### 3b. `PPMConfig.__post_init__` (gw_config.py:541)

Move the three checks from `gw_driver_helpers.py:237-242` plus the value-set checks
currently at `ppm_sigma.py:1450-1482` (values only — capability gating stays in the
kernel, see 3e):

```python
def __post_init__(self):
    if self.omega_step_ev <= 0.0:
        raise ValueError("ppm.omega_step_ev must be > 0.")
    if self.omega_max_ev < self.omega_min_ev:
        raise ValueError("ppm.omega_max_ev must be >= ppm.omega_min_ev.")
    if self.fermi_reference not in ("vbm", "midgap"):
        raise ValueError("ppm.fermi_reference must be 'vbm' or 'midgap'.")
    if self.omega_accumulation not in ("auto", "kij", "kij_stream"):
        raise ValueError("ppm.omega_accumulation must be auto/kij/kij_stream.")
    if self.invalid_mode not in ("zero", "skip", "2ry", "static_limit", "infinity", "imaginary"):
        raise ValueError(f"ppm.invalid_mode: unknown value {self.invalid_mode!r}")
    if self.omega_batch_size < 1:
        raise ValueError("ppm.omega_batch_size must be >= 1.")
```

(The `.strip().lower()` normalizations move to the parse site in `from_input_file`
(`gw_config.py:~915-931`) so the stored values are canonical — one normalization,
not one per reader.)

### 3c. One ω-grid derivation (gw_config.py:758-775)

Replace the dead arange properties with the live builder's formula, Ry derived from
eV so the pair can never disagree:

```python
@property
def omega_grid_ev(self) -> np.ndarray:
    p = self.ppm
    n = int(np.floor((p.omega_max_ev - p.omega_min_ev) / p.omega_step_ev + 0.5)) + 1
    return p.omega_min_ev + p.omega_step_ev * np.arange(n, dtype=np.float64)

@property
def omega_grid_ry(self) -> np.ndarray:
    return self.omega_grid_ev / RYD_TO_EV
```

Every downstream `ppm_outputs.ppm_options.omega_grid_*` read (`sigma_dispatch.py:227,
246-247`, `gw_jax.py:451-455`) becomes `config.omega_grid_*` — both call sites already
hold `config`. `PPMOutputs.ppm_options` (`ppm_pipeline.py:53`) is deleted.

### 3d. Target signature of `compute_sigma_c_ppm_omega_grid` (ppm_sigma.py:1397)

```python
def compute_sigma_c_ppm_omega_grid(
    wfns,
    ppm: PPMBuildResult,               # poles: B_q, Omega_q, valid_mask_q (direct reads, no getattr)
    meta,
    mesh_xy: Mesh,
    *,
    ppm_cfg: PPMConfig,                # validated frozen scalars — the ONLY config object
    quad: SigmaQuadratureConfig,       # REQUIRED (delete the None-fallback defaults at :1430-1433)
    omega_grid_ry: np.ndarray,         # derived data — host np.float64, 1-D (config.omega_grid_ry)
    sigma_kij_h5_path: str | None,     # resolved data — input_dir-resolved by the caller; None = no stream
    print_fn=print,
) -> SigmaOmegaResult:
```

Inside, the getattr block `:1435-1441` becomes direct reads (fail-loud):

```python
regularization_width_ry = float(ppm_cfg.regularization_ev) / RYD_TO_EV   # kills the magic literal
edge_factor        = float(ppm_cfg.window_edge_factor)
omega_batch_size   = int(ppm_cfg.omega_batch_size)
omega_accumulation = ppm_cfg.omega_accumulation      # already validated + normalized
fermi_reference    = ppm_cfg.fermi_reference
invalid_mode       = ppm_cfg.invalid_mode
```

and `:1415` becomes `valid_mask_q = ppm.valid_mask_q`. The re-validation at
`:1449-1452` and `:1461-1463` is deleted (done at construction); the ω-array shape
check at `:1447-1448` stays (it guards the *argument*, not the config).

Caller (`ppm_pipeline.py:353-357`):

```python
sigma_omega = compute_sigma_c_ppm_omega_grid(
    wfns, ppm, meta, mesh_xy,
    ppm_cfg=config.ppm,
    quad=config.sigma_quadrature_config,
    omega_grid_ry=config.omega_grid_ry,
    sigma_kij_h5_path=_resolve_input_path(input_dir,
        str(config.paths.sigma_kij_h5_file or "").strip()) or None,
    print_fn=print_fn,
)
```

Note the trailing `or None`: today `""` flows through and is caught by falsy checks
at `ppm_sigma.py:123` — normalize to `None` at the seam so the kernel contract is
`str | None`, not "str, possibly empty, treated as falsy in two places".

`ppm_pipeline`'s own helpers (`_inject_analytic_head:109`,
`_eval_sigma_c_at_dft_energies:166`, `_write_sigma_omega_h5:236`) swap their
`ppm_options: PPMSigmaRuntimeOptions` params for what they actually use: the first
two take `omega_grid_ry`/`omega_grid_ev` (or just `config`, which they can already
reach), the writer takes `config` (it already does) and reads
`config.omega_grid_ev` / `config.ppm.omega_batch_size`.

### 3e. The two class-(c) carriers stay wired

- **`ppm_invalid_mode`** — travels as `ppm_cfg.invalid_mode`. *Value* validity is
  `__post_init__`'s job (3b); *capability* gating (`static_limit`/`infinity` →
  `NotImplementedError` until 2C's `−½·Wc0` term + `Wc0` retention land;
  `imaginary` → unsupported) **stays in the kernel** at `:1470-1482`, because
  capability is a property of the kernel, not of the input. The `keep_invalid`
  bool → `_prepare_sigma_state` wiring (`:1483`, `:1492`, consumed at `:346`) is
  unchanged. When 2C implements `static_limit`, the extra ingredient is `Wc0` on
  `PPMBuildResult` — a **data-seam** change (fit output), not a config field.
- **`sigma_at_dft_energies`** — was never a kernel concern; the mirror copy was
  never read, so deleting the mirror is a no-op for it. Its re-wiring point is the
  QP-solve dispatch at `gw_jax.py:649-695`, read directly as
  `config.ppm.sigma_at_dft_energies` — the *identical pattern* its sibling
  `sigma_at_dft_extrapolate` already uses at `gw_jax.py:673`. (`True` → the
  at-DFT interp `sigma_c_at_dft_ev` becomes the authoritative E_qp; `False` →
  fixed-point solve. Physics change; own commit + gate, per SIGMA_PPM_MAP §2C.)

### 3f. Also deleted

- `PPMSigmaRuntimeOptions` + `build_ppm_sigma_runtime_options`
  (`gw_driver_helpers.py:16-34, 230-269`) and the imports at `ppm_pipeline.py:30-34`.
- The stale docstring reference at `gw_config.py:25`.
- The mirror-resolved-but-unread `sigma_freq_debug_file` path resolution
  (`gw_driver_helpers.py:268`). Decision to settle in discussion: `gw_jax.py:920`
  currently writes the debug table to the **unresolved** config string (cwd-relative).
  Either resolve it at `gw_jax.py:920` with `input_dir`, or declare cwd-relative
  intended; don't keep both behaviors latent.

---

## 4. What jit-friendliness requires (and how the proposal satisfies it)

`compute_sigma_c_ppm_omega_grid` is a host-side driver; the jit boundary is below it
(`_prepare_sigma_state:307`, the τ-kernels, the projectors). The seam must deliver:

1. **Plain hashable Python scalars for anything that shapes a trace** — window
   construction (`target_error`, `max_nodes`, `edge_factor`, batch size) happens
   host-side and keys kernel caches. `PPMConfig`/`SigmaQuadratureConfig` are frozen,
   all-scalar ⇒ hashable ⇒ even directly usable as `static_argnums` if a future
   refactor jits higher. The old mirror could never be (ndarray fields ⇒ unhashable).
   Guard this property with a one-line test: `hash(config.ppm)` must not raise.
2. **Mode strings must not fragment the jit cache** — the existing pattern converts
   them to *traced* bools before entering the jit
   (`jnp.asarray(fermi_reference == "midgap")`, `jnp.asarray(keep_invalid)` at
   `:1491-1492`, documented at `:320`). Preserved verbatim; the seam change only
   alters where the strings come from.
3. **The ω-grid enters as host `np.float64` and is split/converted host-side**
   (`:1446`, `:1455-1458`) — which is exactly why it must be a standalone array
   argument, not a field of a would-be-static bundle.
4. **No config object below the driver** — `ppm_cfg` is unpacked into locals at the
   top of `compute_sigma_c_ppm_omega_grid`; nothing config-shaped is threaded into
   `_run_sigma_branch`/`_iter_branches` (`:1571-1608` already passes scalars). This
   keeps the math kernels import-clean of `gw_config` below the one driver function.

---

## 5. Migration path + gate strategy

Feature branch `agent/ppm-config-seam` (lorrax_D). Five commits, each pytest-green;
the collapse itself is pure plumbing so the end-to-end gate is **bit-identical
output**, not tolerance-pass.

| step | change | gate |
|------|--------|------|
| 0 | Capture baselines: golden gates `test_gw_jax_matches_reference[cohsex]`, `[gnppm]`, `test_ibz_full_bz_equivalence` + one MoS2 GN-PPM 1-GPU run; stash `sigma_mnk.h5` + `eqp*.dat` | baseline green |
| 1 | `PPMConfig.__post_init__` (3b) + normalize-at-parse; fix `config.omega_grid_*` properties (3c); assert new property == old builder grid (`n_omega` + allclose) in a unit test | pytest + the assert |
| 2 | Kernel signature swap (3d): direct reads replace getattr `:1435-1441`, `:1415`; delete re-validation `:1449-1463`; make `quad` required, delete `:1430-1433` fallback | pytest; `sigma_mnk.h5` **bit-identical** to step-0 stash |
| 3 | `ppm_pipeline`: drop builder call (`:325`), swap helper params, delete `PPMOutputs.ppm_options`; `sigma_dispatch.py:227,246-247` + `gw_jax.py:451-455` → `config.omega_grid_*` | pytest; golden gates; h5 bit-identical |
| 4 | Delete `PPMSigmaRuntimeOptions` + builder + imports + `gw_config.py:25` docstring; repo-wide grep proves 0 refs | pytest; `grep -rn PPMSigmaRuntimeOptions src/` empty |
| 5 | (separable, physics) wire `sigma_at_dft_energies` at `gw_jax.py:649` per 3e | new unit gate: flag flips eqp source; golden gates unchanged with flag off |

Steps 1-4 must not change a single output bit (the h5 diff is the gate; grid-formula
step 1 is safe because the property version had zero readers — the live formula is
adopted verbatim). Step 5 is deliberately last and separable: it is the only
behavior change, and it defaults off.

Ordering vs the rest of the tighten effort: this seam lands **before** the file
split (the new signature is the boundary the split will cut along) and **before**
2C's `static_limit`/2D's bug fixes (both want a trustworthy single-path
`invalid_mode` and an explicit head/grid seam to build on). It is independent of
2A's remaining dead-code deletes.

## 6. Risks

- **Silent-default removal changes behavior for out-of-tree callers** passing a
  partial options object — getattr defaulted, direct reads raise. Repo-wide there is
  exactly one caller (`ppm_pipeline.py:353`) and zero test constructions of the
  mirror; risk is nil in-tree, and fail-loud is the point.
- **`""` vs `None` for `sigma_kij_h5_path`**: today `""` reaches
  `_select_accum_mode` and is handled by falsiness (`ppm_sigma.py:118,123`). The
  `or None` normalization (3d) must land with a streamed-mode smoke test
  (`omega_accumulation=kij_stream` on the gnppm regression input) so KIJ_STREAM
  selection is exercised, not just the in-memory path.
- **ω-grid formula unification**: adopting the builder formula into the property is
  provably safe only because the property had 0 readers — verified by grep (§1b);
  the step-1 equality assert makes it mechanical.
- **`__post_init__` on a dataclass constructed everywhere config round-trips**
  (SC restarts, tests with hand-built configs): any test that previously built an
  *invalid* PPMConfig (e.g. `omega_step_ev=0` placeholder) now throws at
  construction instead of at pipeline entry. Fix such tests to use valid defaults —
  that's the feature working.
- **Frozen-dataclass `__post_init__` + `from_input_file` normalization ordering**:
  normalize (`strip().lower()`) must happen *before* construction (parse site), since
  frozen fields can't be rewritten in `__post_init__` without `object.__setattr__`.
  Keep it at the parse site; `__post_init__` only checks.

## 7. Interaction with the other three lenses

- **File-split lens (ppm_sigma.py ~5 concerns)**: direct dependency, sequencing not
  content. The split should cut along the *new* signature — the driver
  (`compute_sigma_c_ppm_omega_grid` + accumulators, `:1393-1631`) is the only code
  that may import `PPMConfig`; windows/τ-kernels/state modules receive plain scalars
  as they already do (`:1571-1588`). Land this seam first so the split doesn't have
  to move the getattr grab-bag and then re-fix it. If that agent instead proposes
  relocating the driver into `ppm_pipeline`, the target signature is unchanged —
  only its home moves. No conflict.
- **τ-kernel readability lens (_SigmaBranch signs/masks/windows)**: near-zero overlap
  by design — nothing below `_iter_branches`/`_run_sigma_branch` sees a config
  object, before or after. Shared touchpoint: `quad` becomes required (3d), so any
  re-derivation of window construction can assume `SigmaQuadratureConfig` is always
  present — one code path, not two. Also flag: `MinimaxConfig` vs
  `SigmaQuadratureConfig` (`minimax_config.py:9,23`) duplicate
  `target_error`/`max_nodes`/`use_shipped_tables`; if that agent wants to merge
  them, my seam only pins *how the kernel receives it* (one required hashable
  object), not its internal shape.
- **Physics/bugs lens (2C invalid_mode, 2D head bugs)**: strongest coupling, one
  boundary to agree on: **value-vs-capability** for `invalid_mode` (3e) — values
  validated at config construction, capability errors in the kernel; when they
  implement `static_limit`, `Wc0` goes on `PPMBuildResult` (data seam), NOT into
  config or any options object. For Bug B (streamed head drop,
  `ppm_pipeline.py:126-127`): my step-3 rewiring touches `_inject_analytic_head`'s
  signature — whoever fixes Bug B should rebase on the new explicit
  `omega_grid_ry` + stream-path args (both ingredients the fix needs are then
  explicit parameters instead of buried in the mirror). Coordinate the two edits to
  avoid a merge fight over the same 40 lines.
- **Potential conflict to settle in discussion**: if another lens proposes a *new*
  "resolved runtime bundle" for their concern (e.g. a windows-plan object), the
  contract in 3a is the tie-breaker — bundles may carry only what they **derive**;
  anything readable off `config` travels as `config.ppm` or a scalar.
