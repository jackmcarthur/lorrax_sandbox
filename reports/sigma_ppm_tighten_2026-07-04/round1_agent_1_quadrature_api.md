# Round 1 — Agent 1: Unify the quadrature / minimax backend (W chi(omega) <-> Sigma tau)

**Lens**: the quadrature / minimax API. Can W's chi0(ω)/screening evaluation and Σ's
τ-window integration share one config, one node abstraction, one evaluation backend —
and where do they *genuinely* diverge?

**Code state read**: `sources/lorrax_D` @ `3cad3dd` (2A dead-delete + invalid_mode wiring
+ head sign-flip fix already landed). Files: `gw/minimax_screening.py` (935 L),
`gw/w_isdf.py` (690 L), `gw/ppm_sigma.py` (1631 L), `gw/minimax_config.py` (36 L),
`gw/gw_config.py` (:737-757), `gw/screening.py`, `gw/ppm_pipeline.py`,
`gw/head_correction.py`, `common/minimax.py` (1049 L, the numerical solver).

**Verdict up front**: the lead's seed idea is ~70% already true — `minimax_screening.py`
*is* the single engine and `MinimaxNodes` *is* the single node currency, by deliberate
design (`minimax_screening.py:290-305` docstring says exactly this). What has rotted is
everything **around** the engine: a dead vestigial "window pair" layer only W touches,
**three** copies of the Σ quadrature defaults, **three** different alpha-folding
conventions scattered across call sites, **two** independent implementations of the
two-point pole-fit algebra (the scalar copy is where Bug A lived), and a shipped-table /
`regenerate_tables` honor matrix with silent holes. The unification job is mostly
*deletion + one accessor*, not a new abstraction — which is exactly what the repo's
no-new-API-layers discipline wants.

---

## 1. Inventory — what is ALREADY shared (don't re-invent it)

The stack, bottom to top:

| Layer | Where | Shared? |
|---|---|---|
| Remez/varpro solvers: `noncrossing_grids`, `noncrossing_imag_grids`, `crossing_grids` (+ `G_hgl`/`G_fermi` targets) | `common/minimax.py:205,562` and crossing solver | ✅ one engine |
| Cache front-ends (lru + disk npz + shipped-table catalog): `solve_laplace_minimax_interval` (:560), `solve_laplace_minimax_imag_interval` (:618), `solve_phase_minimax_bandwidth` (:662) | `gw/minimax_screening.py` | ✅ one engine |
| Quadrature value objects: `LaplaceMinimaxQuadrature` (:352), `CrossingMinimaxQuadrature` (:377) | `gw/minimax_screening.py` | ✅ shared |
| Node currency: `MinimaxNodes` complex-(t, α) pytree with `time_axis ∈ {'real','imag','crossing_hgl'}` conversions (:290-349, :366-374, :391-394) | `gw/minimax_screening.py` | ✅ shared **by design** — the :293-298 docstring states both chi0's real-τ Laplace and sigma's `-1j·τ` / `τ/ξ` quads live in one complex128 storage "so one sibling function shape handles both pipelines" |
| Two-point PPM pole fit (tensor): `fit_gn_ppm_from_wc_pair` (:408) | `gw/minimax_screening.py` | ✅ shared by W→Σ seam (`ppm_sigma.fit_ppm:706`) |

Who consumes which node **kind**:

| Node kind | Solver family | Shipped tables? | W consumers | Σ consumers |
|---|---|---|---|---|
| Laplace `1/x` | `noncrossing` | **yes** | static chi0 (`w_isdf.py:457`), HL real-freq ±branches (`w_isdf.py:521,532`) | `single` window (`ppm_sigma.py:767`), `a_stripe`/`b_slab` (`ppm_sigma.py:861`) |
| Laplace-imag `x/(x²+ω²)` | `noncrossing_imag` | **no** (see §2.5) | chi0(iω_p) probe (`w_isdf.py:468`) | — |
| Crossing `sin`-phase (HGL target) | `crossing` | yes | — | `core` window (`ppm_sigma.py:843`) |

And the two τ-integrators are **documented siblings**: `w_isdf.minimax_tau_integrate_chi`
(:144, `lax.scan` inside one jit — body emits no collective) and
`ppm_sigma.minimax_tau_integrate_sigma` (:1169, Python τ loop — body emits NCCL; a
monolithic scan regressed MoS2 3×3 by ~80%, per the :1180-1183 docstring and the
reverted-experiment note at `ppm_sigma.py:466-470`). Both take a `MinimaxNodes` in the
same slot. **This is the correct end state already** — the loop mechanics must NOT be
merged (§5).

So the "one evaluation backend" the lead asks for exists. The unification work is in the
five duplications below.

## 2. What is DUPLICATED or divergent (the rot)

### 2.1 Two configs, and THREE copies of the Σ defaults

`MinimaxConfig` and `SigmaQuadratureConfig` (`minimax_config.py:8-34`) share 3 fields
(`target_error`, `max_nodes`, `regenerate_tables`) and a copy-pasted
`use_shipped_tables` property; they differ by `energy_reference` (W-only) and
`crossing_max_nodes`/`crossing_eps_q` (Σ-only). Two dataclasses for one concept.

Worse, the Σ-side default values exist at **three sites** that can drift independently:

1. `minimax_config.py:27-29` — dataclass defaults (`64`, `500`, `1e-3`).
2. `gw_config.py:751-757` — the `sigma_quadrature_config` property re-hardcodes
   `crossing_max_nodes=max(500, sigma_max_nodes)` and `crossing_eps_q=1.0e-3`.
3. `ppm_sigma.py:1430-1433` — `compute_sigma_c_ppm_omega_grid` has a
   `sigma_window_quad is None` fallback that hardcodes `1e-6, 64, 500, 1e-3` a third
   time. Every production caller passes the config (`ppm_pipeline.py:355`), so this
   branch is a drift trap serving nobody.

Also: `ppm_sigma.py:68` imports `MinimaxConfig` and never uses it (dead import,
post-2A leftover).

### 2.2 A dead vestigial layer W is still paying for: `MinimaxWindowPair`

`build_static_minimax_window_pair` (`minimax_screening.py:715-772`) returns
`([MinimaxWindowPair], LaplaceMinimaxQuadrature)` — and **both** call sites discard the
window-pair list: `w_isdf.py:457` `_, quad = build_static_minimax_window_pair(...)`.
Repo-wide zero readers of the pair. That drags dead with it:

- `MinimaxWindowPair` (:258-287) incl. `with_imag_freq_modulation` (:269) — 0 callers,
  and its cos-reweighting is the *documented-incorrect* approach that
  `build_imag_freq_minimax_window_pair`'s docstring (:788) says was replaced.
- `EnergyWindow` (:243-254) — only feeds `MinimaxWindowPair`.
- `build_imag_freq_minimax_window_pair` (:775-835) — 0 callers (superseded by
  `w_isdf.build_imag_quadrature`).
- `extract_gn_ppm_parameters` (:838-896) and `extract_gn_ppm_parameters_from_Wc`
  (:899-935) — 0 callers (superseded by `ppm_sigma.fit_ppm` → `fit_gn_ppm_from_wc_pair`).
  The first one even contains its own host-side `Π = χ(I−Vχ)⁻¹` solve loop — a
  duplicate W-solve in numpy hiding in the quadrature module.
- The `w_kernel = quad.alpha * np.exp(-quad.tau)` fold at :753 and :814 is dead *work*
  computed to fill the discarded object.

That's ~200 L of the 935 L module. Deleting it makes the real currency —
`LaplaceMinimaxQuadrature` in, `MinimaxNodes` out — visible.

### 2.3 Three alpha-folding conventions at three call sites

The same idea — "fold a reference shift / prefactor into the weights so the kernel sees
small non-negative exponents" — is implemented three different ways:

1. **chi0, build time**: `compute_chi0` (`w_isdf.py:597-605`) computes
   `alpha_chi = -2·α·exp(-τ·E_gap)` and then constructs `MinimaxNodes` **by hand**
   (`jnp.asarray(tau/alpha_chi, complex128)`), bypassing
   `quad.to_minimax_nodes(time_axis='real')` entirely. The one place that motivated the
   `time_axis='real'` branch of `_laplace_to_minimax_nodes` (:321-322) doesn't call it.
2. **Σ Laplace, integrate time**: `minimax_tau_integrate_sigma` (`ppm_sigma.py:1210-1212`)
   folds `α_eff = α·exp(-i·E_ref_sum·t)` per window on host at loop entry
   (`E_ref_sum = E_ref_A + E_ref_B`, :1284).
3. **Σ crossing, window-build time**: `_build_three_sigma_windows`
   (`ppm_sigma.py:851-853`) rescales `MinimaxNodes(t=raw.t/ξ, alpha=raw.alpha/ξ)` by
   hand after `to_minimax_nodes(time_axis='crossing_hgl')`.

None of these is wrong; but a reader auditing sign conventions (the Σc(−ω)=−Σc(ω)*
decomposition lens) has to find three fold sites in two files to know what `alpha`
means at kernel entry. One accessor fixes this (§3.2).

### 2.4 TWO implementations of the two-point pole-fit algebra — Bug A lived in the copy

The GN/HL two-point fit `Ω² = −z²·Wc(z)/(Wc(0)−Wc(z))`, `B ∝ −Wc(0)·Ω` exists twice:

- **Tensor**: `fit_gn_ppm_from_wc_pair` (`minimax_screening.py:408-457`) — elementwise on
  (nq, μ, μ); invalid policy = `good` mask + `fallback_omega` substitution;
  `B = −½·Wc0·Ω`.
- **Scalar (head)**: `head_correction.fit_head_ppm` (:280-351) — same algebra re-derived
  for the q→0 head sample; invalid policy = *continuation* branch (`Ω_h = |Ω²|^½`, keep);
  amplitude carried as `B_h = −w1·Ω_h²` with `R_h = B_h/(2Ω_h)` — i.e. the scalar `R_h`
  equals the tensor `B` up to the normalization `B_h = 2Ω·B_tensor`.

Consequences of the duplication, both already observed:
1. **Bug A** (head sign flip, fixed 2026-07-04 at `head_correction.py:320-327`) could
   only exist because the negative-Ω² handling was re-implemented instead of shared —
   the tensor fitter never had the bug (its `omega_vals` only uses `omega_sq_re` where
   `good`, :454).
2. **Policy skew**: the body drops/fallbacks invalid poles (`ppm_invalid_mode`,
   `_prepare_sigma_state` `ppm_sigma.py:344-349`) while the head *continues through*
   Ω²<0 with a different formula. Nobody has stated whether that asymmetry is intended;
   it is invisible today because the policies live in different files with different
   variable names.

### 2.5 Shipped-table / `regenerate_tables` honor matrix has silent holes

`MinimaxConfig.regenerate_tables` → `use_shipped_tables` is honored **only** on the
static path (`build_static_minimax_window_pair:727-730` unpacks the config). The other
two W builders take `minimax_config` as an argument and then drop the flag:

- `build_imag_quadrature` (`w_isdf.py:462-478`) → `solve_laplace_minimax_imag_interval`,
  which has **no `use_shipped_tables` parameter at all** (`minimax_screening.py:618-625`)
  — there is no shipped `noncrossing_imag` family, so every GN probe quad is a live
  solve (disk-cached). Fine as a fact, but the asymmetry is undocumented.
- `build_real_quadrature` (`w_isdf.py:521-535`) calls `solve_laplace_minimax_interval`
  **without** `use_shipped_tables=` → defaults `True`. So
  `regenerate_minimax_tables = true` silently does not regenerate the HL-PPM ±branch
  quads. A user regenerating tables to rule out a stale-table artifact on an HL run is
  being lied to. Σ's path honors it everywhere
  (`ppm_sigma.py:1429`, threaded to :767/:848/:861).

### 2.6 Energy-reference conventions: three, loosely coupled

- W: `resolve_minimax_energy_reference` (`w_isdf.py:382-418`, midgap/vbm/cbm/float,
  from `screening.minimax_energy_reference`) — algebraically neutral for chi0, kept
  "synchronized with sigma paths" per its own docstring.
- Σ: `fermi_reference` (vbm/midgap) resolved **inside the jit**
  `_prepare_sigma_state` (`ppm_sigma.py:333-337`) from `ppm.fermi_reference`.
- Head: `wfn.efermi` (canonical WFN-load midgap) at `ppm_pipeline.py:136`.

These are physically distinct references (chi0's is a neutral shift; Σ's defines the
cond/val split; the head reuses the WFN Fermi level) — I do **not** propose merging their
values. But the *vocabulary* (`"midgap"`) and the resolve helper should be one function,
so that a future audit can see at a glance which reference feeds which stage.

## 3. The concrete proposal — target API

Guiding constraints (repo memory): no new classes/wrappers where an accessor on an
existing object does the job; consolidate by deleting parallel routines; single source of
truth. The target is **one config class (two instances), one node accessor, one pole-fit
routine, zero dead layers** — not a `QuadratureBackend` object.

### 3.1 One config class, two instances

Replace both dataclasses in `minimax_config.py` with a single frozen `MinimaxConfig`:

```python
@dataclass(frozen=True)
class MinimaxConfig:
    """Minimax quadrature controls. One class; W and Σ hold separate instances
    because their accuracy knobs are independently user-tunable."""
    target_error: float = 1.0e-6
    max_nodes: int = 64
    regenerate_tables: bool = False
    energy_reference: str | float | None = "midgap"   # consumed by W only
    crossing_max_nodes: int = 500                     # consumed by Σ only
    crossing_eps_q: float = 1.0e-3                    # consumed by Σ only

    @property
    def use_shipped_tables(self) -> bool:
        return not self.regenerate_tables
```

- `gw_config.minimax_config` (:737) and `gw_config.sigma_quadrature_config` (:748) keep
  their names and knob sources (`screening.minimax_*` vs `ppm.sigma_*` — these are
  *deliberately* independent accuracy budgets; do not collapse the user knobs) but both
  return the one class. The `max(500, sigma_max_nodes)` and `1.0e-3` at
  `gw_config.py:754-755` move into the dataclass default / a documented derivation —
  **one** default site.
- Delete the `sigma_window_quad=None` fallback block at `ppm_sigma.py:1424-1433`:
  `compute_sigma_c_ppm_omega_grid` takes a **required** `quad_config: MinimaxConfig`.
  Third default-copy gone. (Interacts with the config-seam lens: when
  `PPMSigmaRuntimeOptions` dies, this rides the same signature change — §7.)
- Delete the dead `MinimaxConfig` import at `ppm_sigma.py:68` (until the above makes it
  live again).
- Fields the other side doesn't read are simply unread — same cost as today's two
  classes, minus the drift surface. If the discussion round hates the two Σ-only fields
  in W's instance, the fallback position is keyword-only fields with a comment; a class
  split is the thing that already failed.

### 3.2 One node-preparation accessor — all folds through `to_minimax_nodes`

Extend the existing accessor (no new class) so **every** `MinimaxNodes` in the codebase
is born in one function:

```python
# minimax_screening.py — LaplaceMinimaxQuadrature / CrossingMinimaxQuadrature
def to_minimax_nodes(self, *, time_axis: str,
                     alpha_scale: float | complex = 1.0,   # e.g. -2.0 chi0 prefactor
                     exp_fold: complex = 0.0,              # α ← α·exp(-exp_fold·τ_real)
                     t_scale: float = 1.0,                 # crossing 1/ξ
                     ) -> MinimaxNodes:
```

with the semantics `α_out = alpha_scale · α · exp(−exp_fold · τ)` applied on the **real**
τ before the `time_axis` cast, and `t_out = t_axis_cast(τ) · t_scale`. Then:

- `compute_chi0` (`w_isdf.py:602-605`): `quad.to_minimax_nodes(time_axis='real',
  alpha_scale=-2.0, exp_fold=E_gap)` — the hand-rolled `MinimaxNodes(...)` dies, and the
  `'real'` branch of `_laplace_to_minimax_nodes` finally has its caller.
- `_build_three_sigma_windows` core (`ppm_sigma.py:851-853`):
  `q_cross.to_minimax_nodes(time_axis='crossing_hgl', t_scale=1/ξ, alpha_scale=1/ξ)` —
  the manual rescale dies.
- The per-window ω-independent `E_ref_sum` fold stays in
  `minimax_tau_integrate_sigma:1212` **unchanged** — it multiplies `exp(-i·E_ref·t)` on
  the *complex* t after axis cast, is per-window not per-quad, and moving it would change
  the locked numerical path for zero legibility gain. Instead: one comment block in
  `MinimaxNodes` enumerating the three folds and where each is applied. The point is
  discoverability, not forcing a single fold site where the math doesn't want one.

Everything keeps bit-identical values (same multiplications, same order) — this is a
call-site refactor, not a numerical change; the MoS2 chi0 regression hash
(`w_isdf.py:170-173` comment) must survive untouched.

### 3.3 One two-point pole fit; head = the 1-element case

Extract the shared algebra into the tensor fitter's home and make the scalar head call it:

```python
# minimax_screening.py
def two_point_pole_fit(wc0, wc_probe, z_probe):
    """Ω² = −z²·wc_probe/(wc0 − wc_probe); B = −½·wc0·√Ω²  (elementwise, jnp or float).
    Returns (omega_sq, B_of_positive_branch, good_mask). NO invalid policy here —
    callers apply drop / fallback / continuation explicitly."""
```

- `fit_gn_ppm_from_wc_pair` (:408) = `two_point_pole_fit` + the fallback-Ω policy +
  `unfulfilled_fraction` tally (unchanged public signature).
- `head_correction.fit_head_ppm` (:280) = `two_point_pole_fit` on scalars + its
  **explicit, named** continuation policy for Ω²≤0 (keeping the 2026-07-04 fixed
  `|Ω²|` amplitude), + the `B_h = 2Ω·B` normalization **stated in one comment** next to
  the `R_h = B_h/(2Ω_h)` line. Bug-A-class bugs become structurally impossible: there is
  one place where the sign of B is decided.
- This also surfaces the §2.4 policy skew (body drops, head continues) as two visible
  policy arguments at two call sites — a physics decision the discussion round can then
  actually discuss.

### 3.4 Delete the dead layer; collapse `build_static_minimax_window_pair`

- Delete from `minimax_screening.py`: `MinimaxWindowPair`, `EnergyWindow`,
  `with_imag_freq_modulation`, `build_imag_freq_minimax_window_pair`,
  `extract_gn_ppm_parameters`, `extract_gn_ppm_parameters_from_Wc` (~200 L, all 0-caller
  per §2.2; same class-(b) discipline as SIGMA_PPM_MAP 2A).
- `build_static_minimax_window_pair` becomes `solve_static_screening_interval` — or
  simpler, its body (vmax/cmin/x_min/x_max + one `solve_laplace_minimax_interval` call)
  moves into `w_isdf.build_static_quadrature` (:446), which is its only caller and
  already owns the sibling interval logic for imag/real. Then `minimax_screening.py`
  contains **exactly** the engine: solvers-with-cache + quadrature objects +
  `MinimaxNodes` + the pole fit. Interval derivation (a physics question) lives with the
  physics callers on both sides — symmetric with Σ's `_build_*_sigma_windows`.

### 3.5 Close the shipped-table honor holes

- `build_real_quadrature` (`w_isdf.py:521,532`): pass
  `use_shipped_tables=minimax_config.use_shipped_tables` to both branch solves.
- `solve_laplace_minimax_imag_interval` (:618): add the same keyword; today it resolves
  to "no shipped family exists → always live-solve", with a one-line docstring note. If
  we later ship `noncrossing_imag` tables (cheap: the GN probe uses one ω̂ per run) the
  plumbing is already uniform.

### 3.6 One reference-resolver (vocabulary only)

Move `resolve_minimax_energy_reference` (`w_isdf.py:382`) into the quadrature module and
have `_prepare_sigma_state`'s caller (`ppm_sigma.py:1461-1463` validation +
`use_midgap` flag construction at :1491) use the same midgap/vbm string set through it.
Values stay per-stage (§2.6); only the parse/validate is shared. Low priority; do last
or drop if the sign-bookkeeping lens restructures `_prepare_sigma_state` anyway.

## 4. Migration path + gate strategy

Ordered by (value × safety), each step is a self-contained commit on the feature branch,
gated by `uv run python -m pytest -q` + the MoS2 3×3 1-GPU e2e gates (per the
no-16-GPU-gating rule; the checkpoint skill's golden-gate set). Steps 1-2 are
value-identical by construction; 3-5 must be bit-identical and get an extra numeric gate.

| Step | Change | Gate |
|---|---|---|
| 1 | **Delete-only** (§3.4 dead layer, §2.1 dead import, `ppm_sigma:1430` default fallback → required arg) | pytest + e2e gates compile-and-match; grep-zero on deleted symbols |
| 2 | **Config merge** (§3.1): one class, two `gw_config` properties, one default site | pytest; assert `config.sigma_quadrature_config == old values` in a throwaway check; e2e eqp diff = 0 |
| 3 | **Node accessor** (§3.2): route chi0 + crossing node construction through `to_minimax_nodes` | chi0: MoS2 3×3 locked regression hash (the `w_isdf.py:170` guarantee) must be unchanged; Σ: eqp0/eqp1 bit-diff vs pre-step run |
| 4 | **Pole-fit unification** (§3.3) | new unit test: `fit_head_ppm(w0, wp, z)` vs `fit_gn_ppm_from_wc_pair` on a 1×1×1 tensor for {GN imag probe, HL real probe} × {Ω²>0, Ω²<0, denom≈0} — the full sub-case grid, per the audit-failure-modes rule (no "matches at ULP" hand-waving); plus the existing head-parity gate from the Bug A fix |
| 5 | **Shipped-table holes** (§3.5) + reference resolver (§3.6) | pytest incl. `tests/test_minimax_assets.py` (already monkeypatches `_solve_noncrossing_scaled_cached` at :117 to prove no live solve — extend to real-quad path with `regenerate=true` asserting the solver IS hit) |

Step 4 is the only one that can change physics (it must not — but it touches the Bug-A
region, so it gets the repro fixture from the 2D bug ledger). If the discussion round
wants to also *resolve* the head-vs-body invalid-policy skew (§2.4), that is a **separate
physics decision with a BGW parity check**, not part of this refactor.

## 5. Where W and Σ genuinely NEED different treatment — do not force-merge

1. **τ-loop mechanics**: chi0's `lax.scan`-inside-jit vs Σ's Python loop. The divergence
   is load-bearing (NCCL in the Σ body; the scan experiment regressed MoS2 3×3 ~80%,
   `ppm_sigma.py:466-470`). The shared surface is `MinimaxNodes` — that's enough. Any
   "one `minimax_tau_integrate` to rule them both" is a trap.
2. **Interval derivation**: W's is a band-energy difference range (`E_c−E_v`, one
   interval per run); Σ's is a per-branch Minkowski-ish sum range
   (`min/max(E_A) + min/max(Ω_B)`, ω-extended for `kernel_sign=−1`, split at
   `T = ω_max + z_edge` for the 3-window family — `ppm_sigma.py:759-766, 809-860`).
   Different physics, different masks, different cardinality. Both sides should *call*
   the same solver front-ends (they do) and *construct* nodes the same way (§3.2), but
   the interval math stays with its physics.
3. **Node kinds**: the crossing quadrature is intrinsically Σ-only (real-ω pole-crossing
   regularization, `project="imag"` HGL windows); the imag-Laplace is intrinsically
   W-only (chi0 on the imaginary axis). The kind table in §1 is the natural contract —
   don't invent a fourth abstraction to hide a 3-row table.
4. **Accuracy knobs**: `screening.minimax_*` vs `ppm.sigma_*` stay independent
   user-facing knobs (a user converging Σ windows at 1e-7 shouldn't silently re-tighten
   chi0). One *class*, two *instances*, two knob families.
5. **Energy references**: three stages, three physically distinct references (§2.6);
   share the resolver, never the value.

## 6. Risks

- **R1 — locked chi0 hash** (step 3): the `alpha_chi` fold currently multiplies
  `float64·float64` then casts complex; routing through the accessor must preserve the
  exact op order (the `w_isdf.py:184-187` comment documents why complex·complex with
  Im=0 is bit-safe — keep that reasoning attached to the accessor). Mitigation: hash gate
  before/after, and the accessor applies folds on real τ pre-cast, exactly as today.
- **R2 — AOT precompile signature drift**: `precompile_sigma` (`ppm_sigma.py:623`)
  matches the runtime `(shape, dtype, sharding, committedness)` tuple; the quadrature
  refactor never touches kernel signatures, but step 2's required-arg change alters
  `compute_sigma_c_ppm_omega_grid`'s call contract — coordinate with the config-seam
  lens so we don't change that signature twice in two PRs.
- **R3 — disk-cache invalidation**: none of §3 changes cache payload keys
  (`minimax_screening.py:192-198`); step 5's new kwarg on the imag solver must not enter
  the payload dict (it's a lookup-path choice, not a solve parameter). Check the payload
  dicts explicitly in review.
- **R4 — head normalization trap** (step 4): `B_h = 2Ω·B_tensor` — the unit test grid in
  §4 exists precisely to pin this; do not let the unifier "simplify" the head to the
  tensor normalization without also changing `compute_ppm_head_sigma_kij`'s consumption
  of `R_h`.
- **R5 — scope creep**: §3 deliberately does NOT touch `_SigmaWindow`, mask_B modes, the
  4-branch decomposition, or accumulators. If reviewers pull those in, the numeric gates
  stop isolating causes.

## 7. Interaction with the other 3 lenses

- **Module-split lens (ppm_sigma.py = ~5 concerns)**: natural seam agreement — after
  §3.4, `minimax_screening.py` is the pure engine, and ppm_sigma's window builders
  (`_build_single_sigma_window:743`, `_build_three_sigma_windows:789`,
  `_build_windows_for_branch:1089`) + `minimax_tau_integrate_sigma:1169` form a coherent
  "Σ quadrature/windows" unit that a file split can lift out wholesale (they depend only
  on the engine + numpy stats, not on kernels/accumulators). **Request to the splitter**:
  keep window-build and τ-integrate in the same new module — they share the
  `MinimaxNodes` conventions — and let the kernel/accumulator/driver concerns split
  elsewhere. Conflict risk: none structural, but merge-order matters (my step 1 deletes
  lines the split will move — do delete-pass first, split second).
- **Config-seam lens (`PPMSigmaRuntimeOptions` collapse)**: overlapping edit at the
  `compute_sigma_c_ppm_omega_grid` signature. Today the function takes BOTH
  `ppm_options` (getattr grab-bag, `ppm_sigma.py:1435-1441`) AND
  `sigma_window_quad` (`ppm_pipeline.py:355`) — two config objects through one door.
  Proposed contract for the merged seam: `(config.ppm-or-equivalent, quad_config,
  omega_grid)` where `quad_config` is the §3.1 instance built **once** in
  `ppm_pipeline.compute_ppm_sigma_pipeline` (not re-derived per call via the gw_config
  property, which allocates a new dataclass every access). One PR should own that
  signature (R2). Note the `use_shipped_minimax_tables` thread
  (`ppm_sigma.py:1429→1587→1103`) collapses into the quad_config field — one less loose
  kwarg for the seam lens to carry.
- **Sign/mask-bookkeeping lens (`_SigmaBranch` kernel_sign/scale/project, mask_B)**: two
  touchpoints. (a) The `time_axis` conventions and the E_ref/ξ folds (§3.2) are the
  quadrature half of the sign story — if that lens writes the "one page of sign
  conventions" doc, the `MinimaxNodes` docstring should be its anchor and my §3.2 comment
  block should merge into it rather than duplicate it. (b) `project="imag"` /
  `project_code` is *decided at window build* (`ppm_sigma.py:854,868`) and *consumed at
  projection* (`_combine_coeff_with_sigma_tau:361`) — a crossing-node-kind property, not
  a free string. If that lens re-types `_SigmaWindow`, `project` should derive from the
  node kind (`CrossingMinimaxQuadrature` → imag) instead of being a parallel field that
  can disagree. (c) The §2.4 head-vs-body invalid-policy skew lands in their court as a
  physics question once §3.3 makes it visible.
- **(If a head lens exists)**: §3.3 is the shared edit; the Bug-B streamed-head-drop fix
  (`ppm_pipeline.py:126-127`) is orthogonal to quadrature and unblocked by nothing here.

---

*Agent 1 / quadrature-minimax lens. All line numbers verified against
`sources/lorrax_D` @ `3cad3dd` (branch `agent/memplanner-cleanup` checkout state),
2026-07-06.*
