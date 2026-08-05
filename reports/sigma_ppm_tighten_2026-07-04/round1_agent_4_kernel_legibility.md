# Round 1 — Agent 4: Make the tau-kernel + accumulators legible and bug-resistant

**Lens**: kernel legibility + correctness. The τ-kernel sign/mask bookkeeping
(`_SigmaBranch.kernel_sign/scale`, the 4-branch ±ω decomposition, `mask_B` modes,
`E_ref_A/B`, project codes 0/1), the two accumulators, and where Bug B lives.

**Ground truth**: `sources/lorrax_D/src/gw/ppm_sigma.py` @ 1632 L (post-2A deletions —
the map's 1702 is stale; `mask_B_mode="explicit"` and the dead debug knobs are already
gone, `ppm_invalid_mode` is already wired as `keep_invalid` into `_prepare_sigma_state`
at ppm_sigma.py:324-348 with the mode validation at :1465-1483). All line numbers below
are against this working tree.

**Confidence markers**: ✅ verified in code this session · 🟡 believed, wants a trace/run
check · ⚠️ open risk.

---

## 1. Verdict: which complexity is essential, which is incidental

I agree with the lead's "good bones, rotted implementation" — with one sharpening: the
τ-kernel's *structure* is not the problem; its **sign bookkeeping is scattered across
seven locations with no single derivation**, and the two accumulators are ~70% the same
machine written twice (including a duplicated projector that must "match exactly" by
comment discipline alone, ppm_sigma.py:928-933).

**Essential complexity (keep, document better):**

| What | Why it's essential |
|---|---|
| 4-branch (±ω × cond/val) decomposition, `_iter_branches` :197-234 | Physics: the Laplace representation only converges for a definite-sign denominator; the ω-grid must be split at E_F and each half split by pole species. Not removable. |
| 3-window (core/a_stripe/b_slab) decomposition :789-886 | The denominator ω̃ − (E_A+Ω) can vanish on exactly one branch per ω-half; the crossing (HGL) quadrature exists only for that region. Not removable. |
| σ^τ carried as (re, im) pair :1217-1219 | Crossing window needs only Im[coeff·σ]; complex σ^τ would double HBM + NCCL volume. |
| Python τ-loop (not lax.scan) :1180-1183 | Deliberate — monolithic scan regressed MoS2 3×3 ~80%. **Non-negotiable constraint on everything below.** |
| E_ref_A/E_ref_B gauge shifts | Laplace-argument conditioning: minimax tables want x ∈ [x_min, x_max] with x_min > 0. Essential, but currently *illegible* (§2, item 6). |
| Async D2H lag deque in `_HostOmegaAccumulator` :993-1015 | Overlaps GPU τ_{k+lag} with numpy accumulate of τ_k. Keep semantics bit-exact. |

**Incidental complexity (the rot — all fixable without touching the math):**

1. `kernel_sign` is one int overloaded with **three meanings** (§2).
2. Signs live in 7 places with no ledger; the ±ω derivation comment is *actively
   misleading* (§2, the "Σc(−ω) = −Σc(ω)\*" prose).
3. Two accumulators + two mirrored projectors (jax :361-424, numpy :923-945) where one
   accumulator + one projector suffices (§4).
4. A **third** accumulation layer in the driver itself: `per_half` dict keyed by
   `tuple(br.omega_idx.tolist())` — a tuple of potentially hundreds of ints as a dict
   key (:1610-1611) — plus the `sigma_kij_host` scatter (:1531-1534, :1613-1620).
5. The getattr grab-bag config read :1435-1441 — a third silent default layer
   (e.g. `sigma_regularization_ry` default `0.018374661087827496` hardcoded at :1435,
   parallel to the config default). Same defect class as the invalid_mode "3 modes
   fighting" bug.
6. Bug B (streamed head drop) is a **structural consequence** of the two-accumulator
   split: everything downstream of `compute_sigma_c_ppm_omega_grid` branches on
   `sigma_c_omega is None`, and three of those branches forget the streamed case
   (§5).

---

## 2. The sign ledger — where every sign lives today

There is no place in the codebase where a reader can see all the signs at once. Today
they are:

| # | Sign | Where | What it encodes |
|---|---|---|---|
| 1 | `inv_sqrt_nk = -1.0/√N_k` | ppm_sigma.py:540 | The **global Σc minus sign is hidden inside an FFT normalization constant**. A reader auditing branch signs will never look here. |
| 2 | `kernel_sign` ±1 per branch | :218-232 (table), consumed as `omega_sign` at :780, as the window-family selector at :1124, and as an x_max extension at :763-766 | Whether ω enters the Laplace exponent with + or − (i.e. denominator ω̃ − S vs ω̃ + S). |
| 3 | `scale` ±1 per branch | :220-232, folded into `pref` at :990/:1059 | The overall − from re-parameterizing the −ω half at \|ω\|. |
| 4 | window `prefactor` | +1 Laplace :773/:869, **−1** for kernel_sign=−1 single :773, **−1** for crossing :856 | Per-window sign from the Laplace vs HGL quadrature identity. |
| 5 | W-side phase `e^{−i(Ω−E_ref_B)τ}` | `_build_W_t_q` :600 | Pole phase, gauge-shifted. |
| 6 | `α_eff = α·e^{−i·E_ref_sum·t}` | :1212 | The compensating gauge factor for #5 + the G-side `e_ref`. |
| 7 | crossing keeps only `Im[coeff·σ]` | `_combine_coeff_with_sigma_tau` :390-391, numpy mirror :941-942 | HGL quadrature projects the imaginary part. |

Two concrete legibility defects:

**(a) The ±ω derivation prose is wrong-as-stated.** ✅ Both the module docstring
(:29-30) and `_iter_branches` (:211-213) say the −ω half "exploits
Σ_c(−ω) = −[Σ_c(ω)]\*". Read literally, that global identity would make the two −ω
branches **redundant** (just conjugate the +ω result) — but the code runs them, because
the identity is *per pole term*, not global: for a single term
T(ω) = B/(ω − E_s ∓ Ω ± iη), evaluating at ω = −\|ω\| gives
T(−\|ω\|) = −[T′(\|ω\|)]\* where T′ is the *same-form* integral with the cond/val
kernel-sign swapped (the conjugation is the iη-orientation flip, which is why the
crossing window — the only place Im enters — moves from the cond branch on the +ω half
to the **val** branch on the −ω half). The code is right; the comment describes a
different, false theorem. Any future auditor "verifying" the branch table against that
comment will either waste a day or, worse, "fix" correct code.

**(b) `kernel_sign` overloading.** ✅ Three consumers, three meanings:
- ω-kernel sign: `omega_sign=int(kernel_sign)` at :780 (single window only; three-window
  hardcodes `omega_sign=+1` at :878).
- Window-family dispatch: `if kernel_sign == +1 and omega_max > 1e-14:` at :1124 —
  i.e. "+1 means the denominator can vanish, build crossing windows".
- Laplace domain extension: `x_max = S_max + omega_max` only when `kernel_sign == -1`
  at :763-766.

### Proposal K0 — the canonical branch table + sign ledger (docs + rename, zero math change)

Replace the `_SigmaBranch.kernel_sign: int` with an explicit denominator kind, and make
the 4 branches a module-level constant the arrays get bound onto:

```python
class _DenomKind(enum.Enum):
    OMEGA_MINUS_S = "ω̃ − (E_A+Ω)"   # can vanish → 3-window (crossing); ω-kernel e^{+ω̃τ}
    OMEGA_PLUS_S  = "ω̃ + (E_A+Ω)"   # positive-definite → single Laplace; ω-kernel e^{−ω̃τ}

# The four branches of Σc(ω).  DERIVATION: docs/SIGMA_SIGN_LEDGER (below).
#   half   space  E_A        denom            scale   crossing?
_BRANCH_TABLE = (
    ("pos", "cond", "E_c-E_F", _DenomKind.OMEGA_MINUS_S, +1.0),   # ω≥E_F cond
    ("pos", "val",  "E_F-E_v", _DenomKind.OMEGA_PLUS_S,  +1.0),   # ω≥E_F val
    ("neg", "cond", "E_c-E_F", _DenomKind.OMEGA_PLUS_S,  -1.0),   # ω<E_F cond
    ("neg", "val",  "E_F-E_v", _DenomKind.OMEGA_MINUS_S, -1.0),   # ω<E_F val
)
```

- `_iter_branches` (:197-234) shrinks to a loop that binds `(E_cond|H_val,
  cond_mask|val_mask, omega_pos|omega_neg_abs)` onto `_BRANCH_TABLE` rows. The
  crossing-vs-single dispatch at :1124 becomes `denom is _DenomKind.OMEGA_MINUS_S`
  (self-documenting), the ω-kernel sign at :780 becomes an explicit
  `omega_sign = +1 if denom is OMEGA_MINUS_S else -1`, and the x_max extension at
  :763-766 gets its own comment ("ω shifts the positive-definite denominator's upper
  edge").
- Add a **module-docstring sign ledger**: the 7-row table above, verbatim, with the
  corrected per-term ±ω derivation replacing :29-30 and :211-213. One paragraph deriving
  each branch's (denom, scale, prefactor) from the pole form. This is ~40 lines of
  comment and is the highest-value/lowest-risk item in this proposal.
- Rename the constant at :540 from `inv_sqrt_nk` to something that carries the sign,
  e.g. `neg_inv_sqrt_nk`, with a one-line pointer to the ledger row.
- Make project codes an IntEnum (`_Project.FULL = 0`, `_Project.IMAG = 1`) replacing
  the magic ints threaded through `_SigmaWindow.project_code` (:170-177) and the numpy
  projector (:938-944). Grep-ability only.
- Add the **E_ref factorization identity** as a comment at :1212 where α_eff is built:
  `e^{−(E_A+Ω)τ} = e^{−E_ref_sum·τ} · e^{−(E_A−E_ref_A)τ} · e^{−(Ω−E_ref_B)τ}` — the
  three factors live in α_eff (:1212), `build_G_tau(e_ref=...)` (:564-567), and
  `_build_W_t_q` (:600) respectively. Today a reader must reconstruct this identity from
  three files.

⚠️ Constraint: none of this may change the `_tau_kernel` argument structure —
`precompile_sigma` (:648-672) AOT-compiles against the exact
(shape, dtype, sharding, committedness) signature and a silent retrace costs the whole
compile again. K0 is renames + comments only; the kernel signature is untouched.

---

## 3. The window builders: one data table instead of parallel if/elif

`_build_three_sigma_windows` (:789-886) encodes the (window ↔ A-condition ↔ B-mask ↔
quadrature ↔ project ↔ prefactor) coupling as a `for name in (...)` loop with two
string-matched if/elif ladders (:815-833 and :841-869). The coupling is the physics;
the ladders are incidental. Flatten to a spec table:

```python
#   name       A-condition      B-side   quadrature      project  prefactor
#   core       E_A ≤ T          Ω ≤ T    HGL crossing    IMAG     −1
#   a_stripe   E_A > T          Ω ≤ T    Laplace         FULL     +1
#   b_slab     (all)            Ω > T    Laplace         FULL     +1
```

with one row-builder function. `T = ω_max + edge·ξ` gets defined once with the comment
"poles/transitions below T can make the denominator vanish somewhere on the ω grid;
above T they never can — that is the entire meaning of mask_B le_t/gt_t". The `mask_B`
machinery (`_materialize_window_mask_B` :265-280) is fine post-2A (three modes, all
live) — it only lacks that one sentence connecting `T` to "crossing possible".

This is a pure re-layout: same windows, same nodes, same order. Gate = bit-identity
(§6).

---

## 4. Proposal K2 — one accumulator, two sinks; delete the duplicated projector

### The duplication, precisely

✅ `_HostOmegaAccumulator` (:948-1026) and `_StreamedH5Accumulator` (:1029-1079) both:
cache `(omega_sign, pref = prefactor·scale, project_code)` at `begin_window`
(:986-991 vs :1057-1060), then per τ apply the *same* projection formula
`pref · α_eff · e^{i·sign·ω·t} · P(σ_re, σ_im)` — via **two separate implementations**:
`_project_tau_onto_omega_np` (:923-945) and the jitted
`_project_tau_onto_omega`/`_combine_coeff_with_sigma_tau` pair (:361-424). The numpy
one's docstring says "Matches the jax version exactly" — enforced by nothing. That is
the canonical parallel-path drift hazard (and the exact smell the repo's
no-redundancy rule bans).

### The perf asymmetry nobody wrote down

✅ The streamed accumulator is strictly *worse* on data movement, not just on h5 RMW:

- Host path: D2H per τ = one σ tile, `nk·(nb/p_x)·(nb/p_y)·2·16 B` — **ω-independent**
  (:997-1001). Projection outer-products on host.
- Stream path: it projects **on device** into `(n_ω_batch, nk, nb, nb)` blocks, then
  `jax.device_get`s **every batch** (:1568) — D2H per τ = `n_ω · nk · nb² · 16 B`,
  i.e. **n_ω× the volume**, plus `n_ω/batch` h5 read-modify-write round-trips per τ
  (:1563-1569). For the large-ω-grid runs streaming exists for, this is maximally
  backwards.

### Target design

One accumulator class owning the async-D2H deque + the single numpy projector; the
"what happens to a finished window" decision becomes a sink:

```python
class _TauAccumulator:                      # replaces both classes
    # __init__(local_shape, omega_vec_np, sink, lag=2)
    # begin_window(window, *, scale)  — zero win_acc, cache sign/pref/code
    # add_tau(σ_re, σ_im, t_c, α_eff_c) — copy_to_host_async + deque (verbatim :993-1015)
    # end_window() — drain; sink.consume_window(win_acc)
    # finalize()   — sink.result()

class _MemoryTileSink:   # ≡ today's _HostOmegaAccumulator tail
    # consume_window: total += win_acc
    # result: make_array_from_process_local_data(...)      (verbatim :1024-1026)

class _H5Sink:           # replaces _StreamedH5Accumulator
    # consume_window: for each ω-batch: dset[idx] += win_acc[batch]   (rank-0)
    # result: None
```

Consequences:

- **Delete** `_project_tau_onto_omega` + `_combine_coeff_with_sigma_tau` (:361-424,
  ~64 L) — their only caller is `_StreamedH5Accumulator.add_tau` (:1068). The numpy
  projector becomes the single source of truth. The lax.switch-HLO-stability concern
  documented at :380-383 dies with the function.
- h5 RMW count drops from `n_τ · n_ω/batch` to `n_windows · n_ω/batch` (~n_τ ≈ 15-40×
  fewer); D2H volume drops n_ω-fold to match the host path.
- The τ-loop (`minimax_tau_integrate_sigma` :1169-1224) and
  `_integrate_tau_windows_for_branch` (:1227-1289) are **untouched** — they already
  program against the `_SigmaAccumulator` protocol (:897-920). This refactor is
  entirely behind that seam.

⚠️ **Host-RAM tradeoff**: `_H5Sink` holds one full-window `(n_ω_branch, nk, nb_proj²)`
c128 buffer on host, which the old per-τ streaming avoided. On Perlmutter (256-512 GB
host) this is a non-issue for any grid we run; the `_select_accum_mode` "0.5 GiB"
auto-threshold (:117) is already an arbitrary host-side number and should be re-stated
as "host window buffer budget". If a grid ever appears where even one window's buffer
doesn't fit, fall back to per-ω-batch drain inside `consume_window` — documented
follow-up, not built now (YAGNI).

### A latent third bug found while auditing (Bug C, ⚠️ PLAUSIBLE — needs a check)

`_HostOmegaAccumulator.add_tau` takes **only shard 0** (`addressable_data(0)`,
:997-998) and sizes its tile as one *device* shard (`shard_shape`, :973), but
`finalize` hands that tile to `make_array_from_process_local_data` (:1024-1026), which
expects **process**-local data. These coincide only when each process owns exactly one
GPU. Single-process multi-GPU (a plain `lxalloc` 4-GPU interactive session, or any
end-user desktop with 2 GPUs — see the no-16-GPU-gating rule: this code ships to
arbitrary device counts) hands a 1/P-sized tile where a full-size one is expected. Best
case it raises; worst case silent wrong Σc. Fix inside K2: iterate
`addressable_shards` and accumulate into per-shard tiles (the projector already works
per-tile), or `assert jax.local_device_count() == 1` with a loud error until then.
Verify with a 5-line repro on a 2-GPU single-process mesh before assuming which.

---

## 5. Bug B — the streamed head drop, and its two siblings

The map places Bug B at ppm_pipeline.py:126-127 — correct, but it understates the blast
radius. Everything downstream branches on `sigma_c_omega is None`, and **three**
consumers mishandle the streamed case:

1. ✅ **Head never enters the h5** — `_inject_analytic_head` early-returns
   `(None, None)` (ppm_pipeline.py:126-127). The stream file keeps head-less Σc, and
   `head_sigma_diag_w_kn_ry` is None so even the diagnostic print (:154-157) vanishes.
2. ✅ **At-DFT eval reads the head-less h5** — `_eval_sigma_c_at_dft_energies` streamed
   fallback reads `sigma_c_kij_ry` directly (ppm_pipeline.py:209-220), so
   `sigma_c_at_dft_ev` is silently missing hundreds of meV.
3. ✅ **The QP fixed-point solve is skipped entirely** — gw_jax.py:628
   (`elif mode.is_dynamic and sigma_c_omega is not None:`): streamed runs produce no
   on-shell QP energies at all, only the at-DFT diagnostic. Arguably intentional
   (the full ω-tensor isn't on device), but today it's *undocumented* fallthrough.

**Fix (concrete):** inject the head into the stream file, *before* step 4 reads it.
`compute_ppm_head_sigma_kij` already returns host numpy (head is a scalar-pole formula,
no device work). In `_inject_analytic_head`, replace the early-return with:

```python
if sigma_c_omega is None:
    if sigma_kij_h5_path is None:            # no output at all → hard error, not silence
        raise ValueError("streamed Σc has no h5 path; head cannot be injected")
    if meta.rank == 0:
        with h5py.File(sigma_kij_h5_path, "r+") as h5:
            d = h5["sigma_c_kij_ry"]
            for ibeg in range(0, n_omega, batch):      # ω-batched RMW add
                d[ibeg:iend] += head_sigma_kij_ry[ibeg:iend]
            h5.attrs["head_injected"] = True           # idempotence guard for restarts
    return None, head_diag_w_kn_ry              # diag returned → print + PPMOutputs live
```

(signature change: `_inject_analytic_head` gains `sigma_kij_h5_path=sigma_omega.sigma_kij_h5_path`
and `meta`; call site ppm_pipeline.py:365-369.) The `head_injected` attr makes the
operation idempotent — without it, a restarted pipeline double-adds the head into the
RMW file. Consumer 2 then reads a correct file with no change; consumer 3 gets one
sentence of documentation ("streamed mode: at-DFT eval only, no on-shell solve") or a
loud `print_fn` warning at the gw_jax.py:628 seam.

**Bug A status:** ✅ already fixed in this tree — head_correction.py:320-338, comment
dated 2026-07-04 (`B_h = -w1 * abs(omega_h_sq)` in the negative-Ω² branch). But
`tests/test_head_correction.py` has **zero coverage of `fit_head_ppm`'s negative
branch** (only 3 tests: static terms, kij broadcast, override resolution). Add the
regression test now: fit with samples engineered so `omega_h_sq < 0`, assert
`sign(R_h)` matches the `omega_h_sq → 0⁺` limit (continuity across the branch), and
assert `R_h/omega_h` continuity explicitly. Cheap, host-only, closes the door on the
sign flip returning.

---

## 6. Migration path + gate strategy

Everything runs on the MoS2 1-GPU fixture class (per the no-16-GPU-gating rule);
red-green ordering is deliberate — the parity gate must FAIL on Bug B before the fix
lands.

**Stage 0 — gates first (no source change):**
- **G1 accumulator-parity gate**: one small MoS2 run twice — `omega_accumulation=kij`
  vs `kij_stream` (+ a `sigma_kij_h5_file`) — assert `sigma_c_kij_ry` in the h5s equal
  to ~1e-12 *and* `eqp` outputs consistent. This gate does not exist today, which is
  **why Bug B survived**: nothing ever compared the two accumulator paths.
  ⚠️ Expected RED at head-injection until Stage 1.
- **G2 branch/window reference gate**: symmetric ω grid with ω_max large enough that
  all 3 windows and all 4 branches are non-empty (assert via window count in the log);
  store per-branch Σc tiles as a checked-in `.npz`; assert match. This is the
  bit-identity anchor for every rename in K0/§3.
- **G3 Bug-A regression test** in `tests/test_head_correction.py` (§5). GREEN already
  (fix landed) — pins it.

**Stage 1 — close Bug B** (ppm_pipeline only, §5 patch). G1 goes GREEN. ~30 L.

**Stage 2 — K2 accumulator unification** (delete jax projector pair, `_TauAccumulator`
+ 2 sinks, Bug-C shard handling or loud assert). Gates: G1, G2, plus the 3 golden e2e
gates. Net ≈ −80 L. Also fold the driver's third layer here: build one accumulator
**per ω-half** at the driver and pass it through `_run_sigma_branch` (dependency
injection replacing construction at :1359-1375), killing the `per_half`
tuple-of-ints dict (:1599-1611). 🟡 cond+val summation moves from one device add to
numpy tile adds — same values, same order, elementwise → bit-identical, but G2 verifies.

**Stage 3 — K0 sign ledger + renames + §3 window table.** Zero math change; G2 must be
bit-identical, precompile must not retrace (check the `sigma.compile` timing section
doesn't reappear at exec time).

**Stage 4 — hand off the config seam** (getattr block :1435-1441) to the
`PPMSigmaRuntimeOptions` collapse owner (§7) — my only requirement is the narrowed
signature: `compute_sigma_c_ppm_omega_grid(wfns, ppm, meta, mesh_xy, *, omega_grid_ry,
ppm_cfg, debug_cfg, sigma_window_quad, print_fn)` with **required** fields (no getattr
defaults — every getattr default here is a third silent default layer).

---

## 7. Risks

1. **Perf: the Python τ-loop is load-bearing** (:1180-1183). K2 does not touch the loop
   or the per-τ dispatch; the accumulator's `add_tau` remains non-blocking
   (async D2H + deque, drain threshold `> lag` verbatim). Any change that makes
   `add_tau` synchronous re-serializes GPU/host overlap → watch the `sigma.exec`
   timing section on the e2e gates (budget: ±3%).
2. **AOT retrace** — `precompile_sigma`'s signature-matching comments (:650-657) are
   the contract; K0/K2 must not alter kernel arg commitment. Detection: compile time
   appearing inside `sigma.exec`.
3. **Reduction-order bit-identity** — the per-half sum comment (:1599-1601) promises
   bit-identical traversal; Stage 2's per-half accumulator preserves traversal order
   (cond before val, windows in order, τ in order) by construction, G2 enforces.
4. **h5 RMW `+=` on a chunked dataset** in the Bug-B fix and `_H5Sink`: fancy-index
   RMW on h5py is the slow path; batch over contiguous ω slices (the dataset is
   chunked `(o_chunks, k_chunks, nb, nb)` :1548-1558, so ω-contiguous slabs are
   aligned).
5. **Bug C uncertainty** ⚠️: I have not run the single-process multi-GPU repro; if
   `make_array_from_process_local_data` raises on the short tile, today's severity is
   "crash", not "corruption" — the fix is the same either way, but the report language
   should match. 20-minute check on a 2-GPU alloc.
6. **Streamed-mode semantics widening**: after Bug B's fix, streamed runs produce
   different (better) numbers than before. Any downstream consumer that compensated
   for the missing head (none found by grep, 🟡) would double-count.

---

## 8. Interaction with the other 3 lenses

- **Module-split lens** (ppm_sigma.py = ~5 concerns in 1 file): sequencing conflict —
  if the file split lands first, K2 moves code twice. **Proposed order: K2 (unify)
  before any split**, then the accumulator+sink trio is a clean standalone module
  (~150 L) and the window builders + branch table another. My K0 ledger wants to live
  in the module docstring of whichever file ends up owning `_BRANCH_TABLE` — one home,
  not two.
- **Config-seam lens** (`PPMSigmaRuntimeOptions` collapse, map §2B): shared ownership
  of the getattr block :1435-1441. I define the narrowed kwargs (§6 Stage 4); they own
  deleting the mirror in gw_driver_helpers.py:16-34 and build_ppm_sigma_runtime_options
  :230-269. One negotiation point: whether the ω-grid derivation
  (gw_driver_helpers.py:244-250) moves into the driver or stays at the seam — I have no
  stake, but the `sigma_kij_h5_path` empty-string-vs-None convention (:267 produces
  `""`, `_select_accum_mode` :123 tests falsy — works by accident) should become
  explicit `None` in the same pass.
- **Physics/BGW-parity lens** (invalid modes, head parity): `ppm_invalid_mode` is
  already wired for zero/2ry (:324-348, :1465-1483); the `static_limit` (BGW mode 3
  = BGW default) NotImplementedError at :1471-1476 needs `Wc0` retention and an
  analytic −½Wc0 term — that lands in `_prepare_sigma_state` and `fit_ppm`, which K0
  touches only cosmetically → no conflict, but coordinate the `_prepare_sigma_state`
  docstring rewrite. **Potential conflict on Bug B**: if that lens prefers "forbid
  streaming without a head path" over injection, I push back — large-ω-grid streamed
  runs are precisely where the ω-dependent head matters; forbidding removes the
  feature instead of fixing it. My h5-injection fix (§5) keeps the head's BGW-parity
  check meaningful in both accumulation modes, and G1 is the shared gate both lenses
  need anyway.
- **All lenses**: G1/G2 (Stage 0) are prerequisites for *everyone's* refactors, not
  just mine — propose the discussion round adopts "no Σ_PPM source change lands before
  G1+G2 exist" as a consensus rule.
