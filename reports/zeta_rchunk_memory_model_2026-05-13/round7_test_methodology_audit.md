# Round 7 — Test methodology audit (Agent 4 WS-B)

**Triggering event**: Round 6 G1 surfaced two design bugs in
`z_q_from_psi_sm` (commit `f567aa0`) that Agent 2's own G1 — a
full MoS2 3×3 charge-channel ζ-fit against the lorrax_A `ff5873c`
baseline — completely missed.  Both bugs fired in adversarial
sub-gates Agent 4 added to the synth-WFN bit-identity test:

- **G1.1b — short final bc** (`nb_total = bc·k + r`, `r < bpd_max`):
  max rel = 11.5.
- **G1.1c — asymmetric L/R windows** (`band_range_left = (0,5)`,
  `band_range_right = (3,8)`): max rel = 5.04.

Both have the same root cause: XLA's
`lax.dynamic_slice_in_dim(arr, start, size, axis=...)` silently
clamps the start to `max(0, axis_size - size)` when the requested
slice would extend past the array end.  The L/R mask is built from
*logical* band indices, but the slice returns *physically different*
bands when clamping fires.  The new design's front-pad fixed the
negative-offset case (bc.lo < L_lo_g) but not the symmetric
end-overflow case (bc.lo + bpd_max_global > psi_l_X.shape[2]).

This audit answers: **what test fixture grid would have caught this
up-front?**  And: **what canonical fixtures must any future
pair-pipeline-like bit-identity test include?**

## 1.  Why Agent 2's own G1 missed it

Agent 2's G1 was an **end-to-end MoS2 3×3 charge-channel run**
compared against `lorrax_A ff5873c` (max rel = 1.02e-10, broadly
distributed → "ULP-class drift" reported).  This is a perfectly
reasonable production-validation test.  But it has three structural
properties that *hide* the back-pad bug:

| Property | MoS2 3×3 charge | Bug trigger |
|---|---|---|
| L window | `(0, nb_total)` — full coverage | bug fires when `bc.lo + bpd_max_global > L_hi` |
| R window | `(0, nb_total)` — full coverage | bug fires when `bc.lo + bpd_max_global > R_hi` |
| `nb_total % bc_size == 0` | True (e.g. 80 = 5·16) | bug fires when last bc is short (pad-bands) |

**All three conditions must hold for the bug to fire.**  Charge
channels with band-divisible bcs satisfy zero of them; bispinor
transverse channels (L = val window, R = cond window) satisfy two
of them by design; "anything with a trailing short bc" satisfies
the third.

Production-shape end-to-end tests are valuable for catching
*regressions vs known-good output*, but they ONLY exercise the
production-shape design assumptions.  Bugs that hide in those
assumptions are invisible.

## 2.  The two canonical fixtures every pair-pipeline test must include

### Fixture A — `short_final_bc`

```python
band_chunk_ranges = ((0, 4), (4, 8), (8, 9))    # 3 bcs, sizes 4, 4, 1
band_range_left   = (0, 9)                       # full
band_range_right  = (0, 9)
```

What it probes:
- `_bpd_max = max(bc_size) = 4` ≠ last bc's actual size (1).
- io_callback's pad-rows-are-zero contract (the 3 pad rows in bc 2
  must contribute exactly nothing).
- L/R mask's `bc_valid = g_axis < b_hi_global[bc_idx]` cutoff on
  the trailing bc.
- *And the back-pad bug*: `bc 2.lo + bpd_max_global = 8 + 4 = 12 >
  nb_l = 9` → `dynamic_slice_in_dim(psi_l_X_padded, 8, 4, axis=2)`
  on a length-9 array clamps offset 8→5, returning bands [5,6,7,8]
  instead of [8,?,?,?].

This is the simplest fixture that triggers the back-pad bug.
**It should be in every test of a kernel that band-chunks against
a fixed-size psi_l_X / psi_r_X.**

### Fixture B — `asymmetric_L_R`

```python
band_chunk_ranges = ((0, 4), (4, 8))             # 2 bcs of 4
band_range_left   = (0, 5)                       # L != R
band_range_right  = (3, 8)
```

What it probes:
- The per-bc L offset and per-bc R offset have DIFFERENT signs
  (front-pad on R side, back-pad on L side for some bcs).
- The "bc 1 = (4,8)" case hits L window only at band 4 (out of 5
  total in L window): `bc.lo + bpd_max_global = 4 + 4 = 8 > nb_l =
  5` → clamp 4→1, slice [1..4] instead of [4..8].

This is the simplest fixture that triggers the bug in the way
**bispinor production code will hit it**.  bispinor transverse runs
with `L = val_window`, `R = cond_window`; any bc straddling the
val/cond boundary triggers both sides' offset math at once.

### Why both, not just one

`short_final_bc` exercises the back-pad bug on the *end of the
band axis* (offset + size overruns the trailing edge of psi_l_X).
`asymmetric_L_R` exercises it on the *interior* of bcs (window
edge cuts through a bc, not just the trailing bc).  A buggy fix
that only back-pads to handle the trailing case (e.g. "pad psi_l_X
to a multiple of bc_size") would pass Fixture A but still fail
Fixture B.

## 3.  The XLA `dynamic_slice_in_dim` clamp footgun

Independent of this specific kernel: **`lax.dynamic_slice_in_dim`
silently clamps out-of-bounds starts.**

```python
arr = jnp.arange(5, dtype=jnp.float64)
out = lax.dynamic_slice_in_dim(arr, jnp.int32(4), 4, axis=0)
#                                 ^^^ start  ^ length on len-5 array
# Returns [1., 2., 3., 4.] — NOT [4., ?, ?, ?]
# Start was clamped from 4 → max(0, 5-4) = 1.
```

The clamp behavior is documented (it's needed for static-shape
codegen), but it's *exceptionally easy to miss* when:
- The caller builds a mask in parallel from logical/global indices,
  assuming the slice returns those positions.
- Static analysis can't catch the mismatch because both the slice
  size and the array size are static — only the **traced offset**
  varies, and only at runtime do we discover the offset times its
  intended bands extend past the array.

**Recommended general-purpose mitigations** (any time a traced
`dynamic_slice_in_dim` offset can take values where `offset +
slice_size > axis_size`):

- **Pre-pad the array** so the slice never overflows.  Pad value:
  for `where`-masked consumers, zero (math-neutral).  For
  arithmetic consumers, NaN (forces a debug crash if mask is
  buggy).
- **Or use `jax.lax.dynamic_index_in_dim` with explicit indices**
  + `jnp.where(mask, ..., 0)` instead of `dynamic_slice`.  Slower
  but no silent clamp.
- **Or add a trace-time assert**: enumerate all possible offsets
  and verify max + size ≤ axis_size.  In this kernel the
  band_chunk_ranges are closure constants, so the check is
  cheap and catches the bug at compile time.

The plan §5.3 should grow a "JAX API hazards" subsection capturing
this.

## 4.  Proposed canonical fixture grid for pair-pipeline-like tests

For any kernel that takes pre-built `psi_l_X` / `psi_r_X` and
streams `psi_l_Y` / `psi_r_Y` via a band-chunked scan, future
bit-identity tests should include ALL of the following sub-gates.
The grid is structured so the bugs each fixture probes are
independent and orthogonal:

| Sub-gate | Setup | What it probes |
|---|---|---|
| **happy_path_charge** | charge, L=R=(0,nb_total), nb_total = bc_size · k | baseline correctness + ULP-class drift signature |
| **single_bc** | `n_bc = 1` (`band_chunk_size >= nb_total`) | scan-with-one-iter degenerate case |
| **short_final_bc** | `nb_total = bc_size · k + r`, `0 < r < bc_size` | pad-band-zero contract + back-pad bug on trailing edge |
| **asymmetric_L_R** | `band_range_left != band_range_right`, partial overlap | per-bc offset math when bcs straddle window edges (the bispinor production case) |
| **bispinor_gamma_nonzero** | `vertex_mu_L != 0` (γ̃ ≠ I) | γ̃-fold path with non-identity perm/phase |
| **pseudobands_pre_divide** | non-trivial `norms_l` / `norms_r` pre-divided into `psi_l_X` / `psi_r_X` | linearity / pre-divide contract |

**Plus** (uncovered by my Round-6 G1 but should be added for
future iterations):

| Sub-gate | Setup | What it probes |
|---|---|---|
| **traced_r_start_nonzero** | `r_start_dyn = some_nonzero_int` | `r0_local = r_start + y_idx * r_loc` arithmetic when r_start ≠ 0 |
| **multi_rchunk_concat** | run kernel twice with different `r_start_dyn`; verify concat matches single-shot reference | jit re-compile / tracer-leak check on differing r_chunk_size (the canonical Path-D remainder-chunk gate) |
| **bispinor_asymmetric_LR** | combination of bispinor γ̃^1 + L=(0,5), R=(3,8) | the bispinor-production-actual case; both bugs at once |
| **k_axis_unfold_nontrivial** | nk > 4, kgrid asymmetric (e.g. 3×4×2), with random kvecs | tests the (nkx, nky, nkz) reshape after pair-density, separable Bloch phase |

The first six form the **canonical minimum** for shipping the
kernel.  The four extras form the **bispinor-production-readiness
extension**.

## 5.  Why the synth-WFN approach worked

Agent 4's G1 was a CPU-side synth-WFN unit test on a 1×1 mesh, NOT
an end-to-end MoS2 3×3 ζ-fit.  Three properties made it effective:

1. **Adversarial sub-gate selection**: I enumerated the plan's
   §5.3.1-§5.3.6 named edge cases and built one test per case, even
   though the production MoS2 setup doesn't trigger most of them.
2. **Cheap iteration**: each sub-gate runs in <1 s on CPU.  No
   SLURM allocation needed.  This made it natural to add more
   sub-gates (and discover more bugs) without sunk-cost reasoning.
3. **Direct reference computation**: my reference built
   `psi_l_Y` / `psi_r_Y` via a direct `to_rchunk` on the full G-flat
   tile + a global einsum.  The reference computes the SAME math
   the new kernel SHOULD compute, but via a different path.  Any
   indexing-arithmetic bug in the new kernel surfaces as a
   numerical disagreement.

E2e tests against a pre-recorded baseline have the inverse
property: they're easy to write (just rerun the production
pipeline) but only exercise the *specific shape* the baseline was
computed with.  A baseline computed on MoS2 3×3 charge can only
catch bugs that fire on MoS2 3×3 charge.

**Combined**: synth + adversarial fixtures for design-level bug
detection, e2e for regression detection.  Both belong in CI.  The
methodology error in Round 6 was: Agent 2 ran only the e2e test
and called it "G1 passed".  The fix is procedural — **G1 should
include the synth fixture grid by convention**, not just an e2e
run.

## 6.  Process lesson for Round 7+

### What the plan said vs what the test missed

`round5_unified_plan.md` §5.3 EXPLICITLY named both failing
sub-gates:

> §5.3.2 — Asymmetric L/R band windows (nb_L != nb_R).  Plan §5.3.2.
> §5.3.3 — Short final bc (last bc has bpd_per_bc < bpd_max).  Plan §5.3.3.

Both were marked "**Gate**: write a unit test ..." in the plan.
The plan correctly identified the design's hidden assumptions.
But Agent 2's Round-6 G1 was an e2e MoS2 3×3 run that
**incidentally satisfied none of those assumptions** —
nb_total=80 is divisible by bc_size=16, L=R=full, so the test
shape didn't activate either sub-gate.

The methodological fix:

- **Plan-named sub-gates are MANDATORY**, not aspirational.  If
  §5.3.x calls out an edge case, the validation gate (G1) must
  exercise it explicitly.
- **The implementer cannot self-validate edge-case correctness
  via an e2e test alone.**  E2e tests exercise the implementer's
  own mental model; if a bug lives in an unhandled corner of the
  model, the e2e test won't see it.  A *separate* test author
  (here: numerics validator) running the plan's named sub-gates is
  what catches design-corner bugs.

### How to apply this to future PR-shaped rounds

- The Round-N implementer's "G1 passed" should require **two
  separately-authored test sets**: (a) the implementer's own
  regression test, (b) the validator's adversarial fixture grid
  per the plan's §5.3-equivalent edge-case enumeration.
- For pair-pipeline-like kernels specifically: the six canonical
  sub-gates from §4 above are the standing minimum.

## 7.  What I would do differently if I had this audit on day 0

The Round-5 plan §5.3 was correct in naming the edge cases.  The
gap was the *test-authoring assumption*: I wrote §5.4's G1 gate
spec but didn't make it explicit that the validator (me) is
responsible for the sub-gate grid, not the implementer (Agent 2).

In a hypothetical Round-0:

- §5.4 should have read: "Agent 4 (validator) writes the
  bit-identity test scaffold including ALL of §5.3.1-§5.3.6
  sub-gates BEFORE Agent 2 commits.  Agent 2's commit is judged
  against the pre-existing scaffold."
- The scaffold lives in the repo as `tests/_pair_pipeline_
  fixtures.py` (or similar), and any future pair-pipeline-like
  rewrite imports the canonical fixtures.

This audit doc is the basis for that scaffold.  Round 7's commit
that lands the back-pad fix should also lift
`tests/test_zq_from_psi_sm_bit_identity.py`'s six sub-gates into
a reusable fixture module so the next round-N pair-pipeline
rewrite doesn't re-author them.

## 8.  Recommended canonical fixture module signature (for Round 7+
       cleanup commit)

```python
# tests/_pair_pipeline_fixtures.py

@dataclass
class PairPipelineFixture:
    name: str
    band_chunk_ranges: tuple[tuple[int, int], ...]
    band_range_left: tuple[int, int]
    band_range_right: tuple[int, int]
    nb_total: int
    ns: int = 2
    gamma_L: tuple | None = None
    gamma_R: tuple | None = None
    norms_l: np.ndarray | None = None
    norms_r: np.ndarray | None = None
    r_start: int = 0


CANONICAL_FIXTURES = [
    PairPipelineFixture("happy_path_charge", ((0,4),(4,8)), (0,8), (0,8), 8),
    PairPipelineFixture("single_bc",         ((0,8),),      (0,8), (0,8), 8),
    PairPipelineFixture("short_final_bc",    ((0,4),(4,8),(8,9)), (0,9), (0,9), 9),
    PairPipelineFixture("asymmetric_L_R",    ((0,4),(4,8)), (0,5), (3,8), 8),
    # bispinor + pseudobands fixtures with explicit gamma / norms args
    # ...
]


@pytest.mark.parametrize("fixture", CANONICAL_FIXTURES, ids=lambda f: f.name)
def test_pair_pipeline_bit_identity(fixture):
    """Generic bit-identity test driver for any (psi_l_X, psi_r_X,
    psi_G_store) → Z_q kernel.  Pluggable kernel under test passed
    via fixture's `kernel_fn` attribute."""
    ...
```

This module becomes the entry point for *any* future kernel that
streams pair density over band-chunks.  It captures the canonical
fixtures from §4 + the `_reference_z_q` numpy helper from my G1
test.  Cleanup commit owner: TBD (next round's implementer or a
dedicated test-infrastructure commit).

---

**Headline take-away**: the back-pad bug was not subtle — it was
ONE clamped offset two lines into the per-bc body.  But it was
hidden by the production-shape choices in Agent 2's self-G1
(charge + bc-divisible nb_total + L=R=full).  The plan's §5.3
correctly named the missing sub-gates; the methodology error was
treating them as documentation rather than as MANDATORY test
fixtures.  Round 7's cleanup commit should canonize the six
fixtures so this class of bug never re-occurs.

Agent 4 round 7 methodology done
