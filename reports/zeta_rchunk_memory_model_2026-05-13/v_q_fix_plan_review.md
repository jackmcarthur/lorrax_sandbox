# V_q fix plan — reviewer response (fresh-context Agent B)

Reviewer: JAX/SPMD/XLA-savvy, no prior context. Read order: plan
(`v_q_fix_plan.md`), `v_q_g_flat.py:1-545`, `isdf_fitting.py:450-745`
(`z_q_from_psi_sm._local` post-Round-7 template), `round8_efficiency_audit.md`.

## Reviewer-ask answers

### 1. Issue 0: 0a (orbit-aware kmeans) vs 0b (decouple IBZ from sym_perm)?

**Verdict: BOTH — land 0b refuse-path now, schedule 0a separately. (CONCERN
about plan's "approximate unfold" suggestion.)**

The current silent fallback is the worst possible behaviour: it doubles
compile cost, multiplies kernel cost by `|G|`, and the user has no idea
the IBZ optimisation has stopped working. Land 0b's *hard error* variant
this round — one `raise RuntimeError` at `v_q_g_flat.py:183` after the
existing log line. The "KDTree approximate unfold" variant is **not
acceptable for production GW**: V_q enters Σ_X with sub-meV target
accuracy; introducing grid-spacing-scale unfold error to dodge a
centroid-generation bug breaks the validation contract everywhere
downstream. 0a (orbit-aware kmeans) is the real fix and should be its
own initiative — it touches `centroid/kmeans_isdf.py` and affects every
caller of `compute_centroid_sym_perm`, not just V_q. Scope creep into
Round 9 if attempted here.

### 2. Issue 1 Option A (reshard first) vs Option B (slice first)?

**Verdict: Option B (slice first, then reshard).**

The leading axis of `zeta_*_3d` is the q dimension (length 1 here, but
unsharded — sharding spec is `P(('x','y'), None)` at line 95 / 99, so
axis 0 is replicated, axis 1 is flat-`('x','y')`, axis 2 is unsharded).
`zeta_L_3d[0]` therefore removes a *replicated* axis: every rank holds
the full slice already, no cross-rank data movement. The post-slice
sharding is `P(('x','y'), None)` on the rank-2 result, which then
reshards to `P('x', None)` (a flat→axis transition) as a clean,
separately-planned op. Option A's `with_sharding_constraint(zeta_L_3d,
P(None, 'x', None))` asks XLA to reshard a 3-D `P(('x','y'), None)` to
`P(None, 'x', None)` — that's the *same* flat→`'x'` transition the
current code does, just applied before slicing; no clear win, and you
lose the "slicing an unsharded axis is local" property as the easy
explanation to XLA. B is the structurally cleaner ask.

### 3. Issue 2 scan-over-q: sharding-trap risk?

**Verdict: CONCERN — two real gotchas, both manageable.**

(a) `v_q_dev[q]` at line 439 currently uses Python `__getitem__` with a
concrete int. Under scan with a traced `q_idx`, this lowers to
`lax.dynamic_index_in_dim` automatically — fine. But `v_q_dev` is
sharded `P(None, None)` (replicated, line 392) so per-rank gather is
local. Make it explicit: rewrite as `jax.lax.dynamic_index_in_dim(v_q_dev,
q_idx, axis=0)` inside the body so the IR is unambiguous.

(b) The plan's note on carry sharding is right but understates the
risk: under `jit`, the *output* sharding of the body (which becomes the
next-iteration carry) must match the carry's input sharding exactly or
XLA emits a copy each iter, defeating donation. The current per-q jit
returns `V_new` at `V_sh = P(None, 'x', 'y')`; pin the carry the same
way (output WSC at line 131 stays). The `dynamic_update_slice` at line
129 with a leading `None`-sharded axis is fine for this — it's the
already-correct pattern.

No `shard_map`-internal scan needed (and the plan correctly excludes
it). The `isdf_fitting` template uses scan-inside-shard_map because the
*body* needs Manual mode for `io_callback` + `all_gather`; V_q's body
is pure SPMD-Auto with WSCs, so scan-at-`jit`-level is the right
analogue.

### 4. Issue 3 WSC audit: this round or later?

**Verdict: this round, light touch only.**

I count **10 WSCs in the per-q kernel body** (lines 95-138), not 13 —
plan-stated count is off but minor. After Issue 1's split, 1-2 of those
become redundant (the explicit `blk_x_sh` / `blk_y_sh` reshard is now
the only sharding annotation on `zeta_L` / `zeta_R`; the upstream
`blk_xy_sh` on `zeta_L_3d` may also be redundant once XLA sees the
slice-then-reshard pattern). The line 105 WSC on `jnp.zeros` and the
line 125 WSC on the post-scan `V_q` are likely redundant (scan output
inherits carry sharding). Drop those four; keep the boundary WSCs
(input v_q at 102, output V_new/g0_new at 131/138). Target ≤ 6, not the
plan's ≤ 5 — pinning the einsum operands' sharding (`zeta_L`,
`zeta_R`) is load-bearing for the `(p_x, p_y)` decomposition and worth
keeping explicit.

### 5. Anything missing about V_q's relationship to the wfn fix?

**Verdict: NO structural missing piece, but one process miss.**

V_q's bottleneck is *compile time*, the wfn fix's was *runtime memory*.
Different root cause. The template the plan cites is correct for the
shape of the fix (scan-instead-of-Python-loop, WSC-density audit, clean
boundary slicing) but the wfn fix's `_slice_local_tile_bc` / `_bpd_max`
/ io_callback machinery doesn't apply to V_q — `zeta_*_all` is already
device-resident in a single batched read at line 417. The plan
correctly doesn't try to import that machinery. One thing the plan
doesn't mention: after Issue 2, the *outer* read at line 417 is the
only remaining `block_until_ready`-equivalent sync point (it's
implicit, via the `concatenate`-replaced batched read). That's fine
and probably necessary — but worth a one-line comment in the
implementation.

### 6. Bonus — MoS2 3×3 orbit-closure failure?

**Verdict: YES — confirmed from log.**

`runs/MoS2/00_mos2_3x3_cohsex/D_gflat_charge_karmb_2026-05-13/profile_launch.log:172`:
"V_q tile: centroid orbit closure failed — falling back to full-BZ
iteration. … Total failures: 588 / 1280." Same exact error class as
CrI3 6×6. Implication: MoS2 3×3 is **already running full-BZ**, so the
Issues 1+2 baseline timing is at 9 q's (full BZ), not the IBZ count.
Issue 0 fix will give the MoS2 baseline an additional speedup on top
of Issues 1+2 (3 q's instead of 9 = ~3× on V_q+Σ_X), which is
load-bearing for the G_v3 < 10 % target. Note: the "9 q's" figure in
the log is full BZ for MoS2 3×3 (3×3×1 = 9); IBZ would be ~3 q's
under C_3v.

## Additional checks

**WSC count.** 10 WSCs in `_make_per_q_kernel` body (lines 95, 99, 100,
101, 102, 105, 125, 131, 135, 138). Classifying:
- *Boundary, keep*: 102 (input v_q), 131 (output V_new), 138 (output
  g0_new). 3 WSCs.
- *Operand-pin for einsum, keep*: 100 (zeta_L → `'x'`), 101 (zeta_R →
  `'y'`). 2 WSCs.
- *Redundant after Issue 1*: 95, 99 (the offending pre-slice pin).
  Drop both — slice-first leaves the rank-2 result at `P(('x','y'),
  None)` which the operand-pin WSC then transitions.
- *Likely redundant (scan-carry inherits)*: 105 (zeros init), 125
  (post-scan V_q), 135 (g0_q block).
Estimated post-audit: 5-6 WSCs.

**`v_q_dev[q]` indexing under scan.** `v_q_dev` is sharded
`P(None, None)` (fully replicated, line 392). Under `lax.scan` with
traced `q_idx`, `v_q_dev[q_idx]` lowers to a local `dynamic_index_in_dim`
— fine, no extra collective. **Be explicit**: write as
`jax.lax.dynamic_index_in_dim(v_q_dev, q_idx, axis=0)` so reviewers
don't have to chase the Python-`__getitem__`-on-traced-int dispatch.

**Donation pattern under scan.** Current `donate_argnums=(0, 1)`
donates `V_acc` and `g0_acc` per outer jit call. Under `lax.scan`
inside a single jit, donation is implicit through the scan carry: XLA
aliases the carry buffer in-place across iterations. The risk is that
the *outer* jit (which the scan now lives inside) needs the carry as
`donate_argnums=(0, 1)` again so the *first* iter writes into the
caller's `V_acc` buffer, and so the post-scan return is the same
buffer. Pattern from the plan is correct; verify in HLO that
`scan-while` aliases the carry slot (look for "input_output_alias" in
the dumped HLO).

## Verdict

**BLESS** the orchestrator to proceed, with three required tweaks:

1. Issue 0: implement as **hard error, not KDTree approximation**.
   Schedule orbit-aware kmeans (0a) as a separate initiative — do not
   bundle.
2. Issue 1: prefer **Option B** (slice first, then reshard).
3. Issue 2: write `v_q_dev[q]` as explicit `dynamic_index_in_dim`
   inside the scan body; add an HLO check that the scan carry aliases
   `V_acc` / `g0_acc`.

Issues 3 and the test plan are good as written; my WSC count differs
(10 vs plan's 13) but the prune target is roughly the same.

**One-line summary**: BLESS with three tweaks (hard-error on Issue 0,
Option B on Issue 1, explicit `dynamic_index_in_dim` + HLO carry-alias
check on Issue 2).
