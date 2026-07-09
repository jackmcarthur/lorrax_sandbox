# Where to attack the B4 (IBZ symmetry) smear — the real decomposition

_2026-07-02, after reading the cores rather than the catalog. Companion to MAP §2/§4._

B4 (IBZ symmetry) is one of the two most-smeared concerns (MAP §2). C6 (memory
planner) is largely done, so B4 is the remaining top structural target. But the
map's framing ("≥6 parallel unfold helpers → unify them") is wrong. Reading the
actual code, B4 is **three layers that need opposite treatment**:

## Layer 1 — the sym TABLES: already healthy, leave it
`sym.irr_idx_q` / `sym.sym_idx_q` (q→IBZ-parent) live once on the SymMaps object;
the centroid/r-grid permutations live once in `orbit_syms.compute_{centroid,rgrid}_sym_perm`.
Every consumer (v_q_g_flat, compute_vcoul, zeta_loader, v_q_tile) *reads* them.
This is the single-source we'd want. **Not a target.**

## Layer 2 — the IBZ-resolution ORCHESTRATION: the clean, safe, high-value target
`v_q_g_flat._resolve_ibz_q_list` (v_q_g_flat.py:152) and the block at
`compute_vcoul.py:929` are **near-verbatim copy-paste** (~40 lines, same
`compute_centroid_sym_perm` call, the same word-for-word `extend_trs=True` comment
incl. the audit-report reference). They differ only in a variable name, a print
label, tile-specific `nq_total`/`q_list_for_tile` bookkeeping — **and one real
drift**: `v_q_g_flat` honors `LORRAX_FORCE_FULL_BZ`, `compute_vcoul` does NOT.
The copy has already diverged, and the divergence is a latent bug (the tile/bispinor
V_q path can't be forced to full-BZ). Single-source this → removes the dup AND
heals the drift. This is the concrete B4 win.

## Layer 3 — the unfold KERNELS: NOT a merge target
`unfold_v_q` (device shard_map + all_to_all, μ/ν centroid-perm), transverse
`_unfold_v_q_ij_ibz_to_full` (rank-2 R_cart tensor), zeta `_unfold_q_full_bz`
(jit + out_shardings, r/μ perm), `unfold_psi` (host numpy, spinor+phase+umklapp+TRS)
share a *conceptual shape* but differ in object, sharding, tensor rank, and
host-vs-device. A unified sym-action would be a leaky wrapper or a fat abstraction
— over-abstraction that would break physics if forced. **Leave separate; just have
them all pull from Layer-1 tables and stop hosting them inside C5 readers.**

## The blocker that makes ALL of this safe: gate #4 (IBZ-vs-full-BZ)
There is **no end-to-end IBZ-vs-full-BZ equivalence gate** (test_zeta_loader /
test_trs_unfold_centroid_perm test the tables/loader, not pipeline agreement). The
current e2e gate is IBZ-only — it CANNOT catch a symmetric-but-wrong unfold,
because the frozen reference was generated with the same unfold. The seam for the
right gate already exists: `LORRAX_FORCE_FULL_BZ=1` computes V_q at all full-BZ q
directly. Run the MoS2 fixture (already have it) both ways, assert eqp agreement →
that gate fails on any wrong unfold/resolution refactor.

## Recommended sequence
1. **Gate #4 first (linchpin).** Extend `LORRAX_FORCE_FULL_BZ` to the tile path
   (fixes the Layer-2 drift as a side effect), then add the IBZ-vs-`FORCE_FULL_BZ`
   eqp-agreement gate on MoS2. Cheap (seam nearly exists), unblocks everything,
   surfaces the drift bug. Without it, ALL B4 work is flying blind (MAP §5).
2. **Single-source `_resolve_ibz_q_list` (Layer 2).** One home (orbit_syms or a
   symmetry seam), both V_q paths call it. Gated by #1.
3. **Un-smear B4 from the C5 readers (MAP §4 #4/#8).** Move ψ-unfold out of
   `wfn_loader`, ζ-unfold + local table derivation out of `zeta_loader`, so
   "readers read." Kernels stay separate; symmetry stops living inside I/O.
   Gated by #1.

**Do NOT** unify the unfold kernels (over-abstraction — the Phase-3 lesson, now 3×).

## Why this over the other open targets
MAP §4's worst files are `gw_init` (#1) and `v_q_tile` (#3) — both already partly
addressed by the C6 work. The remaining *smear* (as opposed to big-file) disorder
is B4, and B4's safe payoff is Layers 1→2 above under gate #4. The kernels (Layer 3)
look like the target but aren't. Attacking the gate first is the move that both de-
risks and exposes a real bug — the highest-leverage single step left in the refactor.

## Progress (2026-07-02 pm)

- **#1 gate DONE** (`1479162`): `test_ibz_full_bz_equivalence` — runs the MoS2 3x3
  fixture IBZ vs `LORRAX_FORCE_FULL_BZ=1`, asserts sigma_diag agrees. Static COHSEX
  (algebraic unfold) holds at ~1e-9 eV; codified with atol 1e-6. This is the safety
  net that catches a wrong charge-path unfold. (GN-PPM shows a benign ~0.12 meV
  q-set path-dependence from the nonlinear PPM fit — noted, not gated.)
- **Tile-path drift FIXED** (`0d7ba06`): `compute_vcoul` now honors `FORCE_FULL_BZ`
  (it was a drifted copy that silently ignored it → tile/bispinor V_q couldn't be
  forced full-BZ). Only affects the debug flag; production unchanged; 3 gates green.
- **#2 full single-source DEFERRED — the copies diverged.** Beyond `FORCE_FULL_BZ`,
  the two blocks differ in the full-BZ *fallback* (g-flat returns the full q-list;
  compute_vcoul returns `None` for the tile path to build downstream). So it's NOT a
  mechanical merge (the "diverged copies" trap, again). And the tile path is bispinor
  — **ungated**. Right next step for #2: build a **bispinor IBZ-vs-full-BZ gate** (now
  possible since the tile path honors `FORCE_FULL_BZ`), THEN single-source under it.
- **#3 reader un-smearing**: unchanged, still the bigger structural item, gated by a
  charge-path gate that now exists (#1) for the ψ/ζ charge unfolds.

Net: the linchpin (gate) landed, the drift bug is fixed, and the next B4 step is
precisely scoped (bispinor gate → then single-source). No over-reach into the
ungated tile path.

## Update (2026-07-02, later) — the drifted copy was in a DEAD subsystem

Chasing the `_resolve_ibz_q_list` single-source revealed the drifted `compute_vcoul`
copy lives in the r-space V_q "tile" subsystem, which is DEAD: `fit_zeta_to_h5`
writes ζ exclusively `G_flat`, so the r-space dispatch is unreachable; the live
bispinor path (`compute_V_q_bispinor_g_flat_to_h5`) already runs the gate-verified
`_resolve_ibz_q_list`. So no bispinor gate was needed — the honest cleanup was
deleting the dead subsystem, not gating it.

**Deleted ~3k lines** (commits `8369ecc`+`d4bb3ba`, MAP §4 #3): all of `v_q_tile.py`
(relocating its one live symbol `_unfold_g0_ibz_to_full` into v_q_g_flat), the legacy
`compute_V_q_bispinor_to_h5` + 2 dead builder factories, `make_v_munu_chunked_kernel`
+ helpers, the r-space tail of `compute_all_V_q` (now raises for non-G_flat), the
gw_init legacy else-branch, and `compute_bare_coulomb_sphere_idx`. Retargeted
`test_v_q_bispinor_helpers` at the LIVE builder (added projector coverage it lacked).
3 e2e gates + 242 unit green.

MANIFEST-MISS LESSON: the mapping subagent's test-scan missed `test_v_q_bispinor_helpers.py`
(it tested two of the deleted helpers) — the full unit run caught it. Always run the
whole suite after a big delete; a grep-based "no test references" claim is not enough.
