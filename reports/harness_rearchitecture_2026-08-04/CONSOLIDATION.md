# Frontera consolidation runbook — 2026-08-05

Ordered checklist for the owner's stated next step: consolidate the
Frontera work. Compiled from this session's findings; steps on Frontera
are marked [FRONTERA] (this cloud session cannot do them), the rest can
be done from anywhere once the push lands. Facts referenced below carry
their source in parentheses.

## Phase 0 — pre-push verification [FRONTERA]

1. `git -C /work2/08271/jackmc/frontera/lorrax status` — confirm clean
   tree on `fix/zq-band-gather-device-invariance`; run the login AST
   gates + fastloop check one more time at HEAD (the standing pre-commit
   bar).
2. Decide the two UNMERGED branches (both certified, neither merged nor
   pushed — CLAIMS 48-50, 49):
   - `fix/slab-io-audit` @ ca9bad5b (worktree wt-slabio): the phdf5
     bounds fix + implicit padding. CLAIMS 50/51 certify it.
   - `fix/bispinor-zeta-reuse` @ 82cefa2: ζ reuse for bispinor +
     provenance extension. CLAIMS 49 certifies it.
   Recommendation: merge both into fix/zq on Frontera (fastloop + the
   relevant unit gates after each), so ONE branch carries the whole
   certified state. If a merge is deferred, push the branch anyway —
   unpushed certified work is the risk being eliminated.

## Phase 1 — push (the whole point)

3. [FRONTERA] Push `fix/zq-band-gather-device-invariance` (and any
   still-unmerged fix branches) to origin. ~150+ commits; this is the
   single highest-value step in the whole consolidation — every
   off-cluster agent currently orients against 2026-07-22 (verified: no
   decisions.md, no ffi/gate.py, no docs/environment/, zero "frontera"
   mentions on origin).
4. Tag the certified state (e.g. `certified-2026-08-05`) so sandbox
   as-of hash pins have a stable name to cite.
5. Decide main: either fast-forward/merge fix/zq into main, or commit a
   one-line STALE banner into main's AGENTS.md pointing at the branch.
   Leaving main silently 150 commits stale is the one option that keeps
   poisoning fresh agents.

## Phase 2 — truth repair (doable from any session post-push)

6. Rewrite repo AGENTS.md: delete the Perlmutter/Shifter/lxrun/uv
   sections (that environment is gone per sandbox orientation), state
   Frontera + container + FFI-required + square-mesh current state, keep
   the CONVENTIONS block (it feeds RULES_v2), link DESIGN.md at line 1.
7. Land docs/architecture/DESIGN.md from DESIGN_draft.md (owner edit
   pass first — it encodes owner rules, so the owner should bless the
   wording).
8. Reconcile the sandbox: AGENTS.md "Current state (2026-07-31)" block
   and the ~150-commits-NOT-pushed paragraph are stale after the push;
   update both, and repoint any sandbox doc that cites unpushed-only
   files.

## Phase 3 — docs Curator pass (the 07-31 sandbox rebuild recipe, applied
to the repo)

9. docs/dev/ (2.1 MB archive + notes/plans/progress, 24 files): purge or
   archive with dated reasons, re-verify survivors, sign with stated
   deferrals — exactly the recipe that produced the current sandbox.
   Rules mined from docs/dev plans (RULES_seed §7) should be re-homed
   into DESIGN.md / scoped AGENTS.md before their source docs are
   archived, or they vanish with the archive.
10. Doc-correction-closes-claims rule takes effect: the open KNOWN rows
    that are purely documentation (the distributed-tier framing, the
    P=nq ceiling, the ridge docstring) get fixed in this pass and their
    rows closed.

## Phase 4 — rules and gates (RULES_v2 §J order)

11. Re-run the RULES_seed sweep against the consolidated tree (the
    2026-07-22 sweep misses everything the push adds; the sweep prompts
    are in this session's report).
12. Adopt rules_gate.py into tests/ with its allowlist regenerated
    against the consolidated tree (origin baseline for scale: 95
    device_put sites / 30 files, 108 raw jnp.fft sites / 21 files —
    expect different numbers after the push). Wire into the login gate
    runner alongside the AST suites.
13. F0 refusal helper + the F1-F5 runtime checks (port the
    rank_criterion refusal pattern).
14. Scoped AGENTS.md drops (FFT→src/common, FFI→src/ffi, symmetry idiom
    pair→src/common) per RULES_v2 placement map.

## Phase 5 — evidence layer (EVIDENCE_DESIGN.md §3)

15. harness/run_record.py in the sbatch template (needs the real timing
    table format — that's why this waits for the push).
16. GATES.md numeric rows → machine-readable reference dict read by the
    parity harness; scripted re-pin path.
17. Weekly sentinel fastloop (queue permitting) tagged in the records.

## Deliberately NOT in this runbook

CI/GitHub Actions wiring, the planted-defect battery, TASTE.md seeding,
and the metrics trend tool — all real, all in report.md/RULES_v2, none
blocking consolidation. Do the push first; everything above Phase 1
compounds on it.
