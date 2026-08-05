# Round-8 efficiency audit — zeta-rchunk memory model refactor

Fresh-context auditor. Independent critique of the 8-round multi-agent push
on `sources/lorrax_B` branch `agent/zeta-bc-scan-shardmap`. No politeness;
the orchestrator explicitly asked whether this has been unreasonably heavy.

## 1. Verdict

**Yes, the work was unreasonably heavy by a factor of ~3-5×.** A competent
single engineer with the same a-priori knowledge the agents had at the end
of the planning round could have landed the same structural fix in roughly
~600-900 lines of net source change, 2-3 commits, ~500 lines of focused
tests, and ~6-10 hours of wall time, with one short design doc. The actual
output is ~1300 net lines of source delta plus ~700 lines of test scaffold,
7 commits with one major rewrite-of-rewrite, **38 markdown files totalling
~129 000 words of agent process artifacts**, and ~14 hours of orchestrated
wall time from session start (`cdd0fba` at 16:07 PDT) to BLOCKER fix
(`c796420` at 00:26 PDT next day).

The structural fix is real and load-bearing. The numerics validation is
real and load-bearing. The 129 kword paper trail and the rewrite-of-the-
rewrite at f567aa0 are not.

## 2. Quantitative facts

### Source / test deltas (vs `main`)

```
src/common/isdf_fitting.py                        445 +/-  (kernel rewrite)
src/common/load_wfns.py                           252 +/-  (gflat_to_rmu)
src/common/psi_G_store.py                         271 +/-  (slice + dead-code)
src/common/wfn_transforms.py                      413 +/-  (gflat_to_rchunk/rmu)
src/gw/aot_memory_model/kernels/fit_one_rchunk.py 149 +/-  (stub refresh)
src/gw/gw_config.py                                 7 -/-  (knob removal)
src/gw/gw_init.py                                   8 +/-  (knob removal)
tests/test_io_callback_nested.py                  325 new  (G0 gate)
tests/test_psi_g_store.py                         122 +/-  (3 new tests)
tests/test_wfn_transforms.py                      249 +/-  (helper tests)

TOTAL                                  +1791 / -450
                                       ~1340 net new lines
```

The `tests/test_zq_from_psi_sm_bit_identity.py` file the BLOCKER commit
references (471 lines) is **not staged into the branch** — it was a Round-7
validator artifact that lived only on disk.

### Per-commit churn (insertions+deletions)

| Commit  | Time         | Files | + / -      | Notes |
|---------|--------------|-------|------------|-------|
| cdd0fba | 16:07        | 3     | +208 / -1  | Scaffolding helpers |
| d7eaf1c | 18:00 (+1h53)| 2     | +361 / -1  | gflat_to_rchunk standalone |
| 3d0636c | 18:06 (+6m)  | 1     | +10 / -1   | Cache-key hash fix |
| 3606138 | 18:23 (+17m) | 3     | +604 / -153| gflat_to_rmu mirror |
| 5cadd4b | 18:46 (+23m) | 6     | +376 / -277| **First integration** |
| f567aa0 | 23:49 (+5h03)| 9     | +886 / -698| **Second integration (replaces 5cadd4b)** |
| c796420 | 00:26 (+37m) | 1     | +37 / -10  | BLOCKER back-pad fix |
| **Sum** |              |       | **+2482 / -1141** | |

Total churn: **3623 lines touched** to produce a 1340-line net change.
Churn-to-keep ratio: **2.7×**.

The f567aa0 commit alone deletes 698 lines that were added in earlier
commits within the same 8-hour window (much of 5cadd4b's psi_G_device_full
property, the `gflat_to_rchunk` standalone helper, the
`gflat_to_rchunk_chunk_size` cohsex knob plumbing, and the 4
`test_gflat_to_rchunk_*` tests).

### Reports directory

- **38 markdown files**, ~129 200 words.
- Of that, the four largest (`round5_unified_plan.md` 13.2 k words,
  `agent_1.md` 6.9 k, `agent_3.md` 6.6 k, `round8_unified_fft_pipeline.md`
  6.1 k) account for ~25 % of total wordcount.
- Eight separate per-round "discussion" files (`round3_discussion`,
  `round4_discussion`, `round4_improvements`, `round4_next_tasks`,
  `round5_discussion`, `round6_discussion`, `round7_discussion`,
  `round8_discussion`) totalling ~28 k words.
- The `prompts/` subdirectory has ~30 launch prompts (one per agent per
  round). Not counted in the 129 k.

For comparison: the entire LORRAX `sources/lorrax/AGENTS.md` plus the
four canonical skills (`build_inputs`, `execute_workflow`, `compare`,
`checkpoint`) is well under 20 k words.

### Test count

- 6 G0 tests in new `test_io_callback_nested.py`.
- 5 new bit-identity tests in `test_wfn_transforms.py` for gflat_to_rmu.
- 3 new bit-identity tests in `test_wfn_transforms.py` for gflat_to_rchunk
  (the helper is now **deleted**, so these tests no longer have a target
  — actually, looking at the file, they exercise behavior that's now
  inlined into `z_q_from_psi_sm` so the bit-identity contract is
  indirect).
- 6 new tests in `test_psi_g_store.py` for `_slice_local_tile_bc` +
  `psi_G_device_full` (the latter property is now **deleted**, so its
  guard-rail test is dead).
- 6 sub-gate tests in the never-committed `test_zq_from_psi_sm_bit_identity.py`.

## 3. Concrete bloat findings

### B1. The `5cadd4b` → `f567aa0` rewrite-of-the-rewrite (5 hours of wasted execution)

5cadd4b introduced `psi_G_device_full` as a lazy device property + a
plumbed `gflat_to_rchunk_chunk_size` cohsex knob. It validated to within
math-preserving-reorder precision against the lorrax_A baseline. Within 5
hours the design was scrapped wholesale: `psi_G_device_full` deleted,
`gflat_to_rchunk_chunk_size` deleted (config field, gw_init plumbing,
fit_zeta_to_h5 kwarg, kernel kwarg, four cohsex.in references), the
standalone `gflat_to_rchunk` deleted, and its four bit-identity tests
deleted. Round 4-5 then re-discovered that the all-at-once `psi_Y_full`
materialization (which 5cadd4b kept in `_kernel`) was still allocating a
30 GiB transient — the structural fix had stopped halfway up the call
chain.

This is a textbook "scaffolding became real code, then real code became
scaffolding" cycle. Had the planning agents in rounds 1-2 understood
that the boundary needed to be **inside** the consumer (z_q_from_psi_sm),
not in a standalone forward helper, commit 5cadd4b plus the four
gflat_to_rchunk tests plus the `gflat_to_rchunk_chunk_size` cohsex plumb
(~400 LOC added and then deleted) would not have happened. They had the
HLO dump showing 4× FFT-box concurrent slots in the bc-loop *before*
5cadd4b; they should have read it as "the slot pile-up is in `_kernel`,
not in a missing-helper" rather than "we need a helper that does this in
one shot".

### B2. `gflat_to_rchunk` standalone helper (d7eaf1c) is now dead

The 259-line `gflat_to_rchunk` function added in d7eaf1c was deleted in
f567aa0 (commit message: "Delete gflat_to_rchunk from wfn_transforms.py
+ its _GFLAT_TO_RCHUNK_CACHE"). All current references in the codebase
are docstring mentions of "mirrors gflat_to_rchunk" inside `gflat_to_rmu`.
The 3 bit-identity tests for it that were added in d7eaf1c **were also
deleted** in f567aa0.

Net source contribution of d7eaf1c after f567aa0: **zero lines kept**.
Cost incurred: ~360 lines of code + tests + cache infrastructure +
content-hash bug discovery + commit + report writeup.

The only piece of d7eaf1c that survived is 3d0636c (content-hash on
qvec_frac), which is a 10-line one-liner cache-correctness fix that
could have been a 5-minute drive-by commit on `main`.

### B3. `_slice_local_tile_bc` churn (added, deleted, restored)

Per commit cdd0fba, the helper was scaffolded. Per 5cadd4b's body
("dropped... _slice_local_tile_bc... fields") it was deleted as
dead-plumbing. Per f567aa0's body ("PsiGStore._slice_local_tile_bc +
_bpd_max RESTORED (commit cdd0fba helpers that were deleted in
5cadd4b)") it was restored.

Net contribution: kept. Cost: the function got written, written-off,
and rewritten across 8 hours. The scaffolding commit cdd0fba was not
actually scaffolding for the path that landed — it was scaffolding for
the path the agents tried and rejected. The fact that the same name
came back is partly luck, partly that the io_callback pattern was
always going to need this signature.

### B4. `to_rchunk_inner` / `to_rmu_inner` — speculative primitives

These were introduced as "callable from inside another shard_map body or
a lax.scan body" pure helpers. `to_rchunk_inner` has been inlined into
`z_q_from_psi_sm._local` in f567aa0; `to_rmu_inner` is still called by
`gflat_to_rmu`. So:

- `to_rchunk_inner`: speculative — written for composability that no
  caller exercised at the API level (it was always going to be inlined
  into the kernel that wraps it).
- `to_rmu_inner`: load-bearing.

50/50 outcome on speculative API design. Not catastrophic, but the
"introduce the pure helper first, then build the wrapper" pattern is
exactly the kind of premature abstraction that the
`feedback_no_redundancy` memory line warns against.

### B5. Planner accommodations — `band_fft_pool` on lorrax_A

The orchestrator's prompt notes that `band_fft_pool` was added to
lorrax_A's planner in `ff5873c` to model a defect that the Path-D
structural fix has now eliminated. That's a clean example of the
"don't extend the planner to accommodate replicated buffers" memory
line being violated and then corrected. The detour itself was bounded
(one commit on a different branch); it's listed here because it's the
clearest example of *learning that paid off* — the principle was
rediscovered the hard way and is now in agent memory.

### B6. Reports — meta-discussion about meta-discussion

The reports directory has:

- `round5_discussion.md` (8.4 k words) — discussion among agents.
- `round5_unified_plan.md` (13.2 k words) — the plan that came out.
- `round6_discussion.md` (5.7 k words) — discussion of executing it.
- `round7_discussion.md` (4.0 k words) + `round7_dead_code_audit.md`
  + `round7_test_methodology_audit.md` (3.6 k words combined) — audits
  of the work.
- `round8_discussion.md` + `round8_unified_fft_pipeline.md` (9.1 k
  words combined) — design for next pass.

By the time the BLOCKER fix landed at 00:26, ~92 k words of agent
discussion had been produced *in the same session* about a 1340-line
code change. **That's roughly a 70:1 ratio of process artifact words
to net source lines.**

For an honest comparison: the head-correction-fix initiative (`reports/
head_fix_2026-04-04/`) — also a substantive numerical-correctness fix
— produced ~3-5 markdown files totalling well under 10 k words.

### B7. The BLOCKER fix `c796420` was a missed case

The Round-6 commit `f567aa0` claimed "G1 (round5_unified_plan §5 /
round6_discussion §G1): MoS2 3×3 charge end-to-end on 4× A100... bit-
equal at ULP". The Round-7 G1.1b / G1.1c sub-gates then found two cases
where it was wrong by factors of 5-11×.

This is **not bloat** — it's exactly what the G1 sub-gate suite is for,
and the fix is 27 net LOC. But it's evidence that the Round-6 validation
was insufficient before declaring the fix "done" and writing the 5.7-k-
word `round6_discussion.md`. A single-engineer cadence would more likely
have run the bispinor-transverse path before declaring victory.

## 4. What was load-bearing

Honest accounting of what genuinely had to happen:

1. **The structural fix itself** — moving the bc-loop boundary inside
   `z_q_from_psi_sm._local` via `lax.scan`-inside-`shard_map`. This is
   the durable correctness/memory win. ~500 LOC net.
2. **`gflat_to_rmu` + `to_rmu_inner`** (Defect 3 mirror). Independent,
   genuinely needed for the centroid-sample direction. ~250 LOC net.
   Could have been deferred but doing it the same week was sensible.
3. **`_slice_local_tile_bc` + `_bpd_max`** on `PsiGStore`. Required for
   the io_callback static-out-sds contract. The fact that it got
   churned is incidental — the final shape is needed.
4. **Bit-identity validation against `lorrax_A` baseline** at MoS2 3×3
   charge (commits 5cadd4b and f567aa0 both report
   max|diff|=6.13e-07, max rel ≤ 1e-10, signature of float-reorder).
   This is the right kind of validation gate for a refactor.
5. **G0 nested-primitive composition tests** (`test_io_callback_nested.py`,
   325 lines). io_callback × scan × shard_map × all_gather is genuinely
   subtle and the test file isolates the contract.
6. **The back-pad fix `c796420`**. 27 LOC, fixes a real BLOCKER. Cheap
   and load-bearing.
7. **Content-hash on `qvec_frac` (3d0636c)**. 10 LOC, correctness fix.
   Could have been on `main` without all the surrounding ceremony.
8. **The unification plan** (`round8_unified_fft_pipeline.md`) for
   future FFT-box callsites — this is honest forward design that
   captures the lesson "one FFT-box pipeline, not three". Whether it
   pays off depends on tomorrow's execution.

## 5. Counterfactual

What a single competent engineer with the same prior knowledge the
agents had at end of planning (i.e., they have the HLO dump showing 4×
unsharded FFT-box slots, they know the bc-loop is Python-unrolled in
jit, they know the
`feedback_path_d_scaffolding_pattern` memory entry that says
"scan-inside-shard_map, not naive fori_loop") would produce:

| Metric                    | Actual         | Counterfactual | Overhead |
|---------------------------|---------------:|---------------:|---------:|
| Source LOC net change     | ~1340          | ~700-900       | ~1.6×    |
| Test LOC net change       | ~700           | ~400-500       | ~1.6×    |
| Lines touched (churn)     | ~3620          | ~1100-1400     | ~2.7×    |
| Commits ahead of main     | 7              | 2-3            | ~2.5×    |
| Wall time                 | ~14 h orchest. | ~6-8 h focused | ~2×      |
| Process artifact wordcount| ~129 000       | ~5 000-8 000   | ~17×     |
| Validation rigor          | G0 + G1 (charge), missed bispinor; caught next session | Same — would also have missed bispinor | ~equal |

**Aggregate cost-overhead factor: ~3-5× on the things that matter (code
+ wall time), ~17× on documentation overhead.**

The counterfactual single engineer would have:

- Read the HLO dump → identified that `_kernel` allocates `psi_Y_full`
  as the dominant transient → realised the boundary has to be inside
  `z_q_from_psi_sm`, not in a standalone forward helper.
- Written `_slice_local_tile_bc` + `_bpd_max` (~60 LOC).
- Written the scan-inside-shard_map body of `z_q_from_psi_sm._local`
  (~250 LOC, working from the established pattern).
- Run MoS2 charge + MoS2 bispinor + a unit-sized
  asymmetric-L/R + short-final-bc case as their G1 (catching the
  back-pad bug **before** declaring victory).
- Written `gflat_to_rmu` for Defect 3 (~250 LOC). One commit.
- Logged result in a ~1500-word report.

What the agents added on top: ~400 LOC of churned scaffolding, an
intermediate cohsex.in knob that came and went, a standalone
`gflat_to_rchunk` helper whose only surviving artifact is a docstring
mention, and ~120 k words of process commentary.

## 6. Recommendation for tomorrow

**Drop the 4-agent multi-round pattern for the unification work.** Use
one of:

1. **Best — single fresh-context implementer + reviewer.** One agent
   reads `round8_unified_fft_pipeline.md` + `round5_unified_plan.md` +
   the most recent code, implements, commits, ships. A second short-
   context agent reviews the diff. Two more rounds of HLO validation
   and G1 testing. Estimated cost: 4-6 h, 2-3 commits, ~500-800 LOC,
   one 1-2 k-word report.
2. **Acceptable — orchestrator (you) implements directly off the design
   docs.** The design docs are already detailed enough that the
   implementation is mostly mechanical. Cuts the meta-discussion layer
   entirely.

Avoid: another 4-agent planning round followed by another 4-agent
implementation round. The signal-to-noise on the 92 k words of
in-session agent discussion is poor. The agents are good at writing
plans; they are not good at the second-and-third-derivative
self-critique that the round 6/7/8 discussion files chase.

Concretely, the failure mode visible in this session is:

- Round N agents produce a plan.
- Round N+1 implementer produces a partial/wrong implementation.
- Round N+2 auditor agents discover the partial nature, write a
  discussion file, and re-plan.
- Round N+3 implementer rewrites.

This is not faster than one careful implementer; it is slower with more
artifacts. The 5cadd4b → f567aa0 5-hour rewrite cycle is the canonical
example.

Keep the multi-agent pattern for: parallel-independent tasks (Defect 3
mirror + structural fix at the same time was a reasonable parallel
split). Drop it for: sequential or tightly-coupled work where each
agent's output is the next agent's input.

---

**One-line summary**: heavy — ~3-5× on code/wall-time and ~17× on
documentation versus a competent single-engineer baseline, with the
5cadd4b→f567aa0 5-hour rewrite-of-rewrite and the 129 kword 38-file
report directory as the clearest evidence.
