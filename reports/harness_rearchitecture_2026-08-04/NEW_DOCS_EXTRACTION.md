# High-value advice extracted from the newly pushed lorrax docs

2026-08-05, against origin/main (317 new commits, ~50k new lines of
markdown). Sources: my direct read of docs/dev/QUALITY_PATTERNS.md,
docs/dev/device_put_hidden_allgather.md, docs/architecture/decisions.md,
plus two full sweeps (current docs/dev + architecture + device-invariance
reports; and the frontera_campaign/wk_REL archive). File:line pointers
throughout refer to the lorrax repo at origin/main.

## 0. Route these three documents before anything else

1. **docs/dev/QUALITY_PATTERNS.md** — ten failure classes distilled from
   ~14 root-caused production failures, each with a principle, ending in
   a 9-question assessment rubric for any new distributed-physics code.
   This IS the antipattern checklist the harness work wanted to build;
   it already exists. It should be the critic-pass checklist and the
   DESIGN.md companion.
2. **docs/architecture/decisions.md** — dated binding owner rulings,
   each with a "licenses deleting" clause, plus the provenance rule: "a
   code comment must not be able to mint an 'owner decision'; cite an
   entry here or do not use the phrase."
3. **docs/architecture/layers.md** — level decided by VOCABULARY, not
   directory; imports downhill only; budgets are ratchets (exceeding
   fails AND coming in under without editing the table fails); the
   driver purity test ("no jax.sharding, no shard_map, no Mesh(, no
   os.environ on inspection"); five named refusals-to-abstract.

## 1. JAX performance — the easy-to-miss items (gold stars)

1. **`device_put` onto a multi-process sharding is a hidden all-gather**
   (docs/dev/device_put_hidden_allgather.md). JAX asserts all processes
   passed the same value: two P×|x| buffers per rank (measured 13 GiB
   for a 0.34 GiB array at P=16; >900 s timeout). Load-bearing detail:
   only UNCOMMITTED operands take the branch — `jnp.zeros`,
   `jnp.asarray(numpy)`, raw numpy. Committed arrays (jit outputs with
   out_shardings) reshard cleanly. Worst hazard: the cost is charged to
   whatever stage happens to run, so it masquerades as a regression in
   unrelated code. Use `make_array_from_callback` /
   `common.collectives.device_put_process_local`.
2. **A free output sharding can emit a full-size all-reduce.** The same
   einsum with output sharding left free emitted an all-reduce of a
   full 2624 MB cube (identical at P=4 and P=16); pinned to the cube's
   own tiling, two all-gathers of cube/P. The difference is ONE
   `with_sharding_constraint` (docs/dev/large_nmu_operation.md:163).
3. **The partitioner will defeat your intent**: a traced-index slice was
   hoisted by XLA into a full-stack all-gather LARGER than the gather it
   replaced; GSPMD kept a `jnp.arange`-derived mask REPLICATED because
   constants carry no sharding — one 74 GiB buffer that looked like 32
   temporaries (wk_REL/ppm_fusion_notes.md:9-66). The fix is
   structural, not advisory: put the constraint inside `shard_map`
   where the partitioner cannot hoist. Corollary (QUALITY_PATTERNS §4):
   for communication and memory, THE OPTIMIZED HLO IS THE ONLY GROUND
   TRUTH — no gather/keep claim is verified until a trace shows it.
4. **Mixed-dtype dots silently promote f64→c128**: ~400 MB temp per
   channel and 2× the flops. Split the real operand into parts + one
   `lax.complex` (~2× win, measured on CPU and GPU). But do NOT
   over-apply: the same split on complex×complex was tried and REFUTED
   (Eigen's dgemm is per-flop SLOWER than its zgemm) — the same
   transformation wins on one body and regresses another
   (wk_REL/lgemm_notes.md:146).
5. **Collectives are bandwidth-shaped with ~zero per-call latency**
   (measured intercept ≈ 0 on raw MPI, the JAX shim, and gloo). So
   message-count halving buys nothing; what pays is WHICH MESH AXIS
   carries the big payload — node-local (consecutive-rank) vs strided
   replica groups. The BSE inner loop shipped the inversion for months;
   staged_reshard now REFUSES a mesh with the axes inverted rather than
   silently shipping (docs/dev/staged_reshard_primitive.md:137). And
   XLA:CPU does not overlap collectives with compute AT ALL (measured
   overlap fraction ≈ 0): pipelining levers must live below the
   framework or be dropped (wk_REL/RESHARD_OVERHEAD_MEMO.md:233).
6. **Bound the payload of any single collective as a program property**
   (~128 MB default): drive the shard_map from a host-level loop over
   blocks so one XLA execution per block — the compiler cannot
   re-combine what it never sees together. Calibrated by a P=144 death
   inside a 1.15 GB collective with MaxRSS at 12% of budget: "a memory
   cap is not a transport cap" (docs/dev/linalg_ffi.md:305).
7. **Backend asymmetry — the same perf decision is right for opposite
   reasons.** CPU: XLA's Eigen GEMM is ~3× below MKL FLAT across
   threads/shapes/dtypes (kernel quality, hence FFI required); XLA FFT
   at flat-k layout pays exactly 3.00× the tile in transposes. GPU:
   cuBLAS hits 103-105% of nominal FP64 peak — FFI buys ≈ 0 beyond
   recovering layout, and the GEMM FFI correctly refuses CUDA meshes.
   NEVER port a performance ruling across backends without re-measuring
   (wk_REL/FFI_EVIDENCE_AUDIT.md:380-484).
8. **Compile-cache hygiene**: the persistent-cache key is
   process-invariant only on GPU; a naive shared cache at P>1 is a
   permanent silent hang (XLA:GPU compilation is a COLLECTIVE — a rank
   that never compiles never publishes its autotuning share); rank-0
   hit + peer miss blocks peers 20 minutes by default. And the
   anti-hype number: a perfect compile cache removes ~1% of the compile
   storm — the cost is tracing/lowering/dispatch count, which the cache
   is consulted after (LORRAX_FRONTERA_ADVICE.md:94-131).
9. **`jax.process_index()` initializes the XLA backend.** Any code that
   must run before `jax.distributed.initialize` (kernel-cache keys,
   rank-gated prints) must read launcher env vars first. One banner
   call at import once silently pinned every CLI to one process
   (docs/dev/ffi_gate_contract.md:162; QUALITY_PATTERNS §8).
10. **Donation is inert inside fused jits** — it acts only at top-level
    dispatch; and shape-changing ops can never alias ("do not
    cargo-cult input_output_aliases"). Shape-preserving transforms
    aliasing {0:0} is the terminal form of donation
    (staged_reshard_primitive.md:424; flat_k_fft_service.md:64).
11. **Pad extent is a numerical input.** LU at a padded extent changed
    eqp deterministically with device count (a resonance at pad=672
    took a trace from −0.15 to −117.9 eV); pad rows were exactly zero
    the whole time — the leak was pad-SHAPE, invisible to pad-row
    inspection. Rules that fell out: solves run at the LOGICAL extent
    (one wrapper, grep-able); pad axes independently, never to the
    px·py product (up to 3.16× waste); keep the fixed-P pad-flip gate
    (LORRAX_EXTRA_MU_PAD: "any result that moves under this at fixed P
    is a defect") — it isolates deterministic defects from the
    irreducible 1-2 ULP fusion-regrouping floor
    (reports/device_invariance_2026-07-08/ROOT_CAUSE.md).
12. **Eager/host traps**: `np.asarray` on a non-addressable sharded
    array RAISES loudly, but on a replicated global array it gathers
    silently; host-numpy helpers called per iteration cost
    D2H+index+upload+pipeline-sync each (9.2 GB/iter at production
    size); and the one-ULP cliff — numpy's sequential cumsum vs XLA's
    reassociated cumsum flipped a `searchsorted` branch and put E_F at
    the VBM instead of midgap (ibz_self_consistency_scaffold.md:436-507).
13. **Memory accounting blind spots**: `memory_analysis()` misses
    exactly ONE library workspace (FFT — cuFFT scratch comes from a
    runtime allocator outside buffer assignment); ScaLAPACK workspace
    is malloc'd inside the handler, invisible to the planner. And any
    memory-tier claim must NAME THE BINDER IT RELIEVES — removing an
    announced 3.69 GB/rank gather moved peak VmHWM by 0.02 GiB because
    a different transient was binding (large_nmu_operation.md:96).
14. **The thread-main story, resolved**: the MPI guard fires only on
    communicator CREATION, and the collective cache is keyed per
    CLIQUE — warming x+y+world from the main thread is the fix, and it
    works because the warm-up jit is small enough that XLA runs it
    inline on the caller. It is an accident of the implementation,
    recorded as such, with an upstream ask filed
    (wk_REL/jax_threadmain_alternatives.md).

## 2. Architecture rules (the load-bearing ones)

1. **Two plans per solve family** (decisions.md standing + large_nmu):
   a LOCAL plan (default, bit-identical across meshes/P) and a
   DISTRIBUTED plan (explicit opt-in — block-cyclic is a different
   numerical gauge, agreement is κ·ε not bit-exact). `auto` never
   crosses that line silently. Schedules of a plan are not new plans.
2. **Environment grants capability; it must never select policy.**
   Physics/routing choices change only via declared inputs. Every
   resolved choice/demotion/route is ANNOUNCED once, from the rank that
   made it (the test: "can this decision differ per rank?"). Silence is
   legal only where declared — `silent_platform_demote` is a string,
   not a bool: to demote silently you must write down why. Off-dials
   may refuse; typos never do (they resolve to the announced default —
   a grammar error must not kill a run or silently pick a known-bad
   policy).
3. **Checks fire at the earliest phase where the fact is knowable AND
   observable** — resolve-time guard ladder before any collective
   (violation inside a collective = deadlock, not error); trace-time
   for dtype/extent facts (a single-phase API "would have to lie about
   when it checked"); but NOT build-time probes whose wrong answer is
   invisible (check_symbol_exists false-negatived twice, silently
   compiling the 1.9×-slower arm — replaced by runtime dlsym + an
   unconditional first-use receipt).
4. **Named refusals to abstract are as load-bearing as abstractions**
   (layers.md:240): no generic shard_map wrapper — in_specs/out_specs
   ARE the distributed algorithm; the right abstraction is SPECIFIC
   (contract_bands owns one named pattern with axis-order,
   de-promotion, and divisibility policies — which is why it can carry
   a contract document, and why the tree's two generic wrappers are
   among the sites missing check_rep: "genericity did not make them
   safer"). No shared C++ handler base (three scratch models with the
   same silhouette, each measured). Unify the call site, never the
   bodies.
5. **Kill opt-out-by-omission**: `getattr(meta,'n_rmu',default)`
   silently restored a fixed bug for any caller omitting the kwarg —
   make arguments REQUIRED, prefer structural neutrality at birth
   (zero the pads when created, not in every consumer), then one shared
   helper carrying the contract, and never an optional fallback. The
   fix ladder: structural > shared helper > required arg > (never)
   optional kwarg.
6. **Every gate has a red twin, and a cell whose twin passes is VOID,
   not green** — the first run of the FFI contract gate discovered its
   own bit-parity cells were perturbing below the c128 ULP and testing
   nothing. Thresholds at the ULP test nothing; tolerances are derived
   from the arithmetic with stated headroom over the failure they must
   catch.
7. **Declare the parity class of every change before gating it**:
   bit-exact / value-level / gauge-class (κ·ε). Never claim stronger
   than the mechanism supports ("the A/B did come out h5-bit-identical
   — record that as an observation, not as the contract").
8. **Release rule** (tests/KNOWN_FAILURES.md): a release ships LISTED
   known-fails, never unknown ones — every non-passing test accounted
   for.

## 3. Methodology (what the campaign learned about measuring)

- "Across this campaign the measurement infrastructure produced more
  false results than the system under test. Every HIGH finding fails
  SILENTLY" (SIZE_CAMPAIGN_BRIEF.md:1090). A diagnostic reporting a
  suspiciously clean value everywhere is more likely broken than the
  system it measures (the awk that read VmRSS=0.00 for four healthy
  jobs — and got them cancelled).
- `strings` can confirm presence, never absence; absence claims need
  nm/objdump/AST — the corpus's biggest wrong conclusion was a
  four-link chain, airtight from step two onward, wrong at step one.
- Trust the files, not the notifications: completion notifications
  arrived with future timestamps and claimed results 40 min before the
  job could have produced them; every recorded number is read from
  disk after sacct shows terminal.
- Provenance is proved by the run itself: manifest verified at job
  START and END, resolved absolute path in the log,
  PYTHONDONTWRITEBYTECODE=1 (a `.pyc` copied with preserved mtimes
  executes even when every `.py` hash matches).
- Evidence vocabulary: MEASURED / INFERRED / ASSERTED / REFUTED, where
  MEASURED means the auditor reproduced the number from a named
  on-disk file. Nothing is filled in by plausibility.
- Put the invariant on the COLLECTIVE'S OUTPUT, and prefer the free
  one: a Hermitian congruence's result is Hermitian for ANY input, so
  the check tests only the machinery — always-on, zero extra
  collectives. Every invariant used to sit on construction tiles, which
  is why a 5%-rate silent transport corruption left no trace in 1913
  job logs.
- A/B hygiene: mirrored ordering (A B A B B A) so first-process-in-job
  effects can't bias arms; the first timing process in a job measures
  an artifact (measured 113 ms vs 569-581 ms steady state); solo
  controls can invert verdicts (a solo A/B would have scored the NUMA
  fix as a 2× regression); sequential configs in one job are
  confounded by page cache.

## 4. Immediate consequences for the harness plan

1. QUALITY_PATTERNS.md's rubric becomes the critic checklist; do not
   write a new one.
2. rules_gate's device_put rule gets refined per the survey: ban
   uncommitted-operand device_put onto multi-process shardings; the
   blessed wrapper (device_put_process_local) already exists in-tree,
   as do ~20 benign committed-reshard sites (the allowlist).
3. The stale RULES_seed/RULES_v2 items derived from origin@07-22 are
   superseded wherever these docs state the current form — re-sweep
   done in effect by this extraction; reconcile RULES_v2 against
   decisions.md (TRS scoping ruling now official; square-mesh ruling
   is truncate-per-decisions vs refuse-per-implementation — one of the
   two records needs updating).
4. env_vars.md + the startup-report test already implement the
   "report every dial" rule the harness plan proposed — point at them,
   don't rebuild.
