# Round 5 — Agent 3: XLA / HLO / BufferAssignment prediction lens

Read `round5_discussion.md` first. **Your lens is HLO/XLA behavior, but you're an expert in ALL levels of JAX parallelization — challenge the others.**

## Your specific deep-dives

### 1. Predicted HLO for the scan-inside-shard_map kernel

Walk through what the HLO will look like at CrI3 6×6 80 Ry. Predict:

- **Preallocated-temp pool size** — your Round 4 reading said current is 44.56 GiB. What's the new prediction? Break into terms.
- **FFT-box slot count** — current run shows 1 (aliased across r-chunks). What about INSIDE the scan body? Will XLA's scan-internal allocator alias the per-bc FFT box across scan iters → 1 slot? Or will it pipeline-keep 2× live (your morning concern from `agent_2_structural_fix.md` §C)?
- **Pair-density slot count** — `P_l_acc`, `P_r_acc` are carry; need to live ALL scan iters. So 2 slots × 14.85 GiB = 30 GiB just for the carry. **Is that right?** Compare against the current 3 P_pair slots (~45 GiB). Net should be saving ~15 GiB on the carry side ALONE.
- **psi_Y_full materialization** — should be GONE entirely. 30 GiB savings from this alone (matches Agent 1 R4's "psi_Y_full materialized TWICE" finding).
- **Total predicted** — sum of above. Should be ~20-25 GiB if the design is clean. Reality check: the 12 GiB "single unsharded FFT box" that's in the current HLO (cost of remat) — does it go away too? Probably yes, since no remat.

### 2. BufferAssignment behavior on scan carries

The two rank-5 carries `(P_l_acc, P_r_acc)` are c128 each ~14.85 GiB. XLA's BufferAssignment needs to either:

- Pre-allocate 2 slots that live across all iters (standard scan carry).
- Or alias them into a SINGLE 14.85 GiB slot if their lifetimes within an iter don't overlap (probably not — both contribute to the einsum results that live through the iter).

Predict: 2 slots, both alive across iters. Total carry cost = 30 GiB. This becomes the new dominant Peak C term.

### 3. FFT box aliasing inside scan

The per-iter body fetches `psi_G_bc` (small, G-sphere shape), runs `to_rmu_inner`/`to_rchunk_inner` to FFT it (creates an FFT box `c128[k_chunk, bpd_max, ns, n_rtot]` per-rank). The FFT box's lifetime is *within one scan iter* — once the einsum consumes it, it's dead.

- Will XLA alias the FFT box across iters → 1 slot?
- Or pipeline-keep it 2x → 2 slots?
- The `accumulate_rchunk_to_gflat` reference is the canonical scan-with-internal-FFT-box pattern — it shows 1 slot. So extrapolate the same answer here.

But there's a subtlety: `accumulate_rchunk_to_gflat`'s scan body has no `io_callback`. Does inserting `io_callback` (Agent 2's territory) disrupt the FFT-box aliasing? Probably not — the io_callback returns a fresh array per iter; the FFT box derived from it should still alias across iters.

### 4. The "extra remat" question

Current HLO has ~20 `Involuntary full rematerialization` warnings at the helper/consumer boundary. In the new design, the consumer (γ̃ contract, FFT, etc.) consumes the carry `(P_l_acc, P_r_acc)` directly after the scan exits. **No reshard needed** — the carry already has the right per-rank-local shape for the post-pair pipeline.

Walk through: confirm no remat warnings should appear in the new design. If you spot any potential remat trigger, name it.

### 5. cuFFT batching

The per-iter FFT box has shape `c128[k_chunk, bpd_max, ns, n_rtot]`. cuFFT batches over the leading axes. At CrI3 with `k_chunk=36` (replicated, since psig_k_chunk_size is retired), `bpd_max=10` (rank-local of band_chunk=16), `ns=2`, this is a batch of 720 FFTs over `nx·ny·nz = 1.125 M` points.

- Is that an efficient cuFFT batch size?
- Compared to the current design's batch (full nb_local × ns FFTs per r-chunk, no bc loop on device), is this MORE or LESS efficient per-FFT?
- The user's question from earlier — whether `fft_chunk_size = 1` (single FFT per scan iter) would work — is implicitly resolved here: at one bc per iter, we have a moderate batch already, not 1.

### 6. Compile time at scale

CrI3 80 Ry today compiles in ~3 min. The new kernel has more structure (shard_map → scan → io_callback). Predict whether compile time grows substantially (>5×) or stays roughly the same. If you suspect a compile-time hazard, flag it.

## Deliverable

Same as Agents 1 and 2: contribute your section to `round5_unified_plan.md` (Agent 4 owns), Agent 3 → others note in `round5_discussion.md`. Iterate.

Print "Agent 3 round 5 ready" when finished. Read-only on `sources/`. No compute.
