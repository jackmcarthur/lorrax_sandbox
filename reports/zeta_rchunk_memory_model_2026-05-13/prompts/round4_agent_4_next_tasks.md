# Round 4 — Agent 4: Next tasks (incl. profiling)

Read `round4_discussion.md` first (status snapshot + file-polling protocol). Also wait briefly for Agents 1, 2, 3 to publish their reports — your job synthesizes their findings into a prioritized action list.

## Your task

Synthesize a prioritized list of next-action tasks. The user is explicitly asking about **profiling** as part of this — what measurements should land next? Combine:

1. **Defects identified by Agents 1, 2, 3** (read their reports as they publish):
   - The new remat warning at the gflat_to_rchunk → consumer boundary (Agent 1).
   - Any branch reconciliation gaps between lorrax_A and lorrax_B (Agent 2).
   - Any model refinements the planner needs (Agent 3).

2. **Defects from your morning catalog** (`defect_catalog.md`) that haven't been touched yet:
   - Defect 4 — `solve_zeta` q-batch Python loop (~88 GB SPMD-replication trap, prior attempts on record).
   - Defect 5 — `_v_q_per_q_g_chunked_jit` G-loop (~300 MB, cheap `lax.fori_loop` swap).
   - Defect 6 — Davidson CGS2 (you said leave alone; confirm).
   - Anything else from your 11 ruled-out items that deserves revisiting given new info.

3. **Profiling tasks** — what GPU-side measurements should land before the next round of structural changes? Examples:
   - cuFFT throughput vs `chunk_size` for `gflat_to_rchunk` (the user explicitly raised this — could we use `fft_chunk_size = 1` with `lax.scan` cheaply?).
   - End-to-end wall time for CrI3 6×6 80 Ry: morning baseline OOMed at fit_one_rchunk; new run completes (estimate from r-chunk ETAs). What's the per-r-chunk cost broken into (a) bc-load FFT, (b) pair density, (c) CCT solve, (d) gflat accumulate?
   - HBM HWM measurement via `nvidia-smi --query-gpu=memory.used` during the run — does it match the planner's 51 GB / actual 48 GB?
   - NCCL collectives volume — count + size of all-gathers/all-reduces in the new HLO vs the baseline.

4. **The io_chunk / fft_chunk split** that the user raised in the most recent orchestrator conversation:
   - Forward helper has implicit single `chunk_size` for both io_callback batch and FFT batch.
   - User's observation: ψ(G_sph) is ~5–10% of FFT box, so io batch should be ~10× larger than FFT batch.
   - Nested scan structure: outer scan for I/O, inner scan for FFT.
   - This should be a queued task — propose where it lands in the priority list, and what the implementation cost is.

## Deliverable

Write to `reports/zeta_rchunk_memory_model_2026-05-13/round4_next_tasks.md`. Structure:

1. **Top of the queue** — the 1–3 things that should land before any other work. Rationale per item.
2. **Next-priority block** — 4–8 items, with effort estimates and rough payoff.
3. **Profiling block** — specific measurements to take, the command to run, what we'd learn.
4. **Out of scope for now** — items deliberately deferred, with reason.
5. **The CrI3 8 Ry validation gate** — what's the killer test that would say "Path D is genuinely done"? (Probably end-to-end CrI3 wall time + final HBM HWM under the budget, plus all remat warnings gone.)

Synthesize Agents 1/2/3's findings — cite by `agent_N's_file.md §N`. Don't re-do their work; build on it.

Communicate via `round4_discussion.md`. Print "Agent 4 round 4 done" when finished.

Read-only on `sources/`. ~15–25 min. Wait for Agents 1, 2, 3 to publish at least their headline findings before finalizing your priorities.
