# Round 5 — Agent 4: numerics / correctness / validation lead + synthesis

Read `round5_discussion.md` first. **Your lens is numerics + correctness, AND you own the final unified plan. You're the synthesis lead.**

## Your specific deep-dives

### 1. Bit-identity contract

The new scan-inside-shard_map design vs the current `gflat_to_rchunk` + concat + slice + `z_q_from_psi_sm` chain. They compute the same mathematical quantity — `Z_q = FFT(sum_bc(γ̃·(IFFT(P_l_bc)·conj·IFFT(P_r_bc))))`. But the *order* of operations matters for FP rounding:

- Current: `P_l_full = einsum_global(psi_l_X, psi_l_Y_full)`, then partition into rank-5 P_pair via reshape — all-at-once einsum reduction.
- New: `P_l_acc = sum_bc(einsum_per_bc(psi_l_X_bc, psi_l_Y_bc))` — accumulated sum over bcs.

If summation order is the only difference, this is **at most a relative-rounding-level FP delta**, not a bit-equal contract. Specify the tolerance the validation gate should require: probably `rtol=1e-12, atol=1e-14` (one rounding ULP per einsum partial sum) — not bit-equal.

Cross-check: does the existing `accumulate_rchunk_to_gflat` test scaffold use rtol/atol of this magnitude? If so, mirror it.

### 2. Edge cases to enumerate

The unified plan must explicitly address all of:

- **Single-bc case** (`band_chunk_size >= nb_total`): scan with one iter. Trivial, but make sure the code path doesn't special-case it differently from N_BC > 1.
- **Asymmetric L/R band windows** (`nb_L != nb_R`): the L slice and R slice within a bc may have different lengths, or one bc may contribute only to L (or only to R). Define behavior.
- **Short final bc** (last bc has bpd_per_bc < band_chunk): pad to `bpd_max`, mask in body. Confirm the masking-via-`jnp.where` doesn't introduce FP issues (masked zeros add zero in einsum — math-neutral, but worth saying so explicitly).
- **`norms_l` / `norms_r` per-bc**: these arrays are sized to the L/R window widths globally. Inside the scan body, we slice per-bc — the norms need to be sliced consistently. Alternative: pre-divide `psi_l_X`, `psi_r_X` by the norms ONCE before entering the helper. Latter is cleaner. Recommend.
- **`solver_kind == 'auto'` resolution**: irrelevant to the helper but make sure the upstream pipeline is unaffected.
- **bispinor channels**: charge (μ_L=0) and transverse (μ_L=1,2,3) use different γ̃ tuples. The helper's API must thread these through unchanged.

### 3. Validation plan for Round 6 implementation

Three gates, in order:

- **G1 CPU bit-identity**: MoS2 3×3 synth WFN, single-channel charge. Run today's `_kernel` body vs the new body. Should match at rtol=1e-12.
- **G2 HLO slot count** on synth scale: predict ≤ 5 substantive slots (2 carry + 1 FFT box + 1 small + transient). Confirm no remat warnings during compile (filter for "Involuntary full rematerialization" in stderr — must be zero).
- **G3 End-to-end CrI3 6×6 80 Ry on the live allocation**: run gw.gw_jax with the new kernel. Predicted total preallocated-temp ≤ 25 GiB. Run must complete all 16 r-chunks (and the remainder) without OOM, without tracer leak, without remat. (qp_wfn write at the end is a separate downstream bug — ignore.)

Mark each gate's go/no-go criterion explicitly.

### 4. Synthesis ownership

You own `round5_unified_plan.md`. As Agents 1, 2, 3 publish their sections (via `round5_discussion.md` notes pointing to a section in the unified file), you assemble them into a coherent plan with the following structure:

1. **Goal + design** (one-paragraph statement)
2. **The shard_map + scan structure** (Agent 1's contribution)
3. **The io_callback inside scan** (Agent 2's contribution)
4. **Expected HLO + memory profile** (Agent 3's contribution)
5. **Numerics + edge cases + validation gates** (your own contribution)
6. **Implementation sketch** (code pseudocode, file-by-file)
7. **Risks + fallback** (if io_callback inside scan inside shard_map fails, what's Path A from Agent 2 §5?)
8. **Out of scope** (CCT path mirror, planner bookkeeping fixes, etc.)

Iterate with the others until everyone signs off. Specifically wait for "Agent N round 5 ready" markers from Agents 1, 2, 3 before declaring the plan final.

## Deliverable

`round5_unified_plan.md` — the single doc that Round 6 implementer will work from. Print "Agent 4 round 5 unified plan done" when the plan is complete AND all three peers have signed off (you'll know by their "ready" markers in their tmux panes + `round5_discussion.md` confirmations).

Read-only on `sources/`. No compute.
