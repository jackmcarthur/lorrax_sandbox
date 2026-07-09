# Bugs & issues found in the gate-0 / exact-agreement thread (2026-07-02)

Distinct from the 74 suspected-bugs in `DEAD_CODE.md` (those came from the static audit).
These were found while (a) fixing the red regression gate and (b) building the MoS2 exact-
agreement COHSEX gate. Status: F=fixed, W=workaround, O=open, D=doc.

## Code / driver
- **[F] `write_qp_wfn_h5` crashes every IBZ/path run.** The debug one-shot QP-WFN dump
  (`debug.write_wfn_h5` default true) passes full-BZ/path `U_full` to an IBZ-only writer →
  ValueError before the eqp compare. Fix: skip-with-warning on k-count mismatch
  (`gw_jax.py`, branch `agent/gate-0-qpwfn` commit 86349a0).
- **[O] ~3.5 meV W-path drift on the gate reference.** Current main's SX/COH drifted a
  k-uniform −3.3 meV plateau vs frozen `eqp_ref.dat` (traced to `fc1602a` IBZ-q W solve).
  Benign (intended numerics) but the reference was never re-frozen, so the gate is silently
  red. Needs re-freeze. (§7 GATE_AUDIT.)
- **[D] `sigma_diag` had no band energy (`Eo`).** Could not align LORRAX's band window to a
  BGW `sigma_hp` by energy. Fix: added optional `Eo=` column (branch `agent/gate-0-qpwfn`
  commit e849760).

## Config / parser
- **[F 2026-07-03, commit 61ae4b8] Config parser did not strip inline `#` comments on value lines.**
  `cusolvermp_charge = off   # note` is read as value `"off  # note"` ≠ `"off"` → silently
  falls back to `auto`. Cost me a wasted run. Put comments on their own line. (Flag-surface
  refactor target.)
- **[D] `sigma_freq_debug.dat` head columns are a double-count trap.** `x_head`/`sex_head`/
  `coh_head` are diagnostic SUBSETS already folded into `x_bare`/`sex_0`/`coh_0`
  (`cohsex_sigma.py:232`), but `COHSEX_INPUT.md:347` doesn't say so — inviting the reader to
  add them (I did, producing a spurious 2382 meV). Document that they are not additive.

## Infrastructure (also in KNOWN_SANDBOX_ERRORS.md)
- **[F] `lorrax_D` modulefile FFI defaults pointed at purged `$SCRATCH`.** Caused shifter
  mount failures and the *appearance* of a cuSOLVERMp deadlock. Fixed: repointed the three
  `LORRAX_FFI_*_DIR` defaults to `$HOME/software/lorrax_*` (`modulefiles/lorrax_D/0.1.0.lua`).
  **A/B/C modulefiles still carry the dead defaults.**
- **[W→F] cuSOLVERMp `cusolvermp_cholesky` "deadlock" on 2×2 mesh.** Originally hung 18 min at
  0% GPU. NOT reproducible with the correct FFI env — cuSOLVERMp 0.7.2 potrf completes and is
  correct (4-GPU ≡ 1-GPU to 1.3e-5). Original hang was the flaky NCCL-TCP-socket second
  communicator. Diagnostic: rank-0 banner must read `library 0.7.2 … comm path: NCCL`
  (`0.6.0 … CAL` = wrong lib on LD_LIBRARY_PATH). `cusolvermp_charge=off` no longer needed.
- **[O] Native `sharded_cholesky` (2D-blocked) needs `jax.lax.pcast` (JAX 0.9).** Absent from
  the Shifter container's JAX 0.5.3 → the native multi-GPU Cholesky path aborts. So on 2×2 the
  working path is cuSOLVERMp (now fixed), not native. Same container-vs-pyproject JAX gap that
  fails `test_reshard_all_to_all` + `test_aot_memory` (GATE_AUDIT §2).
- **[O] `auto` routes a 640×640 C_q to distributed potrf.** Overkill (native chol is µs); a
  size threshold in `_resolve_solver_kind_charge` (`isdf_fitting.py:950`) would be a perf fix.

## Physics / comparison
- **[O] MoS2 2D COHSEX vs BGW = 368 meV, in the screened SX/CH partition.** Bare-X (head-incl)
  agrees ~11 meV; SX (−489) and CH (+360) large & opposite-sign → static-CH convention
  signature (CH's +360 ≈ known `exact_static_ch` artifact) + under-tuning (640 centroids,
  `epshead` head ≈ 15 meV off). NOT a head or body bug (§10 GATE_AUDIT). Open.
- **[D] Stale doc refs found:** `cohsex.in` header + commit 38bf75f cite `compare.SKILL §4i/§4j`
  which don't exist; the `use_bgw_vcoul` "MC-averaged v(q+G) for body" comment is misleading
  (the q=0/G=0 head is force-zeroed on vcoul load, `read_bgw_vcoul.py:185-192`).

## Methodology note (my own error, logged so it isn't repeated)
Adding `x_head` to `x_bare` double-counts the head. Always confirm whether a debug column is a
subset-of-total or an addend before differencing against BGW.

## GN-PPM 1-GPU (1×1 mesh) crash — FIXED (2026-07-02)
- **[F] GN-PPM crashed on 1 GPU; healthy on 4.** Root cause (pinned via a temp shape print):
  `build_G_tau`'s Σ-branch `mask_A` derives from `wfns.occ`, which carries a leading nspin=1
  axis. A 2×2 mesh squeezes it, a 1×1 mesh does NOT → on 1 GPU `mask=(1,nk,nb)` while
  `enk=(nk,nb)`; `jnp.where` broadcasts `phases` to 3-D and `build_G`'s `'ksxn,kn,knty'`
  einsum rejects the 3-D `'kn'` operand. **This is what all the earlier "WFNsmall crashes"
  actually were — every one ran `LORRAX_NGPU=1`.** Fix: reshape mask to `enk.shape` before the
  where (`greens_function_kernel.py`, branch `agent/gnppm-1gpu-mask-fix` commit 6dbb3b4; no-op
  on the 4-GPU path). Verified: MoS2 3×3 GN-PPM EXIT 0 on 1 GPU, unit suite 250 passed, χ₀
  (shares build_G_tau) unaffected. GN-PPM regression gate now wired + green (commit e7646e1).

## Found during budget validation (2026-07-02) — pre-existing, not from the cleanup
- **[O] The in-code `GPU high-water mark:` self-check is unreliable.** Under the
  default cudaMallocAsync allocator it's a single post-loop `nvidia-smi` sample
  that misses freed transients (read 0.75 GB / 3% vs the real 22.26 GB), and
  `_track_peak` hardcodes `nvidia-smi --id=0` (read a foreign GPU-0's 32.31 GB for
  3 different budgets). The faithful peak comes from JAX `memory_stats`
  `peak_bytes_in_use` under BFC (`LORRAX_MEM_DEBUG=1`). This matters: it's how a
  planner mis-prediction would be *caught in production*, and right now it can't be.
  Fix: base the runtime HWM check on `peak_bytes_in_use`, not the `--id=0` nvsmi sample.
- **[O] The planner's budget governs only the ζ-fit/V_q stages, not the whole run.**
  A separate ~18 GB upstream (pre-`Computing C_q`) transient is unmodeled/unchunked,
  so the whole-run peak floors at ~18 GB — lowering `memory_per_device_gb` below ~18
  shrinks the ζ-fit stage as designed but does not make the whole run fit that budget.
  Irrelevant at the 28 GB default on a 40 GB card (no OOM), but the "budget" knob is
  narrower than its name implies. See reports/memplanner_cleanup_2026-07-02/BUDGET_VALIDATION.md.
