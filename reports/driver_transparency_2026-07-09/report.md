# Driver transparency revision — Phase B + Phase C (executed)

_2026-07-09. Branch `agent/driver-transparency` on `sources/lorrax_D` (base: `9925d43` on
`agent/memplanner-cleanup`). Spec: `reports/gw_refactor_map_2026-07-01/IDEAL_SCAFFOLD_VS_LORRAX.md`
§5, recommendation "B then C, each gated". This report is the as-executed record._

## Summary

`gw_jax.main()` went **991 → ~590 lines** and now reads as the twelve-line physics scaffold:
config → ISDF (ζ, V) → minimax quad → W(ω) by role → Σ_xc dispatch → update_H\[qp_solver\] →
degen-average → eigh → writers. The one-shot path now consumes the **same**
`screening_requests_for` → `compute_screening` → `compute_sigma_xc` pipeline the SC loop uses —
the ~400-line inlined second copy is deleted. The new acceptance gate
`tests/test_sc_oneshot_equivalence.py` pins **SC-iteration-1 ≡ one-shot G0W0** at 1e-6 on
sigma_diag/eqp0/eqp1; all 12 pre-existing e2e gates stayed **bit-identical** throughout.

Two real bugs found by the new gate (both pre-existing, invisible before unification):

1. **SC output-seam basis mismatch.** main() rotated the last `SigmaResult` back to the DFT
   basis using the *converged* U (eigenvectors of the final carry). The SigmaResult actually
   lives in the basis of the *input* U of its iteration; the two agree only at the fixed
   point. At `max_iter=1` this mis-rotated Σ_x/V_H by tens of eV in eqp0/sigma_diag.
   Fix: `SCState.last_sigma_basis_U` — the basis U travels with the SigmaResult it defines.
2. **Iteration-0 eigh ULP noise, amplified ~15 orders of magnitude by GN-PPM.** The SC map's
   `eigh(diag(E_DFT))` roundtrips eigenvalues at ~1 ulp (4.4e-16 Ry measured). The GN-PPM
   pipeline amplifies this to **max|ΔΣ_c(ω)| = 0.44 eV**; a controlled experiment (+1 ulp on
   every WFN energy, `ulp_amplification_evidence/`) moves Σ_c by **1.28 eV** while the run
   itself is bit-deterministic (rerun maxdiff = 0.0). Fix: iteration 0 of the SC map uses the
   exact eigensystem of its exactly-diagonal carry. **The amplification itself is a real
   conditioning pathology of the GN-PPM fit (near-threshold pole modes; same family as the
   device-invariance "Fix-3" on-pole sensitivity) and remains open** — see "Findings".

## Phase B (commit `3102994`) — pure moves, main() 991 → 717 L

Bit-identical under the full suite (257 passed / 9 skipped). Moves:

| Out of main() | New home | Lines |
|---|---|---|
| IBZ slice → per-q Dyson solve → unfold choreography | `screening.compute_static_w()` | ~95 |
| W0_qmunu + q→0 head-scalar restart flush | `gw_output.persist_w0_and_head()` | ~48 |
| Σ-decomposition freq-debug table | `gw_output.write_freq_debug()` | ~115 |
| One-shot WFN_qp.h5 dump (+ IBZ k-count guard) | `gw_output.write_qp_wfn_oneshot()` | ~22 |
| H-build-seam degenerate-set averaging | `degen_average.average_sigma_components()` | ~20 |
| 4× duplicated `get_enk_bandrange` fetch | one hoisted `enk_dft` | — |

## Phase C — the unification (this commit)

- **`compute_screening`**: the `"static"` role now routes through `compute_static_w` (the IBZ
  wedge solve + unfold), so SC iterations get the same static-W path as one-shot. Probe W
  stays full-BZ-direct (the GN-PPM fit has a documented ~0.1 meV q-set path-dependence; the
  frozen goldens pin the current behavior). `sym`/`centroid_indices` threaded (SCInputs too).
- **`compute_sigma_xc`**: gained `Gij`/`wfns_transverse`/`bispinor_v_q_path` pass-through
  (bispinor Σ^B now dispatchable), a **streamed-Σ_c branch** (static-COHSEX eigh stand-in —
  the behavior main() had), `efermi_dft_ev` on `SigmaResult`, and X_ONLY returns
  `sigma_sx ← sig_x` so the sigma_diag sigSX column stays meaningful (and now includes Σ^B,
  which the old inline path dropped for x_only).
- **`qsgw_utils.solve_qp()`** — `update_H[Σ; qp_solver]`: one_shot_dft = pass-through of the
  dispatch's at-E_DFT QSGW build; fixed_point = diagonal on-shell solve + scissor + QSGW
  rebuild (moved verbatim — the frozen fixed_point rotations gate pins it).
- **main()**: statics + mode-pivot + PPM + 100-line QP branch → `screening_requests_for` →
  `compute_screening` → `compute_sigma_xc` → `solve_qp`. SC runs skip the one-shot dispatch
  entirely (they used to compute a full static Σ and throw it away).
- **Config**: explicit screened `compute_mode` × `do_screened=false` now errors; the driver
  derives screening from the mode enum alone. (Behavior fix: explicit `compute_mode=x_only`
  used to silently run full COHSEX because the static branch pivoted on `do_screened`.)
- **SC fixes**: `SCState.last_sigma_basis_U` (bug 1); iteration-0 exact eigensystem (bug 2).

**Gate:** `tests/test_sc_oneshot_equivalence.py` — runs the GN-PPM MoS2 3×3 fixture as
one-shot and as `self_consistent` + `sc_max_iter=1`; compares sigma_diag rows and
eqp0/eqp1 numeric tokens at atol 1e-6. (`qp_wfn_rotations.h5` is deliberately NOT compared:
the SC carry applies the band partition — off-diagonal masking + scissor for
out-of-ω-window bands, 28/80 in-range on this fixture — so its eigh family legitimately
differs from the one-shot full-matrix eigh.)

## Findings for the user (open items)

1. **GN-PPM ULP amplification (measured).** +1 ulp on every WFN energy →
   max|ΔΣ_c(ω)| = 1.28 eV on the MoS2 3×3 fixture (evidence in
   `ulp_amplification_evidence/`; invalid-mode census, window/node counts, and fit
   `unfulfilled%` all UNCHANGED — the sensitivity is in the fit values of near-threshold
   modes, not in a discrete flip). This is the same disease as the Fix-3 on-pole census
   sensitivity and is why cross-platform/cross-P reproducibility of GN-PPM Σ can never be
   tighter than ~eV-scale × (input ULP noise). Worth a dedicated conditioning study
   (e.g. Ω²-regularization near threshold) — physics decision needed.
2. The three deliberate don'ts from the spec were honored: G(τ) stays virtual; the q→0 head
   stays a threaded channel (not a stage); W is evaluated at exactly {0, iω_p}.
3. `fit_ppm` was NOT lifted to main() — `ppm_pipeline` already reads as the scaffold
   (fit → Σ_c → head → at-DFT interp → write) and lifting would add a seam with no dedup.

## Sub-driver audit executed (commit `cece78c`)

A 4-file audit (fresh-grep verified at HEAD, per the "documentation lies about deadness"
rule) then the ranked moves:

| Action | What | Where |
|---|---|---|
| DELETE | `cohsex_sigma.get_cohsex_kernels` | dead (docstring claimed SC-loop use — false) |
| DELETE | `gw_init.get_effective_chunk_size` + `meta.chunk_size` + `chunk_size` input key | write-only chain; key now deprecation-warned + ignored |
| DELETE | `gw_init.get_bandranges` + `gw/__init__` re-export | byte-dup of live `psp/get_DFT_mtxels.py` copy |
| DELETE | `w_isdf.flatten_V_qmunu` (legacy 8-D shim) | restart files rank-3 flat-q since padding consolidation |
| DEDUP | `_ensure_compilation_cache` triple alias | one name: `ensure_jax_compile_cache` |
| MOVE | `build_static/imag/real_quadrature` + `resolve_minimax_energy_reference` | `w_isdf` → `minimax_screening` (B1 code with the engine; `w_isdf` is now pure χ₀/W, 759 → 534 L) |
| FAN-OUT | `gw_driver_helpers.py` **deleted** | `profile_section` → `common/jax_profile`; `_resolve_input_path` → `file_io.paths.resolve_input_path`; `build_bgw_v_grid_fn` → `gw/compute_vcoul`; `setup_runtime` → `gw_jax._setup_runtime` (runtime→gw_config cycle avoided) |
| DEFERRED | `isdf_fitting.mem_probe`/`_nvsmi_used_mb_local_gpu` → `common/gpu_utils` | near-dup nvidia-smi samplers need a careful merge, not a blind move |

## Suite status / commits

Branch `agent/driver-transparency`, three commits, each run against the full suite:

| Commit | What | Suite |
|---|---|---|
| `3102994` | Phase B pure moves | 257 passed / 9 skipped, gates bit-identical |
| `160d22d` | Phase C unification + solve_qp + 2 SC bug fixes + new gate | 258 / 9 / 0 |
| `cece78c` | audit deletes + minimax move + gw_driver_helpers fan-out | 258 / 9 / 0 |

`gw_jax.py`: 991 → **637** lines (main() is the scaffold + the SC branch).
`w_isdf.py`: 759 → **534** (pure χ₀/W). `gw_driver_helpers.py`: deleted.
Net branch diff: +1459 / −1091 (the + includes the 126-line new gate and moved-code
re-homes with restored docstrings).

## Next steps

1. GN-PPM ULP-amplification conditioning study (physics decision — see Findings §1).
2. Deferred `mem_probe` → `gpu_utils` consolidation.
3. NEXT_TARGETS #7 (zeta_loader/zeta_reader merge) and #12/#13 remain the top hygiene items.

## Addendum (evening session): I/O audit executed + suite parallelization

- Pushed to **origin/main** (`d03c857`).  SC machinery → `run_sc_driver` (main() = 502 L,
  one post-Σ seam for SC and one-shot).
- **I/O parallelism audit** (full matrix in the CHANGELOG entry): reads parallel in all 3
  backends; async writes real but FFI-only (mpi_host/allgather deliberately sync —
  documented Cray MPI_THREAD_SINGLE / rank-0 semantics).  Executed: zeta merge (#7, −381 L,
  backend-plumbing bug fixed), AsyncDispatcher single-sourced, explicit `slab_io` key
  (all 3 backends input-reachable), −190 L dead I/O API, one padded-μ read shape in
  v_q_g_flat (duck-typing deleted).
- **Suite runtime audit** (user question): profile = 13 e2e gates ≈ 85% of serial wall
  (top: bispinor_pad4 203 s cold / bispinor 102 s / kij-stream parity 55 s / SC≡one-shot
  39 s); units ≈ 1-2 min.  Fix: pytest-xdist 4-way with per-worker GPU pinning →
  404 s cold vs 470-580 s warm-serial baseline; warm parallel ≈ 200-250 s.  Caveats
  encoded in skills/checkpoint/SKILL.md (lxrun couples tasks to GPUs — use 1 srun task +
  gres=gpu:4; conftest must override SLURM's CUDA_VISIBLE_DEVICES).
