# Changelog


## 2026-07-16: BSE exchange comms-reduction (P2/C1) + nt-aware dispatch (P-NT) [agent/bse-comms-opt, lorrax_A worktree, source, NOT merged]

Two approved matvec-audit items (JOINT_FINDINGS §5-6). P2: `apply_V_ring`'s
6-collective/apply band-ppermute q=0 exchange replaced by ONE shared
`bse_ring_comm.bse_exchange_gspmd` (the stack matvec's GSPMD form; ring+stack now
share it, `apply_V_ring` + 4 wrappers deleted, −62 net lines). Resolvent SOLVE
collectives 40→12 per matvec (2×2 HLO; ppermute rings eliminated); closure
2.4077e-9 unchanged (1×1==2×2), `--compare-wq` per-q closure byte-identical to base.
P-NT: `solve_bse_sharded` dispatches bs≤2→ring, bs≥3/Davidson→stack (crossover
nt≈2-3). Gates green; full 1-GPU suite 221 passed/12 skipped. Honest: the count cut
is wall-neutral at the single-column resolvent (GMRES-reorthogonalization-dominated;
full-mesh collectives cost ~= the tiny ppermutes at nt1) — the win is topology-aware
scale-out + single-source-of-truth; it speeds the batched (b≥8) matvec. See
`reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md` §"P2 + P-NT".

## 2026-07-16: W(omega) block-Lanczos-chain model — full-frequency screened Coulomb [agent/bse-phase2, lorrax_A, source, NOT pushed]

The feature the W-resolvent arc was building toward: full-frequency `W_q(omega)`
from ONE structure-preserving block-Lanczos chain per q, replacing the per-omega
shifted-GMRES solves. New module `src/bse/w_omega_chain.py` (builder + evaluator,
plain arrays/functions, no new class), CLI `bse_w_exact --w-omega-chain`, gate
`tests/test_bse_w_omega_chain.py`. The shifted-solve path (`--compare-w0/-wq`) is
KEPT as the validation ORACLE (gate reference), not a parallel production path.

- **Structure-preservation:** the para-Hermitian screening operator
  `H_RPA=[[A,B],[-B,-A]]` (A=D+V, B=V) collapses to an N-dim SYMMETRIC resolvent
  (Casida Omega^2 / Shao et al.): `W(z)-v = 2 Phi (z^2 I - S)^{-1} Phi^dag`,
  `S=D^{1/2}(D+2V)D^{1/2}` Hermitian, `Phi^dag e_nu = D^{1/2} f_nu` (the SEED
  scaled). `A-B=D` exact for screening => reduction is EXACT (proto rel 5e-16).
  Only `z^2` enters, so ONE chain serves every omega. `S` is applied through the
  production matvec VERBATIM (`(A+B)U = matvec([U;U])[X]`) + a `D^{1/2}` diag —
  no new kernel, no duplicated encode/decode, matvec signature unchanged.
- **Chain:** symmetric block Lanczos (block width = probe width, small), FULL
  (DGKS) reorthogonalization, robust Gram/eigen block-QR (deflates degenerate/
  pad columns). Chain object = host alpha/beta/R0 (block-tridiagonal T + seed
  norms) + device chain blocks (pair basis, `sh.X_full` spec). Evaluator per
  omega: tiny `(z^2 I - T)^{-1}` host solve + device combine + existing PROJECT
  (`snapshot`) reduce-scatter to the `(mu_X,nu_Y)` tile. No matvec/GMRES per omega.
- **Convergence (MoS2 gnppm, 1 GPU):** monotone; omega=0 reproduces the disk
  `(W0-V)` closure through the evaluator (m=64 -> 6.5e-6, m=120 -> 2e-7 toward the
  ~2.4e-9 GW floor); imaginary axis (the GW-relevant axis) saturates the oracle's
  GMRES residual (10i -> 1e-11 by m=64); finite q=(0,1,0) ~10x higher floor,
  cleanly convergent. `--chain-len` is the accuracy knob (default 32).
- **Amortization:** chain built ONCE (m matvecs); per-omega eval 40-100x cheaper
  than a fresh oracle solve (q=0 m=48: build 2.2s, oracle 961 ms/omega, chain
  9.3 ms/omega). **omega-count break-even ~2-12** — far below any GW/BSE frequency
  grid.

Validation: gate 2 passed (46.8 s); full plain 1-GPU suite re-run (see PHASE2_LOG
"W(omega) Lanczos-chain model"). Only-owned edits (`bse_w_exact.py`,
`w_omega_chain.py`); `apply_V_ring` internals untouched.


## 2026-07-16: BSE matvec efficiency follow-up — P3 pair-amp hoist + P5 donation drop + c64 flag [agent/bse-phase2, lorrax_A, source, NOT pushed]

Three approved items from the matvec efficiency audit (`reports/bse_refactor_map_2026-07-15/archive/matvec_efficiency_audit`).

- **P3 (hoist M_X/M_Y):** the V-term exchange pair amplitudes `M = Σ_s conj(ψ_c)ψ_v`
  were rebuilt inside EVERY solver iteration's matvec (a per-iteration black-box jit,
  ψ un-hoistable by XLA). Now computed ONCE at load (`bse_io` → `data["M_X"]/["M_Y"]`,
  single source, μ-on-x / ν-on-y) and threaded as matvec args across ALL sharded
  matvecs (stack, simple, ring, ring-full) via a uniform 11-arg signature; `apply_V_ring`
  slices the y-block of the hoisted M_X. Peak-neutral; between-matvec floor +~2M/p.
  **Warm min timing (inflated nc48/nv48/nk16/μ800, 1 GPU): −1.4 ms/matvec fixed =
  +9.6% at nt1 (single-vector/GMRES) / +2.9% at nt4 (block-Lanczos)**; before/after
  bit-identical (relerr ~1e-16). Finite-q defect caught+fixed: `build_finite_q_data`
  now recomputes M from the rolled ψ_c (stale q=0 M gave closure rel_err 2.66).
- **P5 (drop cosmetic donations):** removed the always-declined `donate_argnums` on
  W_q (`bse_lanczos`, `absorption_haydock`) and T (`apply_W_from_T`, both ring builders,
  4 sites) — silences the "donated buffers not usable" warnings, changes nothing (§3).
- **c64 flag (comment only):** noted the deferred complex64 W-term ~2× bandwidth lever
  at the dtype seam (`bse_stack_matvec._w_stack`), OFF per owner decision, ptr §4.

Validation (module-free srun+shifter, 1 GPU): BSE gates (dense-reference, stack-matvec,
W0+finite-q resolvent) green; **full plain 1-GPU suite 221 passed / 12 skipped / 0 failed**
(golden GW gates included). Owner-excluded P2/P-NT untouched.


## 2026-07-16: Finite-q W_q resolvent — generate + validate at every symmetry-reduced q [agent/bse-phase2, lorrax_A, source, NOT pushed]

Generalized the W(0) resolvent engine (`apply_screening_resolvent_block`) to
FINITE q — the on-grid `|v k, c k+q⟩` RPA density response — WITHOUT forking:
`bse_w_exact.build_finite_q_data` rolls conduction `ψ_c`/`ε_c` by `+q` on the
C-order (nkx,nky,nkz) k-axis (`jnp.roll`) and swaps in `V_qmunu[q_flat]`; the
matvec/seed/project/solver/sharding are byte-identical to q=0. New
`common/symmetry_maps.kgrid_shift_map` (the ONE k+q fold + umklapp-G helper),
`bse_io` `load_v_full` (full V tensor), `--compare-wq` loops the IBZ q-grid
(`SymMaps.q_irr_kgrid_int`) one at a time vs each q's OWN `(W0−V)[q_flat]` tile.

**Convention (dense-sweep validated): roll `+q`, NO umklapp Bloch phase** — GW's
χ0(q) is a periodic FFT-convolution over k (raw wrapped ψ), so the design-doc
`exp(−2πi G_umk·s_μ)` phase BREAKS the match (0.6–3.2 vs 1e-8); it belongs to a
direct-read finite-Q BSE, not this producer's tiles. Finite-q V tiles keep G=0
(no head).

Fixed **three coupled defects in the shared `_get_gmres_solver`** that the stiff
finite-q tiles (large G=0 head, cond(H)~1e8) exposed: normal-equations LSQ
`solve(HᴴH)` (→ `lstsq`), single-pass Arnoldi orthogonality loss (→ DGKS
reorthogonalization), and an operator-blind solver cache that reused q=0's
operator across the q-loop (→ key on `id(matvec)`/`id(data)`). q=0 / FEAST
unchanged.

Per-q closure (MoS2 gnppm, 5 IBZ q): **max rel_err 5.3e-8**, gmres_resid ~2e-10
(quadrature-floor-limited; q=0 stays 2.3e-9). Gates: `test_bse_w0_resolvent`
3 passed (+ `kgrid_shift_map` unit + finite-q), BSE 16/16, FEAST smoke green.
`reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md` §"Finite-q W_q resolvent check".


## 2026-07-16: W-column resolvent perf — jit the seed/project shard_map boundaries [agent/bse-phase2, lorrax_A, source, NOT pushed]

Profiled `apply_screening_resolvent_block` (the W(ω) Lanczos engine). Hot spot
was NOT the GMRES/matvec (2.4% at 1×1) but the two reshard boundaries `gen`
(SEED) and `snapshot` (PROJECT), which were *bare* `shard_map`s — re-traced +
re-lowered to HLO on EVERY eager call (~2.5/2.8 s) while the matvec was already
`jax.jit`-cached. Fix: `jax.jit(in_shardings,out_shardings)` on both builders in
`bse_ring_comm.py` (single-source; also speeds the `bse_pseudopoles` FEAST-seed
path). Isolated: gen **3050×**, snapshot **1069×**. End-to-end warm: 1×1 **4.62→0.75 s
(6.2×)**, 2×2 **18.44→12.07 s (1.53×)** (2×2 remainder is the shared ring
matvec's collective latency, out of scope). Value-faithful: gate 14 passed/1
deselected, closure rel_err = recorded to 4 sig figs; device invariance
preserved at its true pre-existing level (bare 1×1-vs-2×2 5.64e-12 → jitted
6.31e-12, inherent psum order); jit fp reassociation ≈8e-12 rel, 400× below the
2.4e-9 closure. No numerics knob changed. `reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md`
§"W-column resolvent profiling".

## 2026-07-16: W-column resolvent — device-resident W(mu_X, nu_Y) tile + 2x2 seed-bug fix [agent/bse-phase2, lorrax_A, source, NOT pushed]

Sharding-quality upgrade of the `bse_w_exact` W-column path (the future
Lanczos-chain `W(omega)` engine). One single-sourced function
`apply_screening_resolvent_block` (SEED zeta->pair `gen` | SOLVE scan-of-GMRES,
per-column-independent | PROJECT pair->zeta reduce-scatter) emits the screened
Coulomb as a device tile `W(mu_X, nu_Y) = P('x','y')` — mirroring the Sigma_PPM
reduce-scatter (`ppm_tau_kernel`), no replicated `(mu,nu)`, output no longer
host-stacked. `build_density_snapshot_operator` gained `scatter_nu_on_y`
(default off = old `(b,mu_X)` for `bse_pseudopoles`; on = W-tile). Gate now
asserts the PartitionSpec.

Found + fixed a **pre-existing multi-device bug** exposed by the first 2x2 run:
`build_realspace_random_transition_generator` did the centroid contraction as a
local x-slice with the conduction index pre-sliced, dropping off-rank mu on
px>1 (~50% wrong; only ever run on 1 GPU before). Fixed to full-c +
`psum_scatter('x', scatter=c)` like `apply_V_ring` (no-op at px=1).

Validation (gnppm fixture): 1x1 and 2x2 now **bit-identical** (max rel_err
3.203e-9, gmres_resid 4.2e-10), device-count invariant, matches the pre-refactor
1-GPU baseline on overlapping columns. Gate + 14/14 BSE gates pass.
`reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md` §"W-column sharding".

## 2026-07-16: Si sym-centroid BSE-degeneracy experiment — centroids are NOT the degeneracy lever [A, runs only]

Orbit-closed ISDF centroids do NOT restore exact BSE degeneracy (old/sym split
ratio 1.004-1.018, still ~500-2000 ueV; 792 orbit-closed vs 960 literal
centroids near-identical). **ROOT CAUSE FOUND (same-day diag pass, SUPERSEDES
the initial psi-unfold hypothesis): band-window truncation of degenerate
multiplets at high-symmetry k.** Si Gamma multiplets are 6-fold (nspinor=2);
4v4c keeps 4 of 6 -> transition density non-covariant at the cut points; the
518 ueV split is a near-cancellation of +-3000-4300 ueV star contributions.
Degenerate-CLOSED Gamma window restores multiplets to <=36 ueV (~100x).
Energies covariant to 0 ueV, psi-at-centroids to 1e-15 — unfold/zeta-fit
EXONERATED. Property of any fixed-(nv,nc) BSE window, not LORRAX; the right
degeneracy gate uses degenerate-closed windows. Secondary real defect: V0/W0
tiles ~3% non-covariant under centroid permutation (head injection -> ~8%),
contracts to 1e-4 in kernels; filed for the tile/head path.
runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/{report.md,diag/FINDINGS.md}.

## 2026-07-16: W(0) resolvent cross-check — GW screened Coulomb validated to ~2e-9 [agent/bse-phase2, lorrax_A, source, NOT pushed]

Owner-requested cross-validation: `W(0) - v = v(0 - H_RPA)^{-1} v` (Casida
resolvent, ω=0) reproduces the restart's head-less `W0_qmunu - V_qmunu` q=0 tile
to the GW minimax-integration floor. Run on the gnppm gate restart (MoS2 3×3,
nval=26/ncond=20, full χ0 window), 1 GPU. **8 probe columns: max rel_err =
2.41e-9, median 2.24e-9; GMRES resid ~3e-10 (an order below → quadrature-limited,
not solver-limited). It closes at the minimax-noise level.** Report:
`reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md` §"W(0) resolvent cross-check";
run dir `runs/MoS2/A_bse_w0_resolvent_2026-07-16/`.

Three fixes on `bse_w_exact`'s path (all validated bit-for-bit against a dense
RPA reference):
- **`bse_feast.ensure_W_R`** — single-sourced the `W_q→W_R` conversion (the stale
  `data["W_R"]` KeyError in the shifted-matvec chain); the 3 inline copies in
  FEAST/spectral-bound are gone, `bse_w_exact` reuses it.
- **`build_bse_ring_matvec_full(screening=True)`** — the RPA test-charge screening
  H uses the RING kernel `V=K^A` in BOTH symplectic blocks (density-density
  bubble), NOT the excitonic `V_B` (which overshoots W by 1.79×). One matvec,
  physics-flagged; the existing `screening=False` optical-BSE path is unchanged.
- **`bse_io.load_bse_data_from_restart_sharded(inject_head=False)`** — head-less
  body load for body-vs-body diagnostics.
- **`bse_w_exact` rewrite** — correct symplectic combination `rhs=[f;-f]`, readout
  `X+Y` (old `[f;f]`/`X+Y` gave identically 0 at ω=0); shared per-column resolver
  used by both `--out` and the new `--compare-w0`; band-window parity forced to the
  full χ0 window. Gate: `tests/test_bse_w0_resolvent.py`.

## 2026-07-16: BSE Phase 2 COMPLETE on agent/bse-phase2 (lorrax_A) — B1 dense exchange + trial-stack matvec [A, source, NOT pushed]

Four self-contained commits off main 6bd4dc9; report
`reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md`. Full plain 1-GPU suite:
**218 passed / 12 skipped / 0 failed** (orchestrator run, incl. all golden
gates + the new BSE gates with xfails flipped). Multi-device differential
(1x1 vs 2x2 mesh, 4 GPU): the stack matvec is **bit-level device-count
invariant** (bs=1 max|Δ|=1.5e-16; at bs=4 the true Ritz values are
bit-identical across meshes and only the known block-Lanczos GHOST values
differ ~7e-5 — the pre-existing solver_program P1 defect, reproduction
numbers in PHASE2_LOG). Physics review confirmed the dense exchange
per-element; memory review confirmed no n_trials axis on any intermediate
(peak temp flat: 183 MB at 1/4/8 trials vs ring's linear 1.5 GB at 8).
Merge/push decision pending Jack.

- **`6d52999` dense-reference gate** — `tests/test_bse_dense_reference.py` +
  `bse_dense_state` fixture (piggybacks `gnppm_session`; MoS2 3×3, 2v2c, N=36).
  Explicit `⟨cvk|H|c'v'k'⟩` from the same head-injected arrays the matvecs use.
- **`d7b51a1` B1 fix (dense k-summed exchange)** — the Q=0 exchange is DENSE in
  (k,k′); every matvec kept k as a batch axis (only the (k,k) diagonal, scaled
  1/Nk). Fixed single-sourced in `bse_serial`, `bse_simple` (+`sh.S_k0`),
  `apply_V_ring` (transitively the non-TDA B-block). serial/simple/ring now
  bit-exact to the dense H (relerr ~1e-15). Physical shift on the fixture:
  exchange-sensitive states move up to ~25 meV, lowest exciton pair splits a few
  meV; insensitive states unchanged. Preconditioner diagonal + density helpers
  untouched (not H-exchange paths).
- **`11bab32` trial-stack matvec** — `src/bse/bse_stack_matvec.py`
  `build_bse_stack_matvec(..., kernel='bse'|'rpa')`: W-term = ONE shard_map,
  body = `lax.scan` over trials ⇒ one `T(μ,ν,t,s,k)` alive regardless of
  n_trials. `memory_analysis()`: stack temp FLAT 183.4 MB at n_trials∈{1,4,8}
  (≈2× the 91.7 MB one-T bound); ring temp LINEAR 184/734/1468 MB. Bit-exact vs
  dense + simple (bse & rpa). `fft_helpers`: factored `local_{i,}fftn3` (one
  source; scan body calls kernel directly, shard_map can't nest).
- **`5d3819f` consumer wiring** — `solve_bse_sharded` (block-Lanczos + Davidson)
  and FEAST TDA (GMRES + `_rayleigh_ritz` batched subspace) repointed to the
  stack matvec (dtype-adaptive drop-in). Ring/gather/simple retirement NOTED not
  executed. Smoke: both run end-to-end; solve_bse_sharded bs=1 lowest-4 ==
  dense reference.
- **Findings/deferred**: (a) iterative Lanczos (single + block) return ghost /
  below-λ_min Ritz values on the head-injected operator (V/W tiles ~1e5
  near-cancel D) — a solver-conditioning issue, orthogonal to B1; spectrum gate
  materialises the matvec instead. (b) BGW `bsemat.h5` §4 off-diagonal gate
  needs absent Si data. (c) non-TDA B2 unfixed. (d) W(ω)/ladder seam designed,
  not built. Scratch/diagnostics under `tmp_phase2/`.

## 2026-07-16: BSE cleanup MERGED+PUSHED to origin/main; lorrax_A repurposed as the BSE seat [A, infra + push]

- `agent/bse-cleanup` fast-forwarded onto **origin/main c7a30ff → 6bd4dc9**
  (the 13 cleanup commits, Jack-approved). Cleanup worktree + branch ref
  dissolved; lorrax_D untouched (other session on agent/suite-speedup).
- **lorrax_A repurposed** (was dormant on agent/cri3-ppm-maxbands, 4 wks; no
  tracked WIP, branch ref preserved): now on main 6bd4dc9. Its modulefile was
  DANGEROUSLY stale (FFI stage-dirs → purged $SCRATCH paths = hard shifter
  mount failure; old allocator block) — replaced with lorrax_D's, only the two
  root-path lines differ (backup in reports/.../cleanup_verify/). Current
  liblorrax_ffi.so staged over the May build.
- Gate validation on lorrax_A: first run FAILED cohsex — a gitignored May-era
  `tests/regression/cohsex_debug/tmp` restart survived the reset and the
  session fixture reused it (fresh worktrees don't have this; beware when
  repurposing dormant checkouts). After cleaning stale gitignored state:
  **11/11 gates green** (16:17 warm). Also restored tmp/env_D.sh (tracked file
  caught by the tmp/ cleanup).
- Phase 2 (dense-reference gate → B1 dense-exchange fix → trial-stack matvec
  per Jack's spec: scan-over-trials inside shard_map, ONE donated
  T(mu,nu,t,s,k) per step — no n_trials axis on intermediates — RPA/BSE
  toggle, W-tile seam for future W(omega)/ladders) is IN FLIGHT on
  agent/bse-phase2 in lorrax_A; designs at
  reports/bse_refactor_map_2026-07-15/archive/designs/{dense_reference_gate,trial_stack_matvec}.md.

## 2026-07-15: BSE exchange adjudication (DENSE) + cleanup phase 1 (13 commits) [D, source — branch agent/bse-cleanup in worktree sources/worktrees/lorrax_D_bse_cleanup, NOT pushed]

Follow-up to the BSE refactor map, after Jack challenged the B1 finding and set
priorities (fix-broken-first; V/W recover-or-generate; GW-infra proximity).

- **Exchange kernel ADJUDICATED: dense in (k,k'), unanimous** — 3-agent
  adversarial round; the k-diagonal steelman failed (defensible only Nk=1);
  BGW's own bsemat.h5 (Si run) has off-diag k-block exchange the same order as
  diagonal (0.27-0.29 vs 0.39 max). Jack's momentum-shift-0 point is correct
  for the interaction line and orthogonal to delta_kk' (that structure belongs
  to the direct term). Finite-Q: Q-block-diagonal, dense within a block,
  v(Q+G) keeps G=0. Fix = k-summed encode (cheaper than the bug); Q=0 becomes
  the Q->0 case of one unified path. reports/.../archive/adjudication/VERDICT.md.
- **Cleanup phase 1 executed** on agent/bse-cleanup (base origin/main c7a30ff,
  worktree — lorrax_D's own tree busy with another session): 13 commits, net
  +312/-2192. Repairs: B3/B4/B5 loader flat-q normalization single-sourced,
  B2 non-TDA einsum, B6 padding, B7 eqp-window x3->1, KPM phantom kwarg,
  pseudopoles lost wiring restored, W0 bare-V loud warning, doc drift. Deletes:
  verified Tier-F (bse_jax matvec trio, ring pair, symmetrize_W_q, BSEData,
  preconditioner scaffolding, write_eigenvectors.py, stale test_bse loader);
  feast sweep harnesses archived to reports/.../archive/experiments/.
  Independent reviewer: exchange k-structure untouched (B1 fix is Phase 2),
  findings minor. CLEANUP_LOG.md has the ledger + deferred consolidations.
- **Recover-or-generate scout**: GW producers already library-shaped
  (prepare_isdf_and_wavefunctions / compute_screening / persist_w0_and_head are
  pure kwarg functions) — wiring plan at archive/designs/recover_or_generate_vw.md.
- Revised program: reports/bse_refactor_map_2026-07-15/PLAN.md (supersedes MAP §7).
- KSE'd: lorrax_agent module eval glob-expands `*` under non-interactive
  shells (lxrun/lxalloc unusable from scripts; workaround documented).
- Runtime verification COMPLETE: import smoke clean; plain 1-GPU suite
  173 passed; the 10 gnppm/bispinor gate errors reproduced IDENTICALLY at
  origin/main (fresh worktrees lack the .gitignored liblorrax_ffi.so — KSE'd);
  with the .so staged, gate subset = **11/11 passed incl. all four golden
  gates**. Checkpoint valid. reports/bse_refactor_map_2026-07-15/cleanup_verify/.

## 2026-07-15: BSE refactor map — 45-agent audit + 5 feature designs [D, reports only, no source changes]

`reports/bse_refactor_map_2026-07-15/` — the BSE analogue of the GW refactor map:
18 deep-readers over src/bse (~7.7 kL) + src/solvers (~3.9 kL), all 207 reader
claims adversarially verified (88 confirmed-bug / 20 dead / 23 refactor-target),
two independent teleological sorts, and 5 designs for the missing physics
(fine-grid interpolation, SR/LR Coulomb split, head+wings W interpolation,
solver/FEAST program, finite-Q H^BSE). BSE tree byte-identical e18d0e5..HEAD.

- **Showstopper 1**: every BSE load path is broken against restarts written by
  current gw_jax (B3 8-D-only V_q0 reader, B4 missing kgrid attr, B5 head
  injection before layout shim) — package only runs on legacy-era h5 files.
- **Showstopper 2**: the q=0 exchange kernel is k-block-diagonal in ALL four
  matvecs; physical K^x is dense in (k,k'). Triple-confirmed (Henneke deriv.,
  2-k dense-reference numerical diff, BGW kernel_main.f90). Si 3 meV gate was
  exchange-insensitive; ring-vs-serial check is circular.
- Non-TDA-with-W never executed (malformed B-block einsum); KPM broken at HEAD;
  FEAST default route runs the RPA kernel with fixed n_ritz=4 (the
  clustered-eigenvalue failure); bse_pseudopoles = lost wiring, not dead.
- Zero pytest-collected BSE coverage. Attack order in MAP.md §7: gates →
  loader repair → B1/B2/B6/B7 physics fixes → delete pass → single-source →
  solver program → feature designs. 10 design decisions queued for Jack
  (report.md "Decisions needed").

## 2026-07-15: suite-speedup phase 1 — cache audit + bispinor fixture regen at 25 Ry [D, source — branch agent/suite-speedup (stacked on agent/ppm-fit-conditioning), NOT pushed]

User goal: plain suite <60 s for the dev loop. Report:
`reports/suite_speedup_2026-07-15/`. Commit `d1847d9`.

- **Cache audit**: the persistent XLA compile cache is ACTIVE and already
  saving ~90 s on the gnppm gate alone (27 s warm vs 117 s disabled). Not a
  lever. Corollary: post-source-change suite walls are cold-cache outliers
  (this morning's 660 s was Fix-3 recompiles; warm = ~390 s → 373.6 s now).
- **Bispinor fixture regen** (`runs/MoS2/D_25Ry_bispinor_fixture_2026-07-15/`):
  ecutwfc 60→25 Ry, grid 30×30×120 → 20×20×75, WFN.h5 52.6→14.8 MB, transverse
  orbit set 209→208. Golden re-frozen: 3 fresh runs bit-identical, PAD=4 twin
  bit-identical, orbit/cascade properties preserved. **Chunking coverage
  (user requirement): 3 r-chunks in all 4 ζ-fit channels at the standard
  30 GB budget, and the gate now log-asserts ≥2 chunks per channel.**
- **Floor diagnosis corrected**: 3.6× fewer grid points bought only 9% of
  in-suite bispinor wall (87.2→79.3 s) — the per-gate floor is subprocess
  launch + per-process retrace (grid-independent), refuting the 07-09
  "r-chunk streaming floor" claim at fixture scale. The <60 s path is the
  one-process variant runner (next) + a dev/checkpoint invocation split.
- **One-process Tier-2 variant bundle** (`a506a71`): all 7 gnppm variants
  (restart baseline, pad12, kij_stream, sc_iter1, fixed_point, IBZ legs A+B)
  run in ONE subprocess via `tests/run_variant_bundle.py` — import + retrace
  amortized, env knobs applied/restored in-process (read-at-call-time
  verified), per-variant failure isolation, module-cache safety audited.
  Tier-2 chain ~114 → ~71 s.
- Suite: **207 passed / 0 failed — 390 s (start of day) → 283.0 s** warm.
  <60 s full-suite is not reachable with 4 fresh e2e pipelines + the FFI
  matrix serial; dev-loop split options in the report (user decision).
- Stale-tool note: `psp.get_DFT_mtxels` main() debug driver crashes on a
  ψ/ρ-grid broadcast before its kin_ion writer (NEXT_TARGETS #12 fodder);
  `gw.kin_ion_io` is the canonical kin_ion.h5 generator.

## 2026-07-15: Fix-3 PPM census determinism — magnitude-based mode classification [D, source — branch agent/ppm-fit-conditioning, NOT pushed]

Closes the last open refactor-map code item (HANDOFF Remaining #2). The GN-PPM
validity cut `|Wc0−Wc_probe| > 1e-14` thresholded a *cancellation* — every
dispersion-free element's denominator sits at the FP noise floor by
construction, so device-count reduction order flipped modes valid↔invalid and
one garbage-huge Ω moved the window max-Ω → minimax node counts → Σ_c (the
4g↔16g residual, ROOT_CAUSE.md). Replaced with relative magnitude gates in
`fit_gn_ppm_from_wc_pair` (commit `218aeb8`):

- **dead**: `|Wc0| ≤ 1e-12·per-q max|Wc0|` — roundoff elements whose
  Ω² = noise/noise must never enter the valid census;
- **stiff**: `|Wc0−Wc_probe| ≤ 1e-8·|Wc0|` — no resolvable dispersion; within
  the one-pole model denom→0 ⇔ Ω→∞, where the pole formula reduces EXACTLY to
  the static-COHSEX treatment (`static_limit`, the default) — so the invalid
  routing is analytic, not a fallback. No crossfade/ramp needed: the branches
  agree to O(1e-8) at the boundary.

New `tests/test_ppm_fit_classification.py`: four-class contract, exact pole
recovery, pad-birth, ±1-ulp classification stability (incl. an element parked
on the old absolute cut). Suite 207 passed / 0 failed; golden gates unchanged
(zero fixture modes reclassify — evidence is the unit pin, not a re-freeze).
Report: `reports/ppm_fit_conditioning_2026-07-15/`. Still open, physics-side:
on-pole Σ(E_dft) is inherently ill-conditioned (1.28 eV/ulp measured) —
document in the manual, don't engineer around it.

Next initiative queued (user request): suite wall now 11 min (FFI contract
tests added ~31 tests since the 220 s redesign benchmark); target <60 s via
compile-cache audit, one-process restart variants, bispinor fixture regen at
lower ecut (gate fixture is production-sized: 60 Ry, 30×30×120 grid).

## 2026-07-13: manual chapters 8-13 + appendices A-G drafted [B, source — agent/manual, PUSHED 84b8c2a]

All remaining chapters drafted in the locked register: Ch 8 bispinor (formalism/
four-density ISDF/assembly/status; factor_c_q indefinite-transverse-LU confirmed
at HEAD), Ch 9 (DFT ops, BSE matvec + solver table, htransform), Ch 10 (workflow/
outputs/restart), Ch 11 input reference regenerated from gw_config.py by extractor
agent (sys_dim 0/2/3 corrected vs agent claim; deprecated-alias + derived-default
tables), Ch 12 (planner/linalg limits/IO/troubleshooting), Ch 13 (module map,
distribution model, host-memory, testing tiers), appendices A-G (sym unfolds +
reduced-vs-not ledger; q->0 with BOTH GN normalizations; minimax tables; formats;
BGW cookbook; bibliography working set). Verify-TODOs: App B wings, App D dataset
shapes, App G citations, 8.4 Breit magnitudes, 3.1 tutorial numbers, 7.6 periodic
gate. Manual now covers the full outline; next = user revision rounds + figure/
number generation (needs allocations).

## 2026-07-11: GPU backend timing matrix 00-02 executed — cusolvermp+slate green, in-tree BLOCKED by JAX-0.9 pcast [C, runs only]

`runs/MoS2/C_bispinor_backend_timing_2026-07-11/` (MoS2 3×3 bispinor GN-PPM,
2×2 mesh / 4 GPUs, JID 55791797).  All variants need `LORRAX_FORCE_FULL_BZ=1`:
the manifest's `--no-orbit` 208 transverse set loud-fails the transverse ζ_T
IBZ orbit-closure check (`isdf_fitting.py:345`) — no auto-fallback for
transverse channels, env var is the documented bypass.

- **00 in-tree: FAILED (env incompat)** — `lax.pcast` (`common/cholesky_2d.py:186`,
  commit `c7e6695` "fix(jax-0.9)") does not exist in the GPU container's JAX
  0.7.2; every `path=sharded_cholesky` run dies in charge-channel `factor_c_q`,
  both BZ modes.  Logged in `KNOWN_SANDBOX_ERRORS.md` (2026-07-11).
- **01 cusolvermp: complete** — WALL 80 s (recorded 58.6 s); `path=cusolvermp_cholesky`
  charge chol 3.70 s, `path=cusolvermp_lu` transverse solves ~3.0 s/channel.
- **02 slate: complete** — WALL 74 s (recorded 54.4 s); `path=slate_cholesky`
  charge chol 4.48 s, in-tree per-q `path=lu` transverse solves ~1.5 s/channel
  (2× faster than cusolvermp_lu at this tiny size).
- **Correctness**: 01 vs 02 `sigma_diag.dat` max|delta| = **0.000e+00**.
  00-vs-01/02 diffs impossible (00 has no sigma_diag.dat).
- `shared/diff_sigma.py` fixed: `np.loadtxt` can't read sigma_diag.dat's
  labeled format; now regex-extracts decimal floats from non-comment lines.

## 2026-07-11: ScaLAPACK host backend for distributed_lu — Cray LibSci, zero new deps [C, source — branch agent/ffi-host-platform, PUSHED to origin/main 4085672]

**Later same day — e2e backend matrix + two bugs found and fixed** (report
section "e2e backend timing + equivalence matrix"; run set
`runs/MoS2/C_bispinor_backend_timing_2026-07-11/`): MoS2 bispinor GN-PPM,
2×2 mesh, 7 variants.  GPU {sharded, cusolvermp, slate} and CPU {sharded,
slate} × {per-q lu, scalapack} ALL bit-identical (`max|Δ|=0.000e+00`) after:
(1) `1421db1` — `lax.pcast` (jax-0.9-only, main commit c7e6695) broke
multi-rank sharded_cholesky under container jax 0.7.2; version-guarded.
(2) `4085672` — XLA CPU runs independent ffi_calls CONCURRENTLY → per-q
host-slate potrf MPI collectives raced on the shared comm (intermittent
one-q-tile corruption, 2 of 3 runs); shared host-handler mutex; 6/6
post-fix e2e determinism trials bit-clean.  Wall: GPU 73–80 s, CPU 271–274 s
(solver axes are noise at fixture scale — z_q_build dominates).  scalapack
LU was bit-exact from its first e2e run.  Also KSE'd: the §3.5 native venv
CPU recipe is broken (jax-0.9.1 tiled=False at minimax_screening.py:44).

Extends the 2026-07-10 host-platform port (same branch, commit `f0b17f3` +
review-fix guards).  Report section appended to
`reports/ffi_host_platform_2026-07-10/`.

- **Where ScaLAPACK lives on NERSC**: inside Cray LibSci (`libsci_gnu_mpi`) —
  NOT MKL (whose ScaLAPACK targets Intel MPI, wrong ABI for Cray MPICH).
  `liblorrax_ffi_host.so` already linked libsci for SLATE's BLAS, so the
  backend adds **zero link dependencies** (readelf NEEDED unchanged).
- **`distributed_lu = scalapack`**: fused per-q `pXgetrf`+`pXgetrs`
  (`ffi/scalapack/`), BLACS grid on the slate ctx's rank-remapped comm ("C"
  order = JAX mesh coords), square blocks `g = N/max(Px,Py)` → square + BOTH
  1-D mesh orientations (incl. 1×q, which slate's stride assert forbids).
  Explicit-only, host-only (GPU backend rejected at config parse), same
  `solve_zeta` branch as `cusolvermp_lu` (import switch only).
- **Validation**: contract file 35 passed (mixed platforms, 1 process); CLI
  2×2/4×1/1×4 under `JAX_PLATFORMS=cpu` all `0 failures`, in-container AND
  bare-metal Milan CPU node (12/12 pytest there); full suite **217 passed /
  0 failed (5:36)**, golden gates green.
- **Adversarial review** (index-math mandate): 1 confirmed seam defect —
  scalapack-on-GPU-backend failed loudly but only AFTER the C_q build; fixed
  with parse-time rejection + resolver device check.  All numeric checks
  (BLACS mapping, numroc extents, ipiv, donation aliasing, int32, shared
  branch) refuted with explicit derivations.

## 2026-07-10: USER MANUAL started — outline + chapters 1/4/5/6 drafted [B, source — branch agent/manual, UNPUSHED]

New top-level `manual/` in lorrax (one .md per section, LaTeX math). Approved
outline + editorial threads (T1 two-sums rule, T2 one-picture-three-resolutions,
T3 knob-per-figure, T4 BGW boxes, T5 spinor-first) in `manual/00_outline.md`,
including the pre-writing blockers for Ch. 7 (GN B-normalization contradiction,
three-vs-six-window reconciliation, fresh periodic Sigma(w) gate) and the
freq-doc source map (minimax-quadrature.md -> ctsp_revised.md -> physics.md 6.9
-> GN guide -> Kim-2020 transcripts). Drafted: Ch 1 (intro incl. LORRAX-vs-BGW
pros/cons), Ch 4 (GW in r-space: two-sums rule, Nk log Nk, time-integration
heuristics, QP solvers + rCROP citation Wan & Miedlar), Ch 5 (ISDF: pair-density
ansatz, sigma^0 spinor trace, C_q zeta_q = Z_q stated via P_k, orbit closure,
rank), Ch 6 (Coulomb: truncation, miniBZ MC average, 4x-Ecut pair bandwidth vs
BGW cap, q->0 pointer). lorrax_B also rebased: agent/bxc-vscf-magnetic onto main
adc2197 (clean). Docs dead-weight audit done earlier this session (dev/plans all
landed, ENVIRONMENT_COMPREHENSIVE overlaps installation/, multihost.md ~90%
generic JAX tutorial, hl-gpp-derivation.md is a raw chat transcript in nav).
3-lens review round EXECUTED same session (code-fidelity /
physics-pedagogy / genre-structure agents) -> adjudication -> fixes applied in
0d5822d. Headline: bare_coulomb_cutoff DEFAULT IS NOW ecutwfc upstream (matches
BGW; old 4x-default memory was stale, memory file updated); qp_solver enum
strings fixed; chi0 spin prefactor made spinor-consistent; COHSEX split
un-double-counted; crossing-floor = imposed broadening; 0D box downgraded to
kernel-only (v_q_g_flat raises for sys_dim=0). Report:
reports/lorrax_manual_2026-07-10/. CONTINUED same session: Ch 2 (installation,
cluster-tier phdf5+distributed-linalg framing), Ch 3 (Si tutorial + GN-PPM
upgrade + MoS2 teaser), Ch 7 (full freq-integration chapter) drafted — commits
85d3189, 4102a95. Ch-7 blockers 1+2 RESOLVED from code: GN model is
W_c = 2B*Omega/(w^2-Omega^2) with B = -W_c(0)*Omega/2 (physics.md 6.9 has the
WRONG normalization, flagged for upstream fix); shipped sigma windows = three
(core/a_stripe/b_slab, T = w_max+1.5xi, A_core = 2T/xi), six-window note is
unshipped design. STYLE round 2026-07-11: register locked to Jack's prose (manual/STYLE.md;
1.1 + 2.1 user-revised as references), PRL-density sweep of ch 1-7, 1.2/1.3
archived. Install-doc sweep: pre-JAX-0.9 advice purged (cuda12 extra, 0.5.x pins,
lorrax-bse, load_wfns refs). agent/manual MERGED TO MAIN and pushed (c7a30ff,
also merges origin FFI-host work). Remaining for manual: Ch 8/9, Parts III/IV,
appendices;
blocked items = periodic Sigma(w) gate run (7.6 numbers), tutorial reference
numbers (3.1), T3 figures — all need 1-GPU allocations.

## 2026-07-10: FFI host-platform port — SLATE distributed linalg on the JAX CPU backend [C, source — branch agent/ffi-host-platform, PUSHED to origin/main 4085672]

Executes the P2 "FFI CPU story" follow-up from `reports/slate_linalg_ffi_2026-07-10/`.
Report: `reports/ffi_host_platform_2026-07-10/` (PLAN + full validation matrix).
lorrax_C fast-forwarded to origin/main `adc2197` first; its in-tree GPU FFI rebuilt
from pristine main and re-verified (23/23 contract, 2×2 CLI clean) before any edits.

- **All five slate FFI ops now run on the JAX CPU backend**: host handlers in
  `slate/cpp/host_ffi.cc` (`fromScaLAPACK` on host buffers — same block-cyclic
  layout as `fromDevices`, so every transpose/rank-remap/validation carries —
  `Target::HostTask`, memcpy staging), compiled into a separate **CUDA-free
  `liblorrax_ffi_host.so`** (`common/cpp/host/build_host.sh`, Cray PE, slate
  `gpu_backend=none` + MPI only; readelf-gated; XLA FFI headers staged from the
  CONTAINER's jaxlib — runtime match, the host venv's 0.9.1 headers are too new).
- **jaxlib-style per-platform registration**: same target names under
  `platform="CUDA"` and `platform="cpu"`; wrappers resolve the library from the
  MESH's device platform (`ensure_registered`); `distributed_cholesky = slate`
  passes through on CPU backends (never auto-picked; loud on absence).
- **Validation**: contract file 31 passed (23 CUDA + 8 cpu) mixed in one process;
  multi-rank CLI 2×2 `0 failures` on BOTH backends; bare-metal Milan CPU node —
  clean ldd, 7/7 pytest, 2×2 + 4×1 CLI clean on jax 0.9.1 (forward-compat OK);
  full suite **213 passed / 0 failed (4:15)**, golden gates green.
- **Adversarial review workflow** (4 finders → per-finding refutation, 15 agents):
  6 confirmed findings, all fixed (`d30cfa4`) or documented — headline: three
  classes of platform-blind `get_lib()` callers repointed by the loader change
  (slate lifecycle helpers, five multi-rank drivers whose distributed init
  silently broke, WfnLoader's phdf5 probe on CPU backends), plus the dual-lib
  `libslate.so.2` soname caveat now documented.  Mixed-platform testing also
  caught a batched-wrapper jit-cache key missing device identity (`ec08b04`).
- Open: ScaLAPACK host backend for `distributed_lu` (joins the same .so + loader
  tables); two PRE-EXISTING `test_qp_solver_config` failures under a CPU backend.

## 2026-07-10 (overnight): block-cyclic FFI stability program + SLATE CPU/GPU portability [D, source — branch agent/slate-linalg-ffi, UNPUSHED]

Overnight autonomous program (2 worker agents + orchestrator), 7 commits, final suite
**205 passed / 0 failed** (plain 1-GPU, 4:02).  Full record:
`reports/slate_linalg_ffi_2026-07-10/` (executive summary at top).

Headlines: every block-cyclic linalg FFI swept (mesh × dtype × divisibility ×
determinism) into a permanent skipif-gated contract suite; **cuBLASMp found dead in
production since 2026-05-10** (CAL-ABI stage drift) and fixed; **SLATE distributed_eigh
eigenvector defect root-caused (stale MOSI tiles, not layout) and fixed**; scripted
SLATE builds for Perlmutter GPU AND CPU nodes (testers pass both); portable input-file
axes `distributed_cholesky = auto|off|cusolvermp|slate` / `distributed_lu` with
deprecated cusolvermp_* aliases and loud-failure optional-dependency semantics;
e2e slate ≡ default at 2-4e-6 eV.  Open follow-ups: FFI host-handler port (~1-2 d,
spec'd) for CPU-node execution, SLATE getrf, slate::trsm back-solve, hang-guard
geometries documented.

## 2026-07-10 (overnight): block-cyclic linalg FFI hardening — sweep, 9-item failure catalog, cuBLASMp restored, SLATE eigh fixed, contract suite [D, source]

P1 of the overnight FFI program (`reports/slate_linalg_ffi_2026-07-10/`,
PLAN.md; P2 = SLATE build hardening, same report).  Branch
`agent/slate-linalg-ffi`, 4 commits after P2's `981f1ac`.  Sweep of every
distributed-linalg FFI (cusolvermp potrf/potrs/getrf+getrs/syevd,
cusolvermg, cublasmp gemm/fused-W-solve, slate potrf/trsm/heev/batched)
over 1×1/2×2/4×1/1×4 process meshes × c128/f64 × divisible/non-divisible/
tiny sizes; every cell = numpy-residual + bit-determinism.  Headlines:

- **cuBLASMp had been dead on EVERY mesh since 2026-05-10** (the 0.7.2
  cuSOLVERMp stage ships no libcublasmp → RUNPATH fell back to CAL-ABI
  0.4.0 while cuSOLVERMp used NCCL → grid create status=6; this killed
  `screening_solver = cublasmp_ffi`).  Restored: staged cuBLASMp 0.5.1
  (+nvshmem) via new `stage_cublasmp_redist.sh`, version dispatch now
  asks the loaded library, libcal linked explicitly.  1e-16..1e-14 on
  1×1/2×2 post-fix.
- **SLATE `distributed_eigh` eigvec artifact ROOT-CAUSED + FIXED**: not
  a layout transform — the FFI read stale device tiles (heev's
  back-transform leaves valid Z on host; needed `tileGetForReading`),
  plus a use-after-free race, an XLA-input-buffer mutation, and the
  missing local-transpose pair.  Now returns TRUE eigenvectors,
  `A@Q == Q@diag(W)` ≤4e-14 on 1×1 and 2×2.
- **SLATE trsm with rectangular RHS aborted all ranks** (square-nb X
  tiles + uncatchable OpenMP-task exception) — fixed with
  per-dimension X tiles; m≠n verified at 4e-16 on 1×1/2×2/4×1.
- Library limits guarded with clear errors instead of hangs/status
  codes: cusolverMpPotrf + cuBLASMp = square meshes only (mb==nb);
  cusolverMpSyevd non-square mesh DEADLOCKS; cuBLASMp op(B)≠N on
  multi-rank grids is rank-divergent (deadlock); SLATE 1×q grids
  SIGABRT (lld≠nb stride assert — root of the historical batched-1×4
  assert); `block_size` overrides on multi-rank meshes were silently
  wrong (new `validate_tile_layout`).
- **`tests/test_ffi_linalg_contract.py`**: 21 contract tests (~19 s,
  1 GPU, skipif-clean without the FFI stack) + CLI mode for multi-rank
  meshes.  Suite: **197 passed / 0 failed (4:03)**.
- cusolvermg: confirmed bench-only (no live consumer) — kept.

## 2026-07-10: Test-suite redesign — 3-tier architecture, 578→~220 s plain 1-GPU run, fixtures shrunk, bispinor gate → GN-PPM [D, source]

On lorrax_D `agent/driver-transparency` (5 commits, `df5befe`..`f3a982a`).
Report: `reports/test_suite_redesign_2026-07-09/`.  The suite now meets the
plain-invocation contract: `LORRAX_NGPU=1 lxrun python3 -m pytest -q tests`
= **176 passed / 24 deselected in 218-258 s** (two runs), no xdist/srun
overrides (xdist stays optional; worker→GPU pin kept; 4-GPU `-n 4` verified).

- **Tier 1** (4 frozen e2e pins): si_cohsex_3d (BGW anchor, untouched),
  cohsex (kept — only IBZ-stored WFN ⇒ only e2e ψ-unfold), gnppm (shrunk
  642→399 orbit-closed centroids, ncond 54→20, nband 80→46; re-frozen;
  IBZ-cascade activation now log-asserted), bispinor (upgraded static
  COHSEX → **bispinor GN-PPM** — runs at HEAD once `whead_imfreq` is
  provided, a lost-wiring find; 640/668→256/209 centroids, 116→40 s).
- **Tier 2** (7 invariance gates, self-checking): restart≡fresh (NEW),
  μ-pad flips, kij↔kij_stream, SC-iter1≡one-shot, fixed-point rotations,
  IBZ≡full-BZ — all but the bispinor pad flip run as `restart = true`
  variants from a COPY of the gnppm session state (session-scoped pytest
  fixture; the ζ-fit/V_q happen once).  Measured on the shrunk fixtures:
  restart, SC-1, and BOTH pad flips are bit-identical; kij_stream 4.5e-13;
  IBZ≡full exactly 0.
- **Tier 3**: 37 unit files → 17 (+5 behind a new `-m extra` marker,
  deselected via pyproject addopts).  Deleted: planner-refit archaeology,
  io_callback nesting smoke, rchunk/gflat pair, pivoted-Cholesky, 690-line
  kmeans suite (→ ONE hex smoke test).  Merged: symmetry/TRS ×3 → 
  test_symmetry_unfold; zeta loader+reader+mf_isdf+slab_io → test_file_io;
  minimax ×2; wfn_loader ×2; per_q_sphere→V_q; psi_g_store→zq.  Full
  triage + pin-mapping tables (all 13 old gates accounted) in the report.
- Fixes: `charge_density.py` missing `import jax` in the wfn_ibz ρ
  fallback (df5befe).  Checkpoint skill updated (plain invocation is the
  standard; golden-gate names refreshed).  KNOWN_SANDBOX_ERRORS: module
  load in non-login shells entry.

## 2026-07-09 (evening): pushed to origin/main; SC-driver move; I/O cleanliness batch (zeta merge #7); 4-GPU parallel test suite [D, source]

**origin/main fast-forwarded to `d03c857`** (66 commits: the whole memplanner-cleanup program
+ the driver-transparency line).  Four new commits on `agent/driver-transparency`, each
suite-gated (258 passed / 9 skipped):

- **`60ca703` SC machinery → `sc_iteration.run_sc_driver`**: main()'s last machinery slab
  (partition, SCInputs, loop, dumps, QP→DFT rotate-back) moved out; run_sc_driver returns a
  DFT-basis SigmaResult so SC and one-shot share ONE post-Σ seam.  gw_jax.py = 502 lines,
  every block a physics stage call or input-flag pivot.
- **`cece78c`** (earlier) sub-driver audit: minimax builders → engine, gw_driver_helpers
  deleted, 5 dead symbols removed.
- **`eec4de8` I/O cleanliness** (audit in reports/driver_transparency_2026-07-09/):
  zeta_loader/zeta_reader **MERGED** (NEXT_TARGETS #7 CLOSED, −381 L; fixes the
  advisory-backend bug — ZetaLoader's SlabIO now receives backend=; the dead eager/phdf5
  string axis deleted); the v_q_g_flat load-vs-read_zeta_G_slab duck-typing deleted (one
  padded-μ read shape — the merge had flipped production onto the un-padded path, caught by
  test_mu_pad_invariance); FFI async write loop → common.async_io.AsyncDispatcher (single
  source; write-behind semantics unchanged); **explicit `slab_io` input key** (all 3 backends
  reachable: phdf5_ffi / phdf5_host / h5py_allgather; use_ffi_io stays the legacy auto-route);
  dead I/O API deleted (slab free funcs + accumulate_slab ×4 + cache alias, −190 L).
  Audit verdicts kept as-is: reads are parallel in ALL backends (no rank-0+broadcast
  anywhere); mpi_host sync writes + no ζ-read prefetch are deliberate (documented).
- **`d03c857` 4-GPU parallel test suite**: pytest-xdist + conftest worker→GPU pin (must
  OVERRIDE SLURM's gres CUDA_VISIBLE_DEVICES — first rollout ran every gate on a 4-device
  mesh and all 13 gates failed on their 1-GPU-frozen refs).  Measured 404 s cold-compile
  parallel vs 470-580 s warm serial (warm parallel ≈ 200-250 s; critical path =
  bispinor_pad4 gate 203 s).  checkpoint SKILL updated with the invocation + the lxrun
  task/GPU-coupling caveat; `-m "not regression"` documented as the 1-2 min unit-only loop.

Deferred: FFI create_dataset drops `chunks` (live write asymmetry — needs a perf-validated
fix); mem_probe → gpu_utils; duration-aware xdist scheduling for the 203 s straggler.

## 2026-07-09: Driver transparency B+C executed — main() IS the scaffold; SC-iter1 ≡ one-shot GATED; 2 SC bugs fixed; GN-PPM ULP amplification measured [D, source]

On lorrax_D **`agent/driver-transparency`** (base 9925d43), three commits, full suite green
after each (final: **258 passed / 9 skipped**, all 12 prior e2e gates bit-identical).
Report: `reports/driver_transparency_2026-07-09/`.

**Phase B (`3102994`, pure moves):** the IBZ slice/solve/unfold block →
`screening.compute_static_w`; restart W0+head flush → `gw_output.persist_w0_and_head`;
freq-debug table → `gw_output.write_freq_debug`; one-shot WFN_qp dump →
`gw_output.write_qp_wfn_oneshot`; degen averaging → `degen_average.average_sigma_components`;
one `enk_dft` fetch (was 4×). Bit-identical.

**Phase C (`160d22d`, the unification):** one-shot main() now consumes the SAME
`screening_requests_for → compute_screening → compute_sigma_xc` pipeline as the SC loop —
the inlined duplicate deleted; `qsgw_utils.solve_qp` = update_H[Σ; qp_solver]; dispatch
gained bispinor pass-through + streamed-Σ_c branch; static W in SC now IBZ-solved; the mode
enum is load-bearing (explicit `x_only` used to silently run COHSEX). **NEW GATE**
`tests/test_sc_oneshot_equivalence.py`: SC-iteration-1 ≡ one-shot at 1e-6 on
sigma_diag+eqp0+eqp1. The gate caught **two real pre-existing SC bugs**: (1) the output seam
rotated Σ back to DFT with the CONVERGED U (one iteration ahead — wrong before the fixed
point; tens of eV at max_iter=1) → `SCState.last_sigma_basis_U`; (2) iteration-0
`eigh(diag(E_DFT))` roundtrips eigenvalues at ~1 ulp and **GN-PPM amplifies that to
0.44 eV in Σ_c** → exact eigensystem for the exactly-diagonal carry.
**Measured pathology (open):** +1 ulp on every WFN energy → max|ΔΣ_c(ω)| = **1.28 eV**
(bit-deterministic rerun = 0.0; census/windows/nodes unchanged — near-threshold pole-mode
fit values). Same family as device-invariance Fix-3; needs a conditioning decision.

**Sub-driver audit executed (`cece78c`):** DELETED `get_cohsex_kernels`,
`get_effective_chunk_size`+`meta.chunk_size`+`chunk_size` key (write-only no-op chain),
`gw_init.get_bandranges` (dup), `flatten_V_qmunu` (legacy shim), the compile-cache triple
alias; MOVED the minimax quadrature builders w_isdf → minimax_screening (w_isdf now pure
χ₀/W, 759→534 L); `gw_driver_helpers.py` DELETED via fan-out (profile_section →
common/jax_profile, `resolve_input_path` → file_io.paths, build_bgw_v_grid_fn →
compute_vcoul, setup_runtime → gw_jax-local). Deferred: mem_probe → gpu_utils.
`gw_jax.py` 991 → **637 lines**.

## 2026-07-09: gw_refactor_map report dir consolidated — HANDOFF.md is the fresh-session entry point [reports only]

`reports/gw_refactor_map_2026-07-01/` reduced to 10 live docs + `archive/`. Read order for a
fresh session: **HANDOFF.md first**, then STATUS.md + NEXT_TARGETS.md (both updated 2026-07-09:
TIER-0★A static_limit and TIER-0★B device-invariance marked DONE with the correct padded-μ root
cause; driver-transparency B→C added as the new top item; zeta_loader/zeta_reader merge marked
UNBLOCKED by `tests/test_mu_pad_invariance.py`). Executed plans, audit catalogs, verify scripts,
raw audit JSON, and run-debris dirs moved to `archive/` with a per-file README explaining why;
stale cross-references in kept docs + this file repointed. No source changes.

## 2026-07-08: ppm_invalid_mode=static_limit implemented + made DEFAULT (BGW mode 3) — WS6 last physics item [D, source]

On lorrax_D `agent/memplanner-cleanup` (fdf89c2 + follow-ups). Invalid PPM poles (Ω²<0) are now,
by DEFAULT, excluded from the τ-pole sum with an analytic ω-independent **static-COHSEX term**
added for exactly those modes: `Σ_static = sigma_sx(G_occ, Wc0·inv_mask) + sigma_coh(Wc0·inv_mask)`
(cohsex kernels reused; `Wc0_q` retained on PPMBuildResult, replacing the dead W0_q/Wiwp_q; term is
pad-neutral and lands identically on kij/kij_stream). **Physics note:** BGW mode 3 keeps BOTH the
static SEX (occ) and CH terms (`mtxel_cor.f90` ω̃→∞: ssx→−I_ε, sch→−½I_ε) — the research note's
"SX pole → 0" was wrong; the implemented term is the exact Ω→∞ limit of the pole sum (occ→−½Wc0,
unocc→+½Wc0). **Validated three-way on Si 4×4×4** (same-code rerun triple 03b_zero/03b_2ry/03_static
vs BGW m0/m2/m3; `reports/bgw_invalid_mode_refs_2026-07-08/lorrax_mode_table3.dat`): PASS — deep
valence up (+1.5 meV vs BGW +17), window-top down (−0.5 vs −5.2), VBM smallest; mean|Δ| ~12× below
BGW (≈ population ratio 1.13%/8.63%); Δ(static−2ry)=0.21 ≪ Δ(static−zero)=0.71 meV (BGW 3.05≪8.56);
statics bit-identical. Per-q n_invalid diagnostic added (Si: strong q-clustering 166–11134/q).
**Default flip re-freeze:** gnppm golden + fixed-point rotations refs re-frozen (MoS2 3×3 fixture,
0.47% invalid: ΔΣc mean|Δ|≈40 meV, max 0.32 eV — 2D Wc0 scale, band-1/deep-valence dominated;
Σ_X/V_H bit-identical; 2ry−zero cross-check same order). COHSEX/si_cohsex_3d/bispinor/IBZ gates
untouched (static path — verified green). Docs/templates: COHSEX_INPUT.md, templates/cohsex.in,
SIGMA_PPM_MAP §2C marked done, GN_PPM guide §3.4/§8.

## 2026-07-08: qp_solver toggle — G0W0 (one_shot_dft) default / fixed_point / self_consistent + SCConfig + qsgw jit hoist [D, source]

Implemented `reports/gw_refactor_map_2026-07-01/G0W0_SC_TOGGLE_DESIGN.md` on lorrax_D
`agent/memplanner-cleanup` (base 620b501). **New config axis `qp_solver`** (how QP energies
are extracted from Σ, orthogonal to `compute_mode`): `one_shot_dft` (DEFAULT — textbook G0W0,
QSGW-build eigh evaluated at E_DFT, consistent with eqp0) | `fixed_point` (previous
dynamic-mode behavior: diagonal on-shell solve + scissor; dynamic modes only) |
`self_consistent` (QSGW loop). `auto` resolution absorbs the deprecated `self_consistent`
bool AND the orphaned `sigma_at_dft_energies` (TIER-0★ A: its intended meaning IS the new
default; both keys deprecation-warned, still honored). Validation errors for
`fixed_point` × static and `fixed_point|self_consistent` × explicit `kij_stream` (was a
silent static-eigh degradation). **`LORRAX_SC_*` envs promoted to `SCConfig`**
(`sc_max_iter`/`sc_tol_ev`/`sc_accelerator`/`sc_history_depth`/`sc_mixing`/`sc_dump_dir`;
envs remain deprecated overrides with a printed note — closes NEXT_TARGETS #11). **Default
flip proven a pure re-labeling** on the GN-PPM fixture: sigma_diag/eqp0/eqp1 bit-identical
(timestamp-only header diff); only the eigh family moves (max 1.215 eV, design §2a); new gate
`test_gnppm_fixed_point_reproduces_frozen_qp_rotations` pins `fixed_point` to the pre-toggle
frozen E_qp (ref .npy from HEAD 620b501). **qsgw jit hoist:** `_extract`/`_kernel`
(qsgw_utils nested-scope closures) hoisted to module-scope mesh-keyed factory caches —
SC-GN-PPM compiles/iter 91/8/**2** → 91/6/**0** (steady state now 0 compiles / 0 retraces,
matching COHSEX); SC RMS trajectories bit-identical. Stale "only COHSEX is wired" comments
fixed (SC-GN-PPM verified e2e). templates/cohsex.in + docs_gwjax/COHSEX_INPUT.md + skills
updated to the new keys. Validation runs + before/after compile tables:
`reports/gw_refactor_map_2026-07-01/archive/g0w0_sc_toggle_impl/README.md`. Suite: full pass
(golden gates bit-identical; +16 new tests: 15 config-unit + 1 fixed-point freeze).

## 2026-07-08: Padding consolidation executed — PADDING_AUDIT items 1–7, net −553 lines [D, source]

Executed the ranked consolidation from `reports/device_invariance_2026-07-08/PADDING_AUDIT.md`
(new **AS-CONSOLIDATED** section has per-item verdicts) on lorrax_D `agent/memplanner-cleanup`,
commits `6c850bd..620b501` (pushed). Every audit item re-verified live at HEAD before acting;
full suite after EVERY commit; goldens bit-identical throughout. **Two latent bugs fixed:**
(1) restart writers persisted the P-dependent PADDED μ extent (V_qmunu/G0_mu_nu/W0_qmunu/ψ-μ +
init_W0 placeholder) — now clipped to logical via SlabIO `valid_shape` with re-pad-on-read via
`padded_mu_extent`; a restart written at one P is now readable/bit-correct at another (new gate
`tests/test_restart_pad_roundtrip.py`; bse_io's `lcm` divisor retired — ONE μ round-up convention);
(2) the allgather zeta_q_G bypass wrote the padded gather into the logical dataset (crash under
any μ pad) — deleted by unifying both backends on one SlabIO write path. **Consolidation:** pad
modes now born DEAD (Ω=0) at the GN-PPM fit with a REQUIRED `n_mu_logical` (consumer-side
`mu_logical_mask` arm + driver mask deleted; census structurally pad-safe); `Meta.*_jax`
wrong-divisor fields deleted + `w_isdf` getattr fallback → hard `meta.n_rmu` read; the dead
PadAxis shadow API + `tests/test_padding.py` deleted (−757; zero callers); ONE `round_up`
spelling; ONE `solve_at_logical` wrapper for the slice-to-logical solve idiom (5 sites — the
grep-able invariant for the ROOT_CAUSE defect class) + `pad_last_axis_to` for the triplicated
NRHS pad; μ pad baked into the IBZ sym tables at construction with strict both-direction extent
guards in `unfold_v_q`/`_unfold_g0_ibz_to_full` (closes the silent `promise_in_bounds` OOB).
Suite 249→241 passed / 24→9 skipped (= exactly `test_padding.py`'s 24 tests removed + 1 roundtrip
gate added). Known remaining: restart ψ/enk BAND axis still stores padded `b_id_4` (P-dependent
iff `nband % world != 0`); Fix-3 robustness items unchanged.

## 2026-07-08: Device-invariance μ-pad fixes — bispinor 4g↔16g eqp 2.535 eV → 1.45e-7 eV [D, source + runs]

Implemented the ROOT_CAUSE fix plan (`reports/device_invariance_2026-07-08/ROOT_CAUSE.md`, now with
an AS-FIXED section) on lorrax_D `agent/memplanner-cleanup`. **Root cause (proven): solves ran on the
PADDED μ extent `round_up(n_rmu, world_size)`, so the pad extent — which changes with device count —
deterministically changed the answer.** Commit series: (1) `LORRAX_EXTRA_MU_PAD` promoted to a
permanent env-only test knob via `runtime/padding.py:padded_mu_extent` (single source of truth; the
knob exposed and unified two V_q-side local pads that computed their own round-up — pre-fix shape
crash); (2) transverse ζ_T LU + charge triangular back-solve + single-device dense Cholesky +
per-q W-Dyson LU all μ-sliced to the LOGICAL extent with zero-filled pad rows, LU ridge on the
logical trace/n, false "LU logical-block bit-identical" docstring corrected (cuSolverMp falls back
to per-q LU when the logical extent isn't per-axis divisible); (3) PPM mode census / invalid count /
masked-Ω window stats / unfulfilled fraction on LOGICAL modes only; (4) gates + `g0_mu` on-disk
extent clipped to logical (was written padded — caught by the new Tier-2 gate).
**Confirm run** (§6, `A_charge/padtest_4g_pad12`, knob-only): census inflation exactly
pad-deterministic (720,620 = base + 464,640 pads + 0), Σ_C moves 5.14 eV at FIXED P — but the 16g
minimax node change (15→13/14) and "+2 flipped modes" are NOT pad-driven (cross-P noise flipping
near-threshold divergent-Ω modes). **Post-fix:** bispinor 4g↔16g max|Δeqp| **2.535 eV → 1.45e-7 eV**
(Σ^B tile22 −117.914 → −0.152608 = 4g exactly); charge census now P-invariant; charge eqp spread
unchanged (max 5.64 eV on-pole, 0.27 eV on the 26 |Im Σ_C|<100 eV bands) — carried by on-pole
GN-PPM ill-posedness + the noise-driven node-count flip, both Fix-3-class robustness items, NOT pad
defects (deliberately not forced). Both 4g postfix reruns BIT-IDENTICAL to pre-fix (no pad at P=4);
full suite 247 passed / 0 failed. **Gates:** Tier-1 in-suite `tests/test_mu_pad_invariance.py`
(fixed-P pad flip, 1 GPU: bispinor fully bit-identical incl. the catastrophic 672 extent; gnppm
census/nodes/Σ_X exact + Σ_C ≤ 2e-4 eV — bit-identity unreachable for the dynamic chain: XLA
fusion tiling gives 1–2 ULP on ζ/V per pad extent, amplified to 6.3e-5 eV through the near-singular
PPM fit ratio); Tier-2 multi-GPU script `tests/multi_device/run_tier2.sh` (P=1 vs P=4) — first run
PASSES both fixtures under measurement-calibrated tolerances: node counts + n_total exactly equal,
ζ_C 2.8e-6, Σ_X 1e-6 eV, bispinor ζ_T 2e-7 (vs 0.9–5.5e2 pre-fix); Fix-3-gated residuals reported
honestly (invalid split ±8 near-threshold flips; off-pole eqp 69 meV from discrete mode-flip
add/drop under `ppm_invalid_mode='zero'`, provisional 0.1 eV gate; bispinor fixture ζ_C 0.25 frob =
rank-deficient-CCT null space, physical Σ_SX 1.85e-4 eV / eqp 34 meV within provisional bounds).

## 2026-07-08: LORRAX ppm_invalid_mode zero-vs-2ry validated against the BGW references [D, runs+analysis only]

Ran the two wired invalid-pole modes on the BGW-matched Si 4×4×4 window (new variants
`runs/Si/00_si_4x4x4_60band/03_lorrax_gnppm_invalidmode_{zero,2ry}`, single-key cohsex.in diff,
1×A100 each). **Invalid population: 167,092/14,745,600 ISDF poles = 1.13%** (identical in both
runs; vs BGW 8.63% of plane-wave pairs — same order, non-vacuous). **Verdict: wiring PASS,
quantitative BGW anchoring FAIL**: Δ(2ry−zero) per band matches BGW's Δ(m2−m0) in sign at the
window edges (deep valence +1.1 meV vs +22.7; window-top −0.5 vs −3.3; VBM ~unmoved in both) but
is 15–34× smaller overall (max 1.16 vs 39.6 meV ≈ population-ratio scaling) and mid-conduction
sign flips — the ISDF invalid-pole population sits on different pairs than BGW's (BGW's: 8.2% of
Σ|W_c|, all off-diagonal, every q). Statics bit-identical between runs; star-symmetry spread
≤0.4 meV. Per-q localization of LORRAX's invalid poles remains open (April `w_copies_debug.h5`
stores full W, not W_c — offline refit confounded; needs a 1-line per-q count print, source
session). Full table + dig: `reports/bgw_invalid_mode_refs_2026-07-08/lorrax_zero_2ry_validation.md`
(also the static_limit three-way scaffold). Ops: ω-grid −15 eV drove the ω<E_F crossing minimax
fit to A_core≈124 where the exact Remez solver stalls (>7 min); −13 eV + ξ=0.35 (A_core≤77) is
fast — recorded in the report. Compare-skill §2c updated with the header-driven
`parse_sigma_freq_debug_v2` (current named-column format; old parser kept as legacy);
KNOWN_SANDBOX_ERRORS: COHSEX_INPUT.md `ppm_invalid_mode` section is stale (documents
`static_limit` default + `fixed_2ry`; code = `zero` default, `zero/skip/2ry`). No LORRAX source
changes.

## 2026-07-08: G0W0-vs-SC toggle design + Σ-pipeline single-jit audit [D, analysis only]

Delivered `reports/gw_refactor_map_2026-07-01/G0W0_SC_TOGGLE_DESIGN.md` (NEXT_TARGETS TIER-0★ A
`sigma_at_dft_energies` wiring + #11 env-knob promotion, designed not implemented). **Proposal:**
one new axis `qp_solver = one_shot_dft (default, textbook G0W0) | fixed_point (today's dynamic-mode
on-shell diagonal solve) | self_consistent (QSGW loop)`; absorbs the orphaned `sigma_at_dft_energies`
and legacy `self_consistent` via `auto`-resolution; `LORRAX_SC_*` (5 knobs + DUMP_DIR) promote to an
`SCConfig` group; eqp0/eqp1 formulas invariant across all three states. ~30-line sketch at
`gw_config.py` + `gw_jax.py:475/:649`. **Jit audit (empirical, MoS2 fixtures, 1 GPU,
JAX_LOG_COMPILES):** SC-COHSEX steady state = **0 compiles / 0 retraces per iteration** (iters 2-4);
SC-GN-PPM steady state = **exactly 2 retrace+recompiles per iteration**, both from nested-scope
`@jax.jit` closures (`qsgw_utils.py:165 _extract`, `:262 _kernel`) — hoist to module scope to reach
0. Iter-1 pays one extra specialization of the χ₀/W/Σ kernels (rotated-bundle committedness).
**Findings:** SC IS wired for GN-PPM (ran 3 iters end-to-end; `gw_config.py:51,190` comments stale);
streamed `kij_stream` × SC/fixed_point silently degrades (static Σ in the eigh, eqp1 Z lost);
pre-SC one-shot W+Σ pass is redundant for SC runs; static-mode eqp0 writer vs freq_debug disagree on
Σ_SX−Σ_X (flagged, unverified). Runs + parser: `reports/gw_refactor_map_2026-07-01/archive/g0w0_sc_toggle_audit/`.
No LORRAX source changes.

## 2026-07-08: BGW invalid_gpp_mode reference set (Si 4×4×4) — LORRAX ppm_invalid_mode validation targets

Produced complete three-mode BGW reference lines for invalid-PPM-pole handling on
`runs/Si/00_si_4x4x4_60band` (reused WFN + eps; 4 new ~29 s sigma variants, job 55674176).
**GN-GPP line** (freq_dep=3, = LORRAX GN-PPM flavor): `01b`(mode 0) / `01_bgw_gn_ppm`(mode 2,
pre-existing) / `01c`(mode 3) / `01d`(keyword omitted — **bit-identical to mode 3, default
verified**). **HL line** (freq_dep=1): pre-existing `02b`(0)/`02c`(3) + new `02d`(2); `02d` is
bit-identical to the mutated `02_bgw_hl_ppm` → that legacy run was mode 2. **Invalid poles are
plentiful**: exact offline count (BGW's own formula on ε⁻¹(0)/ε⁻¹(iω₂)) = 186,608 of 2,162,680
(G,G′) pairs = **8.63%**, uniform across q. Physical size of the choice: GN ΔEqp0 max 51 meV
(mean 8.5); HL max 104 meV (mean ~35). Movers: deep valence + top-of-window conduction; VBM
manifold nearly immune. Γ direct-gap shifts 6–8 meV (GN), 18–49 meV (HL). LORRAX validation
protocol (compare mode-to-mode DELTAS, not absolute Σc; GN line primary): see
`reports/bgw_invalid_mode_refs_2026-07-08/report.md`. Also: extended compare-skill
`parse_sigma_hp` for the 11-column freq_dep=1 sigma_hp.log layout (was silently returning zero
blocks — KNOWN_SANDBOX_ERRORS entry filed). No LORRAX source changes.

## 2026-07-03: ISDF memory model redesigned — one planner, live-validated [D]

Rewrote `gw/gflat_memory_model.py` to the `MEMORY_MODEL_DESIGN.md` §1a form: **persistent(P)
floor + max over five stage transients**, two-phase picker (rank floor `P_min` → dial `chunk_r`),
q/k folded in at the actual `chunk_r`. **Deleted** `gw_init.compute_optimal_chunks` (the legacy
384-line 5-moment band/r model + helpers) — there is now ONE planner call in
`prepare_isdf_and_wavefunctions`, not two stacked ones; `fit_zeta` is a pure consumer. LOC:
`gflat_memory_model` **922→374**, `gw_init` **1169→741** (net −976 planner lines). Resolved the six
§5 divergences: **centroid term fixed to ÷√P single-axis** (was ÷p_xy — √P× under-count), ns²-aware
util (0.90/0.85/0.78), stale `n_bc` dropped, sphere-idx dropped as negligible, FFT box XLA-queried,
production cuSolverMp solve (no replicated-L). Added the `P_min` rank floor.

**Live validation** (job 55447628, BFC `peak_bytes_in_use`, MoS2 charge ns=2 + bispinor ns=4,
4/16 GPU × 3 budgets): predicted HWM tracks real BFC peak to **≤0.1%** wherever the algorithmic ÷P
memory binds — e.g. bispinor 4-GPU/28 GB **21.84 vs 21.85**, bispinor 16-GPU/28 GB **21.84 vs 21.85**,
all six 4-GPU cells 8.5–21.9 GB within 0.15%. The ns²-aware util (0.78 for ns=4) is the one
calibration — it fixes a bispinor single-arena OOM (a 23 GB contiguous pair-density buffer). **Known
gap**: at 16 GPU / 4 nodes there is a ~8 GB P-independent runtime floor (NCCL/cuFFT/CUDA context) that
the shape-algebra model does not predict; it under-predicts only when algorithmic memory falls *below*
that floor (low-occupancy = huge headroom, never OOMs) — reported, not fudged (design §6 puts NCCL out
of scope). **5 e2e gates green** (gnppm reference re-frozen: new valid `chunk_r` shifts the
chunk-order-sensitive Σc by ≤5.9e-5; sigX/VH bit-identical — see KNOWN_SANDBOX_ERRORS). Fixed the
stale `gw_isdf` import in `tools/profile_gw_xprof.py`. Report + logs:
`reports/memory_model_refit_2026-07-03/`. Branch `agent/memplanner-cleanup`.

## 2026-07-03: First bispinor e2e regression gate [D]

Closed the last gate gap. Added `tests/regression/bispinor_debug/` + a parametrized `bispinor` case to
`tests/test_gw_jax_regression.py::_CASES` — the first e2e gate on the **bispinor path** (nspinor=2,
screened-charge COHSEX Σ_SX+Σ_COH on W⁰⁰ plus bare Breit Σ^B; 4 ζ channels / 7 V_q tiles / transverse γ̃).
System = MoS2 3×3, the hand-validated `reports/bispinor_screened_a_validation_2026-06-16/` config.
**Fixture WFN** = the source 82-band WFN truncated to **34 bands** (52.6 MB, comparable to the 46 MB gnppm
fixture; a buffer over the nband=32 window) — verified **BIT-IDENTICAL** to a full-82-band run on current
code (bands above 32 never enter the COHSEX/ζ window; the ~0.08 meV vs the 2026-06-16 baseline was code
drift, not truncation). **Runs on 1 GPU** (`LORRAX_NGPU=1`, `memory_per_device_gb=30`) — the standard pytest
harness drives it, no multi-GPU marker needed. Σ^B folds into the **sigSX** column, so labels are the same
`sigSX/sigCOH/sigTOT` as scalar COHSEX (no separate Σ^B columns); existing `_parse_eqp_rows` unchanged.
IBZ closure fails on this non-orbit-closed centroid set so the run is full-BZ-direct; Σ_X/Σ^B is covariant,
so the freeze is valid. Full suite: **243 passed, 24 skipped, 0 failed** (regression subset now 5 gates:
cohsex, gnppm, si_cohsex_3d, ibz_full_bz_equivalence, bispinor). Committed + pushed to
`agent/memplanner-cleanup` (`3fc93b4`). Note: WFN.h5 is 52.6 MB → GitHub soft-warns (>50 MB) but accepts.

## 2026-07-02: GN-PPM regression-gate seed on a FRESH full WFN (MoS2 3×3) — clean run [D]

New run `runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02/`. Ran QE **fresh** (SCF 11 it → NSCF → pw2bgw →
wfn2hdf) to a clean full `WFN.h5` (9 k, 82 bands, nspinor=2), plus a shifted `WFNq.h5`. **GN-PPM
(`compute_mode=gn_ppm`) ran EXIT 0** on it in ~27 s (4 GPUs, cuSOLVERMp 0.7.2/NCCL/2×2). Key result:
the `enk_full = wfns.enk[:, s.full]` / `build_G` einsum crash that the reduced `WFNsmall.h5` gate fixture
triggers **did NOT recur** — confirming that crash is **fixture-specific** to the truncated WFNsmall, not a
GN-PPM-path bug. No source change; no diagnostic print needed. Baseline artifacts for the gate:
`00_lorrax_gnppm/{eqp0.dat,sigma_diag.dat,sigma_freq_debug.dat,sigma_mnk.h5}`. Also ran a BGW
`frequency_dependence 3` reference (`01_bgw_gnppm/`, epsilon+sigma Job Done): LORRAX `Re sig_c(Edft)` tracks
BGW `Corp` (SX-X+CH') at Γ to **MAE 1.75 eV** (mean +0.77, near-constant offset — same 2D head/convention
family as the COHSEX study; shape+sign agree). Caveats (reproduced from the reference run, not regressions):
deep FR-pseudo states carry huge imaginary Σc (GN "invalid modes" 1.09%), and conduction-band `eqp0` diverges
from the `QSGW: 444 clipped (61.7%)` diagonal-SC clipping — only near-Fermi **real** Σc is the meaningful
cross-code quantity. **Sandbox finding:** `skills/compare/SKILL.md` §2c `parse_sigma_freq_debug` is **stale** —
`sigma_freq_debug.dat` is now 14 tab-cols with `sig_c(Edft).Re` at **col 8** (col 12 is `eqp0`); the old parser
reads eqp0-as-Σc and yields a bogus ~8 eV MAE. Corrected layout logged in KNOWN_SANDBOX_ERRORS.md. Note:
kmeans emits **642** centroids (orbit closure), so cohsex.in uses `centroids_frac_642.txt`.

## 2026-07-02: GW refactor map + gate-0 repair + cusolvermp fix + MoS2 exact-agreement study [D]

Three interlocked initiatives, all under `reports/gw_refactor_map_2026-07-01/`. **(1) GW refactor map:**
agent fan-out over ~150 files → teleological catalog (`MAP.md`, `FEATURES.md`), 1049 adversarial verdicts
(`DEAD_CODE.md`: 212 confirmed-dead, 185 refactor targets, 74 suspected bugs), full 128-flag surface (`FLAGS.md`),
regression-gate coverage audit (`GATE_AUDIT.md`). Landed a **delete-pass** (5 verified-dead modules ~1600 L) +
a **CONVENTIONS** section in AGENTS.md (sole sharded-FFT helper, k/q flat axes for NUFFT, io_callback host caches,
one sym table, no parallel old/new) — branch `agent/gw-delete-pass`. **(2) gate-0:** the e2e regression gate was
RED on main — fixed the `write_qp_wfn_h5` crash (IBZ/full-BZ Σ mismatch → skip-not-crash) + added an `Eo` column
to sigma_diag for BGW band-alignment; branch `agent/gate-0-qpwfn`, unit suite 250-green. Remaining: re-freeze
`eqp_ref.dat` (benign k-uniform ~3.3 meV W-drift from fc1602a). **(3) cusolvermp FFI "deadlock" RESOLVED:** the
`lorrax_D` modulefile FFI defaults still pointed at purged `$SCRATCH` — repointed to `$HOME/software/lorrax_*`;
cuSOLVERMp 0.7.2 potrf now completes on a 2×2 mesh (4-GPU ≡ 1-GPU to 1.3e-5). A/B/C modulefiles still need the
3-line edit. **(4) MoS2 exact-agreement study:** built a BGW-noavg COHSEX reference (`cell_average_cutoff 1d-12`)
and swept the q→0 head source — native (no overlay, s_tensor head) is 6-20× better than the BGW-vcoul/head overlay
(**62 meV** vs 368 `epshead` vs 1278 explicit-`vhead`). The Si 0.12 meV overlay recipe does NOT transfer to 2D-slab
(ε⁻¹[0,0]≈0.96). MoS2 2D sub-meV is a genuine research problem — residual is the static-CH partition + 2D q=0 head;
lead for a future session: read `whead` directly from BGW `DEBUG HEAD TERMS`/`wcoul` instead of `vhead·ε⁻¹`.
Bug ledger in `BUGS_FOUND.md` (14 issues); infra in KNOWN_SANDBOX_ERRORS.md. **Exact-agreement gate recommendation:**
Si 4×4×4 0.12 meV pair (Option A, zero-GPU) for the sub-meV anchor; freeze MoS2 native run (05) as a ~62 meV 2D gate.
**(5) Gate safety net completed + real GN-PPM bug fixed.** Re-froze the COHSEX gate reference (was silently RED
from the benign fc1602a W-drift) → green. Then discovered GN-PPM crashed on 1 GPU (all prior "WFNsmall GN-PPM
crashes" were actually `LORRAX_NGPU=1`): `build_G_tau` mask_A from `wfns.occ` carries a leading nspin axis a 1×1
mesh doesn't squeeze → `phases` broadcasts to 3-D and build_G's 'kn' einsum fails. Fixed (reshape mask to enk.shape;
no-op on 4-GPU). A subagent ran QE cleanly (SCF→NSCF→pw2bgw→wfn2hdf) to make a fresh full MoS2 3×3 WFN
(`runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02/`), which is what exposed it as 1-GPU-not-WFNsmall. Wired a
parametrized GN-PPM regression gate (dynamic Σc, sigX/sigC/sigXC) alongside COHSEX — **both green** (`2 passed`,
1 GPU). Branch `agent/gnppm-1gpu-mask-fix` stacks the full net (crash fix + Eo + COHSEX re-freeze + mask fix + GN-PPM
gate); unit suite 250 passed. Also: cusolvermp fixed on 4 GPU (modulefile FFI paths repointed).
**(6) Refactor step 2 (single-source):** cohsex.in parser ×2→1 (deleted the diverged `get_DFT_mtxels`
duplicate, reconciled `sys_dim` 3→2 + `ecutrho`→WFN `ecutwfc`; 252 passed); stripped `gw/vcoul.py` 234→72 L
(dead commented `get_V_qG` + 2 zero-caller fns). Found eqp0/eqp1/Z already single-sourced in `eqp_bgw`
(stale map entry). Remaining step-2 items (`load_wfns`, zeta readers) are real migrations, deferred to
step 3+. **(7) Memory-planner cleanup (step 3, C6):** a mapping workflow found THREE "memory models" — gflat
(1007L, sole live closed-form planner) + `gw/aot_memory_model/` (3488L DOE/NNLS/chooser framework, DEAD-by-clobber:
`plan_gflat_chunks` always overwrote its chunk picks) + `runtime/aot_memory.py` (519L live cuFFT-scratch query).
Deleted the aot_memory_model package + `gw_init._apply_aot_chunk_model` + `gw_config` chooser knobs = **~8775 lines**
(252 passed, both gates + planner units). Rewrote `docs/architecture/memory-model.md` 1010→817 (killed the dead
'AOT Memory Model' section + the 'four choosers' code-vs-doc contradiction) and the gflat module docstring 76→42
physics-first. Branch `agent/memplanner-cleanup`. Plan: `reports/memplanner_cleanup_2026-07-02/PLAN.md`. Deferred:
Phase 2 gflat code/comment simplifications + Phase 3 V_q sizing consolidation (touch live code, need verify runs).

## 2026-06-17: VI3+CrI3 GW pipeline overnight — orbital-moment convergence delivered; σ_mnk blocked on head-OOM [D]

Big multi-stage push for VI3 & CrI3 monolayers (6×6 NSCF 600b → GW GN-PPM σ_mnk(ω) −10:10:0.25, 120-band Σ,
4000 centroids → bands → orbmag). lorrax_D moved to `main` (was `agent/install-blitz-integration` — preserved).
Two hard constraints: (1) full-machine maintenance 06-17 06:00→06-24, raced live on an interactive alloc + queued
batch backups (54624232 VI3, 54624233 CrI3); (2) the GW q=0 head can't be built on main — `psp.get_dipole_mtxels`
+ `bandstructure.htransform` load ALL wavefunctions to one GPU (155–231 GiB) → OOM for 6×6/80Ry; `epshead` needs
BGW eps0mat. **DELIVERED:** VI3 +U-gapped FM SCF (gap 0.11 eV) + NSCF + WFN.h5(41GB); **orbital-moment vs #bands**
for VI3 (spin 4.02 μB; m_orb 100b −0.284 → 600b −0.440 μB ∥spin, unconverged ∝N⁻¹·¹⁵, →lit. large-OM); CrI3 FM
80Ry SCF/NSCF/WFN + orbmag (running). **BLOCKED:** GW σ_mnk + DFT/GW bandstructures (head/dipole OOM — needs a
per-k-streaming dipole code fix, or vhead/whead overrides, or BGW eps0mat). Solved 9 pipeline blockers en route
(module names, pw2bgw+DFT+U `.hub1` davcio → strip `<dftU>` from .save XML, wfn2hdf abs-path, freed 16TB
`zeta_q*.h5`, centroid OOM→`--oversample 1.0`, cohsex inline-comment parse, kin_ion `--nb 120`, σ(ω) int64
overflow→`kij_stream`, npools-divides-ntasks). All in KNOWN_SANDBOX_ERRORS.md. Report:
`reports/vi3_cri3_gw_overnight_2026-06-17/report.md`.

## 2026-06-17: Screened bispinor χ/W through IBZ + Σ_xc Breit comparison; Hartree & SC-GN-PPM audits [C]

`sources/lorrax_C` `agent/bispinor-ibz-lorentz-unfold`. **The full screened bispinor χ/W workflow now
runs through the IBZ cascade** (the low-scaling path). Report: `reports/cri3_breit_sigma_xc_2026-06-17/`.

- **Screened-IBZ unfold crash FIXED** (`8605574`): screened supermatrix-W returns TT tiles replicated
  `P()`, but the R⊗R Lorentz-mix jit declares `in_shardings=P(None,None,None,'x','y')` → `ValueError`
  at `symmetry_maps.py:562` (only on `do_screened + _use_ibz_super`). Constrain `V_in` before the jit.
  Regression test reproduces the crash pre-fix, passes after. **Validated:** 16-GPU IBZ screened Σ^B is
  **bit-identical** to full-BZ-direct on FM CrI₃ (−6.796/−6.670/−6.438). (A 4-GPU run gave −9.4 via the
  band-sharding-at-small-mesh mode — CrI₃ MUST run 16 GPU / 4×4.)
- **Σ^B folded into static QP Σ_xc + screened/unscreened Breit comparison** (`c8c9a97`): static COHSEX
  `sigma_total` was dropping Σ^B (it only reached the QP via the dynamic/PPM `sig_x`). Now folded in;
  also evaluate Σ^B with BOTH screened W^{ij} and bare V^{ij} in one pass → `breit_comparison.dat`.
- **Σ_xc Breit result (FM CrI₃, converged 1800/600/200, k=0):** Breit contributes **−30..−36 meV** to Σ_xc
  on the band edges (more on deeper bands), slightly **widening the QP gap**; the gap correction is small &
  basis-sensitive (+12.8 meV @640/200 → **+3.2 meV @1800/600**, not fully converged — abs per-band Breit is
  the robust number). **Screened ≈ unscreened to <1 μeV** → transverse channel *effectively unscreened*
  (χ⁰≈0). Run `runs/CrI3/C_FM_breit_compare_1800_2026-06-17` (16 GPU, 600 s).
- **Bispinor transverse-centroid ψ restart** (`4523686`): round-trips the σ^B ψ; restart skips the ζ-fit,
  bit-identical Σ^B. Per-band Σ^B diagnostic (`4bc0d46`): the eV `tr Σ` is a SUM over all k·band; per-band
  Breit is meV (α²-suppressed vs the ~50 eV exchange) — not a regression.
- **Hartree (Q1):** `cohsex_sigma.hartree` is **charge-only** (ρ=J⁰, no γ̃ⁱ). The current–current
  (J^{1,2,3}) magnetic Hartree is **absent** — correct for zero-current systems, an α²-order omission for
  magnetic ones (no DFT counterpart; Milestone-C).
- **SC bispinor GN-PPM (Q2):** the SC/QSGW dispatch `sigma_dispatch.compute_sigma_xc` takes no transverse
  params and never builds Σ^B, and there is **no config guard** → SC bispinor GN-PPM *runs but silently
  drops Breit*. One-shot GN-PPM carries it (via `sig_x`). SC path needs the transverse plumbing or a gate.
- **Infra:** logged in KNOWN_SANDBOX_ERRORS — the venv editable `.pth` pins `lorrax` to **lorrax_B/src**, so
  bare `pytest` from lorrax_C silently tests lorrax_B (also the root cause of the `gw.w_bispinor` collection
  failure); fix = `PYTHONPATH=<checkout>/src`.

## 2026-06-17: B_xc implemented + CrI3 orbital-mag converged; residual audit exhaustive [B]

Noncollinear **B_xc (xc magnetic field)** now implemented in LORRAX's standalone
V_scf/H reconstruction (branch `agent/bxc-vscf-magnetic`, commits eba8609+c5ec2f0):
spin-polarized PBE (`psp/xc.py` `compute_V_xc_spin`, verified term-by-term vs QE),
`build_magnetization_from_wfn` + signed-magnetization (segni) spin split
(`scf_potential.py`), and `(B·σ)ψ` in `apply_H_k`/`apply_H_k_from_G`/`build_matrix_k`
with `HamiltonianK.B_vec` (`dft_operators.py`). Non-magnetic path bit-identical.
CrI3 ⟨v|H|v⟩−ε: **1.4 eV → ~19 meV** (deep Cr-3s 2 meV); MoS2/Si stay 0.2 meV.

**Residual audit:** every H component checked against QE and matches — density
(0.03%), spin-PBE functional, V_NL SOC D-matrix (VKB deeq_nc, exact), V_loc incl.
2D cutoff (pp.x V_bare, ≤7 meV projected), 2D Coulomb (MoS2 control, same
`assume_isolated='2D'`), spinor frame (native-wfc, self-consistent/frame-invariant),
E_nk (pw2bgw preserves exactly). The remaining ~19 meV is confined to the m≠0-only
finite-ζ spin V_xc on the Cr semicore (vanishes at ζ=0, not directly QE-diffable
since pw2bgw refuses VXC for nspin=4); does not move the orbital moment.

**Orbital moment of monolayer CrI3 (FM): antiparallel, m_z ≈ −0.078 μ_B/cell**
(6×6, 4000-band SOS) — Hund's 3rd rule (Cr³⁺ 3d³<½), exp −0.067. Convergence:
band count is the bottleneck (slow SOS tail crosses zero → −0.078; +0.026 at 180 b);
k-grid flat (+0.026/0.023/0.024 at 6×6/8×8/10×10). Plot
`reports/B_orbital_magnetization_cri3_2026-06-16/cri3_orbmag_convergence.png`.
**Orbital-mag-resolved Γ-M-K-Γ bandstructure** `cri3_orbmag_bandstructure.png`: per-(n,k)
m_z colored red(∥spin)/blue(anti-∥) on a real QE NSCF k-path (121 k, 120 b); SOS formula
validated to 3e-12 vs the 10×10 BZ total; orbital weight in the upper Cr-3d/I-5p valence
manifold. (Workflow built the path WFN; OrbMag/Plot redone manually after API-529 killed
those phases.) `compute_orbmag_bandpath.py` + `plot_orbmag_bandstructure.py`.
**+ CrI3 spin-resolved + VI3 (orbital-mag & spin) DFT bandstructures** (`d257166b`): generic
`plot_band_colored.py` (spin=⟨σ_z⟩ | orb=m_z). VI3 new run `05_bandpath_orbmag`: 80Ry/121k
band-path NSCF off the gap-recipe insulating d² basin → 27.7GB WFN; pw2bgw needed the
**dftU-strip** workaround (lda_plus_u=T aborts on `.hub1`). Sanity occ-summed ⟨σ_z⟩=+4.02 μB
= V³⁺ spin moment; orbital moment −0.31 μB antiparallel. Finished 39s before the 06-17
maintenance outage. **All DFT — GW bandstructures still TODO** (htransform GW-eigenvalue path
exists; GW *colored by orbital-mag* needs centroid→G-space code that doesn't exist).

## 2026-06-17: VI3 monolayer FM band gap OPENED — occupation-matrix bistability, not U/k/cutoff [D]

Resolved why noncollinear PBE+SOC+U VI3 monolayer came out METALLIC (see prior entry) despite being
a known FM semiconductor. A research workflow (3 web-research agents -> synthesis; cites Yang PRB101
100402, Sandratskii&Carva PRB103 214451, Hovancik NanoLett 2023) + experiment confirmed: **DFT+U
occupation-matrix BISTABILITY**. Plain +U self-consistently lands in the metallic equally-occupied-t2g
basin; raising U/ecut/k-points does NOT escape it. **FIX (run `runs/VI3/03_gap_recipe_80Ry_6x6_2026-06-16/`):**
seed the V-3d occupation to the d2 insulating pattern via `starting_ns_eigenvalue` (2 occ/3 empty,
majority spinor) + `Hubbard_occ=2.0` + PIN it the whole SCF (`mixing_fixed_ns=250` — both plain and
local-TF sloshed badly on release: a single static SCF ran 188 electronic iters without converging) +
U=5.0 (ortho-atomic Dudarev). **RESULT: FM INSULATOR, gap = +0.106 eV** (HOMO -5.466 / LUMO -5.360),
converged 15 iter, m_z = 2 uB/V (d2 S=1), clean d2 occ 0.989/0.991. Trail: no-seed -0.17 eV metal ->
U4+seed -0.022 eV (touching) -> U5+seed +0.106 eV insulator. CAVEAT: 0.11 eV < lit ~0.5 eV; Dudarev-U
slope ~0.13 eV/eV-U (reaching 0.5 would need unphysical U~8). Lit 0.5 eV needs Hund's J (Liechtenstein
U=4/J=0.9) splitting the SOC e' doublet; QE noncollinear kind=1+J support is AMBIGUOUS (source shows
only kind=0/kind=2 with noncolin) -- untested. Also a pinned-ns (constrained) gap = upper bound.
Reference clone: /pscratch/sd/j/jackm/qe_cri3/scf.in.

## 2026-06-17: Bispinor transverse-centroid ψ restart — Σ^B works on restart [C]

`sources/lorrax_C` @ `4523686` (branch `agent/bispinor-ibz-lorentz-unfold`). Clean mirror of the
existing charge restart for the σ^B side. Before this, a bispinor restart left `wfns_transverse=None`
(hard-coded "not-yet-supported") so Σ^B couldn't run on restart — forcing a full ζ re-fit (the
expensive transverse fit). The charge V_q/ψ and the bispinor V_q tiles already round-tripped (latter
via the deterministic `v_q_bispinor.h5` path); the only missing piece was the transverse-centroid ψ.
Fix: write `wfns_transverse.psi_yr` as a `psi_full_y_transverse` dataset alongside the charge
`psi_full_y`, read it back with the same `y3_psi_Y`/`x1_psi_X` sharding, and rebuild the bundle on
restart via `build_wavefunction_bundle`. The transverse-`Meta` sizing is factored into
`_transverse_meta(cfg, meta)` so the ζ-fit and restart paths size the bundle identically; old restart
files (no transverse dataset) degrade loudly (warn + skip Σ^B). **Round-trip (CrI3 6×6 30Ry bispinor
x_only, 300/102 cent, 4 GPU):** restart SKIPS the ζ-fit (25→0 fit markers, 898→69 log lines) and
reproduces all 9 Σ^B tiles **bit-identically** (max|diff| 0.0 eV). 19 transverse/helper/sigma_x
bispinor pytest pass. Run dir `runs/CrI3/C_restart_test_2026-06-17/`. Files: `tagged_arrays.py`
(+23), `gw_init.py` (+net 40). Not pushed.

## 2026-06-16: PROD magnetic bispinor GW — FM CrI3, Milestone-B screened-W no-NaN; screened-IBZ unfold bug found [C]

First production-scale bispinor GW on a *genuinely magnetic* (FM) CrI3 monolayer (6×6, 30 Ry, SOC;
mag 6.01 μB/cell ‖z; indirect gap **1.504 eV**), `sources/lorrax_C` @ `9128728` (transverse-CCT
covariance fix), 16×A100-80GB, 4×4 mesh. Run dir `runs/CrI3/C_FM_prod_bispinor_2026-06-16/` (641
charge + 200 current orbit centroids seed 42; nval=8/ncond=8/nband=180; `cusolvermp_lu=off`,
`LORRAX_RCOND_INDEF=1e-5`). **(1) Transverse Σ^B on REAL magnetism is substantial + covariant**:
diagonal xx=−6.80, yy=−6.67, zz=−6.44 eV, in-plane anisotropy **1.88%** (≈isotropic), xy=yx=−0.143
(**no sign flip**) — the fix's signature, NOT the 23%-anisotropic/sign-flipped pre-fix pattern.
**(2) Milestone-B screened-W runs without NaN** on the gapped FM system: minimax R=71.4, fit_err
2.9e-6, 0 NaN; δ−Vχ charge inversion completed (chi0_W 4.07s); sigCOH **nonzero** (~−9.7..−10.4 eV;
0 in x_only). DFT gap re-verified 1.5042 eV. **CRITICAL caveat:** the FM base WFN was NSCF'd
`nosym=.true.` (it's the orbital-mag SOS WFN) → **ntran=1 / nrk=36**, so the IBZ cascade is a NO-OP
(q-IBZ 36/36, disk shrink 1.0×). IBZ (run1) and full-BZ-direct (run2) are **bit-identical**; the
C3-orbit covariance gate is **DEGENERATE** (orbits are singletons → gauge S(q) spread 0.0, IBZ-vs-
fullBZ 0.0 *trivially*, not a tested pass). The covariance fix IS exercised (per-q indef solve), but
the *symmetry unfold* can't be tested on this WFN — needs a **symmetric** FM WFN (ntran>1). **(3)
REAL BUG:** the *screened* bispinor IBZ→full-BZ unfold (`unfold_bispinor_tiles` →
`unfold_v_q_bispinor_lorentz`, symmetry_maps.py:697, from gw_jax.py:343) crashes with a JAX
sharding-spec mismatch — TT tiles `complex128[3,3,36,208,208]` arrive replicated `PartitionSpec()`
but the Lorentz-mix pjit wants `(None,None,None,'x','y')`. Only on `_use_ibz_super AND do_screened`
(previously unexercised). NOT NaN/conditioning — pure plumbing. Bypassed with `LORRAX_FORCE_FULL_BZ=1`
(run4, the complete physics for ntran=1). Peak 58.18 GB/dev → **requires 80 GB nodes** (40 GB OOMs).
Infra: backgrounded `salloc --constraint=gpu&hbm80g -q interactive` revoked twice with "Connection
timed out" before nodes booted; fixed with `SALLOC_WAIT_ALL_NODES=1` + `SLURM_MSG_TIMEOUT=120`
(logged in KNOWN_SANDBOX_ERRORS.md). No source edited.

## 2026-06-16: VI3 monolayer noncollinear PBE+SOC+U=3.5 SCF — set up & converged [D]

New system. VI3 is isostructural with CrI3 (honeycomb V3+ d2 S=1, edge-sharing VI6 octahedra,
out-of-plane Ising FM). Built `runs/VI3/00_monolayer_pbe_soc_u_2026-06-16/` by adapting the proven
CrI3 FM monolayer input (`B_orbmag_FM_6x6_30Ry`): swap Cr->V (V.upf, assets/.../standard, FR-ONCVPSP
PBE z_val 13), a=6.84 A (exp. bulk R-3) with CrI3-isostructural internal coords (NOT relaxed),
c=18 A, ecutwfc=50, 3x3x1, noncolin+lspinorb, starting_magnetization +z, assume_isolated='2D',
`HUBBARD ortho-atomic / U V-3d 3.5`. **SCF converged 43 iter / 64s on 4x A100** (-npools 4):
E=-450.2721 Ry; net m_z=4.40 uB/cell, |m|=6.91; per-V m_z=2.80 uB (these are SPIN-only; pw.x
magnetization excludes the orbital moment L -- use psp/orbital_magnetization.py for L);
V-3d Tr[ns]=3.76. **CAVEAT:** highest-occ (-5.79 eV) sits ABOVE
lowest-unocc (-5.87 eV) -> near-metallic at this coarse/unrelaxed/50Ry level (lit. ~0.3-0.5 eV FM
semiconductor). Next: relax internal coords (VI3 V-I ~2.80 A vs CrI3 ~2.73), converge ecut(I wants
>=60-80)/k-grid, recheck gap (U+SOC has quenched-vs-large-orbital solutions; may need
starting_ns_eigenvalue). Report: `reports/vi3_monolayer_setup_2026-06-16/report.md`.
Infra notes: V.upf has the 3D PP_PSWFC needed for Hubbard; ortho-atomic is valid for noncollinear
k-point runs (only gamma-only force/stress & pseudo/wf/norm-atomic k-projectors are blocked in QE
src); the GPU `espresso/7.5-libxc-7.0.0-gpu` module only applies (PrgEnv-gnu->nvidia swap) under
`bash -lc` in the non-interactive agent shell.

## 2026-06-16: bispinor transverse Σ^B C3-covariance — FIXED at the indefinite-CCT solve [C]

Resolves the transverse Breit V_q C3-covariance residual from the sweep below. **The R⊗R IBZ
unfold was never the bug** — proven by a gauge-invariant test `S(q)=Σ_ij‖V^{ij}_TT(q)‖²_F`
(invariant under the per-q ζ̃ basis unitary AND the channel rotation R): IBZ-unfold tiles are
S-orbit-constant to 1e-7, and the loaded ψ / transverse current / γ̃ algebra are covariant to
machine precision. **Root cause: the per-q transverse-CCT solve** (`_ridge_indef_solve`,
`common/isdf_fitting.py`). The transverse CCT is Hermitian-INDEFINITE; its TRS-paired in-plane
near-null current modes (|λ|~1e-7·σ_max, cond≈3e7) sit far above the old fixed ridge
(LU_RIDGE=1e-12·tr/n), so LU inverted them as 1/λ and amplified sub-covariance-floor noise →
q-inconsistent in-plane ζ̃ → non-covariant tiles. **A +ridge can't regularize an indefinite matrix**
(a larger ε shifts a small −λ through zero → Σ^B_yy blew up to −912 eV at ε=1e-4). **Fix
(`9128728`):** relative-|λ| truncated-eigendecomposition pseudoinverse — drop
`|λ|<RCOND_INDEF·|λ|max` (default 1e-5, env `LORRAX_RCOND_INDEF`); correct for indefinite,
covariance-preserving (eigvecs at Sq are sym-images of those at q). Transverse `auto` now routes to
this in-tree path; the cuSolverMp getrf+getrs branch keeps the indefinite-incompatible +ridge →
opt-in only until ported to the PSD `(LᴴL+δ²I)` Cholesky-Tikhonov form (transverse is ~3× smaller
than charge so per-q eigh is cheap; charge keeps cuSolverMp Cholesky). **Validation (CrI3 6×6 30Ry,
300/102 cent, x_only): IBZ-vs-fullBZ Σ^B in-plane gap 23%→~3%; xy sign-flip GONE; both isotropic;
z exact; gauge-inv S(q) in-plane orbit-spread 6–18%→0.3–1.5%. eigh==exact-LU to 5e-15; 21 bispinor
pytest pass.** Residual ~3% = the CCT's ~1e-4 covariance floor (transverse signal overlaps the
near-null modes; needs better ψ-sampling, not a solve fix; `rcond=1e-4` over-truncates). Runs
`runs/CrI3/C_cri3_{fix_validate,ibz_fix}_2026-06-16`; reports `reports/bispinor_tt_conditioning_2026-06-16/`.
Pre-existing: `tests/test_w_bispinor_supermatrix.py` fails collection (`gw.w_bispinor` import,
unrelated to this fix). Knob: `RCOND_INDEF` trades covariance vs magnitude.

## 2026-06-16: bispinor transverse V_q — CLEAN C3-covariance sweep confirms a REAL residual [C]

Redid the muddied transverse-centroid convergence sweep of the bispinor in-plane (x,y) Breit V_q
C3-covariance violation with **rigorously identical methodology** (the prior 102/206/308/410 sweep
was contaminated — its 410 leg needed a per-run `memory_per_device_gb` 36→26 tweak). All counts:
kmeans `LORRAX_NGPU=1 --orbit --density-mode current --seed 42` oversample 1.5; gw full-BZ-direct
(`LORRAX_FORCE_FULL_BZ=1`) x_only, 4×A100-40GB, **budget 36 GB**, charge fixed at the 300 set.
Every run verified identical (Devices 4, full BZ 36 q, peak 34.92 GB — set by the fixed charge FFT,
N-transverse-independent). New runs `runs/CrI3/C_cri3_sweep_t{150,210,264,318}_2026-06-16/`
(orbit-closed to 152/212/266/320). **Verdict: the in-plane covariance violation does NOT shrink
toward 0** — it is erratic in a 0.07–0.33 band (102→0.186, 152→0.072, 212→0.332, 266→0.255,
320→0.086), no convergence/undersampling signature; charge-tr & z-z exact (~1e-8) throughout.
Meanwhile median|TT_22| inflates monotonically ~300× (8.9e2→2.65e5). This **reproduces the prior
sweep cleanly** — the inflation/non-covariance is a **real transverse-ζ̃ conditioning residual**
(indefinite CCT), not basis undersampling. The domain-expert's "well-conditioned, shrinks with
centroids" expectation is not borne out. No source edited (diagnosis only). Report appended:
`reports/bispinor_tt_conditioning_2026-06-16/report.md` ("CLEAN RE-SWEEP" section).

## 2026-06-16: orbital-mag — ROOT CAUSE of CrI₃ H-reconstruction bug: spin-blind V_xc [B]

Pinned the CrI₃-specific `⟨v|H|v⟩−ε_v` ≈ 1.4 eV residual (vs MoS₂ 0.2 meV) that was
blocking the band-sum-free Sternheimer orbital-mag. **Root cause: LORRAX's standalone
V_scf reconstruction (`dft_operators.compute_V_H_and_V_xc:317-323`) builds V_xc from
the CHARGE density only via spin-unpolarized `pbe_functional()`, omitting the xc
magnetic field B_xc(r).** Exact for non-magnetic systems (MoS₂, Si → 0.2 meV / 0 mRy);
wrong by ~the exchange splitting (~eV, peaked at the magnetic ions) for ferromagnetic
CrI₃ (6 μ_B). Confirmed by a full elimination chain (all *ruled out*): diagnostic
unfaithful (MoS₂ gate=0.21 meV ✓), NLCC missing (off makes it worse), radial-table
n_q (flat 4k→32k), Hankel FT accuracy (8× refine Δ≤6e-4), dense-grid/ecutrho (|G|max
24 bohr⁻¹), and ρ_val reconstruction (matches QE `charge-density.hdf5` to 0.03%) — then
`diag_xc_field.py` showed the omitted |V_x↑−V_x↓| is ~6 eV peak / ~4 eV mean over the
Cr atoms, matching the band residual. Scripts in `reports/B_orbital_magnetization_cri3_2026-06-16/diag_{nlcc_toggle,nq_sweep,hankel_radial,qe_density,xc_field}.py`.

**Scope:** affects only paths that *rebuild and apply/invert* the KS H on a **magnetic**
system — the Sternheimer covariant-derivative resolvent (→ Sternheimer orbital-mag
magnitude) and any **rebuilt-V_scf GW screening on magnetic systems (bispinor-CrI₃)**.
*Not* affected: the SOS orbital-mag (velocity-only, no V_scf), the velocity/dipole
operator (kinetic+nonlocal), and the **core `gw.gw_jax` GW driver** (uses WFN ε/ψ
directly, never rebuilds V_scf). Fix = noncollinear V_xc + B_xc·σ̂ + 2×2 spin-dependent
`apply_H` (real feature, scoped in report "Next steps").

**Un-retraction:** the prior entry's "antiparallel is likely a truncation artifact" is
**overturned** — the band-sum-free Sternheimer (no empty-band sum) *also* gives
antiparallel (−0.063 μ_B), agreeing with the SOS extrapolation (~−0.08) and experiment
(−0.067). The antiparallel sign (Hund's 3rd rule, Cr³⁺ 3d³ < ½-filled) is robust and
V_scf-independent; only the Sternheimer *magnitude* is pending the spin-V_xc fix.

## 2026-06-16: orbital-mag — LITERATURE CHECK: direct SOS is the wrong route [B]

Deep-research literature check (cited, `reports/B_orbital_magnetization_cri3_2026-06-16/
RESEARCH_NOTES_band_convergence.md`) on the band convergence. Findings: (1) the slow
`band^−1.15` tail is **real & structural** — the local-circulation ("H−ε") term has a
SINGLE energy denominator `Σ_m v·v/(ε_n−ε_m)` vs the squared denominator of the
Berry-curvature term (Xiao-Chang-Niu RMP 2010; Souza-Vanderbilt). Not a bug. (2) BUT
the **direct SOS is documented as impractical** for first-principles (CTVR PRB 74,024408:
only OK for tight-binding/few bands); the empty-band sum is provably removable via Q=1−P
→ occupied-states-only. (3) **Correct method = band-sum-free covariant finite-difference**
of occupied Bloch states (Lopez-Vanderbilt-Thonhauser-Souza PRB 85,014435; CTVR App.A),
or Wannier90 berry / QE-CONVERSE — no empty-band sum. (4) Published DFT+SOC for monolayer
CrI₃ = **+0.099 μ_B/Cr PARALLEL** to spin (Ovesen-Olsen arXiv:2405.04239, GPAW), exp
−0.067 antiparallel (PBE gets sign wrong). **⇒ our SOS antiparallel −0.08/cell is most
likely a truncation artifact, NOT physical**; neither +0.024 (180 b) nor −0.08 (4000 b)
is trustworthy. The earlier "Hund's-rule antiparallel" reading is retracted. Fix:
implement covariant-FD orbital mag (reuses WFN reader + SymMaps k-neighbors).

## 2026-06-16: orbital-mag — band convergence to 4000: ANTIPARALLEL ~0.1 μ_B (Hund) [B]

Mapped CrI₃ orbital-moment band convergence to **4000 bands** (6×6 IBZ, 8 k;
`nscf_6x6sym_{2000,4000}`, NSCF ~80 s / ~3 min on 8 GPUs). m_z(∥spin) vs SOS band
ceiling: 180→+0.026, 400→+0.006, 800→−0.028, 2000→−0.061, **4000→−0.078**, slope
still shallowing. Per-band increment ~**band^−1.15** (full AND kinetic-only →
intrinsic local-circulation slow convergence, the single-energy-denominator term;
NOT a nonlocal bug). Marginally convergent (partial sums ~N^−0.15) → even 4000
bands isn't fully converged; limit ≈ **−0.08 to −0.12 μ_B**. **Verdict: |m_orb| ≈
0.08–0.10 μ_B, ANTIPARALLEL to spin** — magnitude matches the ~0.1 expectation,
sign is Hund's-third-rule (Cr³⁺ 3d³, <half-filled). The +0.024 "plateau" at 180
bands was a truncation artifact. Direct dH/dk SOS is correct but band-pathological
for orbital mag; a converged number needs a sum-free method (DFPT/Sternheimer or
Wannier covariant derivatives). Plots: `band_convergence_6x6_{2000,4000}.png`.

## 2026-06-16: orbital-mag — 400-band convergence: NOT band-converged [B]

Pushed the SOS band count to 400 (6×6 IBZ, `nscf_6x6sym_400`, 8 IBZ k). Added
`colA_z/colB_z` to the `--out` npz (band-resolved z-columns → cumsum = m_z vs band
ceiling, plotted by `plot_band_convergence_400.py`). **Finding:** m_z(∥spin)
humps to ≈+0.025 near N≈180 then DECREASES MONOTONICALLY — 180→+0.0256,
240→+0.0204, 300→+0.0144, 360→+0.0093, **400→+0.0061** (still dropping). So the
earlier "+0.024 plateau" was a 180-band truncation artifact; the direct
sum-over-states (k·p) orbital sum is very slowly band-convergent (the reason
Wannier interpolation / LVTS12 exists). PBE orbital moment is small and needs
≫400 bands (or Wannier interp) to pin. Plot: `band_convergence_6x6_400.png`.

## 2026-06-16: orbital-mag — k-convergence (8×8/10×10) + symmetry-reduced IBZ mode [B]

Extended `psp/orbital_magnetization.py` (branch `agent/orbital-magnetization`):
(1) **vectorized reductions** (μ-linear `PA/PB`; mu-scan/per-band/convergence now
free, numerically identical — verified). (2) **`--ibz` symmetry-reduced mode**:
loops the stored IBZ k-points (G-flat, no ψ unfold) and symmetrizes the
**axial-vector** orbital-moment density over the **magnetic point group**,
`M = Σ_i w_i (1/|G|) Σ_g det(R_g) R_g m(k_i)` (det factor for improper ops; T and
M_z-flipping σ_v/C₂ excluded). Built via an ultracode workflow (6 agents); both
adversarial verdicts confirmed (axial transform, det factor, no w_i·1/|G|
double-count, frame consistency).

**CrI₃ FM convergence** (`runs/CrI3/B_orbmag_FM_conv_2026-06-16/`, one SCF reused;
orbital ∥ spin, μ_B, midgap): 6×6 = +0.0255, 8×8 = +0.0227, 10×10 = +0.0235,
12×12 = +0.0231, 14×14 = +0.0228, **16×16 = +0.0234** (IBZ 8/12/18/26/34/44 k) —
**k-converged at ≈ +0.023 μ_B parallel to spin** (flat 8×8→16×16; does NOT climb
to ~0.1 with k; the under-convergence is band-count — the SOS sweep oscillates).
**IBZ validated EXACT**: 8×8 IBZ (12 k) = +0.02272 = full-BZ (64 k); 10×10 IBZ
(18 k) = +0.02353 = full-BZ (100 k) to all digits (same SCF ⇒ exact unfold);
6×6 IBZ (8 k)=+0.0256 vs full (36 k)=+0.0255. |G|=6 (S₆), m_x=m_y=0. ~5× fewer
k-points at identical physics. Magnetic sym-reduced NSCF works via `automatic`
grid (the earlier `irrek_nc` crash was the explicit-k-list path).
Report: `reports/B_orbital_magnetization_cri3_2026-06-16/report.md`.

## 2026-06-16: Milestone-A bispinor screened W via channel-blocked supermatrix [C]

New `gw/w_bispinor.py`: assemble the 4×4-Lorentz-block `(nq,N,N)` supermatrix (N=n_C+3·n_T) from
the bispinor V/χ tiles, invert via the existing `w_isdf.solve_w` (per-q LU, reused), extract W
tiles. **Milestone A = χ only in the (0,0) charge block** → collapses to W⁰⁰=scalar screened,
W^{ij}=bare, W^{0i}=0 (unit test `test_w_bispinor_supermatrix.py`, 2 passed). **Wired into the
driver** (`gw_jax`/`cohsex_sigma`/`sigma_x_bispinor` take `w_ij_tiles`; the standalone charge
`solve_w(V⁰⁰)` is gone for bispinor): W⁰⁰→Σ_SX/Σ_COH, W^{ij}→Σ^B. **E2E (MoS2 3×3): sigma_diag
BIT-IDENTICAL to analytic-A** (`runs/MoS2/C_60Ry_bispinor_supermatrixA_2026-06-16/`); pytest 16
passed. Full-BZ only for now (LORRAX_FORCE_FULL_BZ=1; IBZ per-channel unfold is the upgrade).
Branch `lorrax_C agent/bispinor-supermatrix-w` cad7378 (not pushed). **Milestone B** = un-trace χ
(`w_isdf.py:181`) so the transverse χ tiles are nonzero — assembly/solve/extract unchanged.
Report: `reports/bispinor_supermatrix_a_2026-06-16/`.

## 2026-06-16: NEW orbital-magnetization tool (modern theory, explicit dH/dk) + FM CrI₃ [B]

Built `psp/orbital_magnetization.py` + `psp/orbital_magnetization_THEORY.md` (branch
`agent/orbital-magnetization`): per-cell orbital magnetic moment of a SOC spinor crystal via the
modern-theory **sum-over-states** formula, `m_z/μ_B = (−½) Σ_k w_k Im Σ_{n occ} Σ_{m≠n} ε_zab
v^a_nm v^b_mn (ε_m+ε_n−2μ)/(ε_n−ε_m)²`, reusing the psp velocity machinery
(`velocity_matrix_k = 2(k+G) + dV_NL/dk`, **analytic dV_NL/dk, no finite differences**). Prefactor
−½ in Ry AU verified 3 ways + adversarially. Built via an `ultracode` workflow (11 agents:
deep code-read + independent theory derivations + adversarial verify).

**KEY FINDING:** every existing sandbox CrI₃ WFN is **non-magnetic** — `noncolin/lspinorb=.true.`
but **no `starting_magnetization`** → converges to the TRS (Kramers-paired, zero-net-spin) state,
for which spin and orbital magnetization are *identically zero* (tool correctly returns 0).
Generated a **ferromagnetic** variant `runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/` (SCF total mag
(0,0,**6.01**) μ_B, gap **+1.50 eV**). Result (6×6, 180 bands, midgap): spin **6.01 μ_B** ✓,
m_x=m_y≈0 ✓, **orbital m_z = +0.026 μ_B, parallel to spin** — sign matches the expected +0.1 μ_B;
magnitude under-converged (band-ceiling sweep oscillates; μ-scan +0.032→+0.019 across the gap;
6×6 is coarse). Nonlocal velocity sign resolved definitively: `compute_vnl_velocity_cart =
+dV_NL/dk` (off-diagonal FD, ratio 1.000) → physical velocity `p+vNL` (the dipole driver's `p−vNL`
is a BGW convention; it would flip the sign to −0.081). Report:
`reports/B_orbital_magnetization_cri3_2026-06-16/report.md`. Also fixed `lxrun` (missing `--overlap`,
see KNOWN_SANDBOX_ERRORS).

## 2026-06-16: FIX bispinor IBZ-cascade zeta crash + sym==nosym validated [C]

The IBZ-cascade bispinor zeta crash (`B.shape 9!=5`, see milestone-A report) was an **ordering
bug**, not a wrong-unfold bug: `fit_zeta_to_h5` sliced `C_q`/`L_q` to IBZ before the orbit-closure
auto-fallback (which flips `write_ibz_only=False`) ran — so on a non-closed centroid set the charge
channel left `L_q` at IBZ while `Z_q` reverted to full-BZ. **Fixed** by finalizing `write_ibz_only`
before the slice (`lorrax_C agent/bispinor-ibz-zeta-fallback-fix` `fc9984e`, +48/−49 in
`isdf_fitting.py`). `LORRAX_FORCE_FULL_BZ` is no longer needed.

**Validated** (`reports/bispinor_ibz_zeta_fallback_fix_2026-06-16/`): pytest 21 passed; fixed code
with no FORCE completes + bit-identical to forced full-BZ; and with orbit-closed centroids (charge
**641** regenerated orbit-aware + transverse 668) the **full bispinor IBZ cascade (all 4 channels,
5 IBZ q / 9 full-BZ) is BIT-IDENTICAL to the same-basis full-BZ run** → the IBZ→full-BZ unfold (ζ̃
and V_q, charge + transverse) is numerically exact. Data finding: the original `centroids_frac_640`
was not orbit-closed (z-mirror partners absent); regen orbit-aware to get the IBZ speedup. Runs:
`runs/MoS2/C_60Ry_bispinor_{ibztest,fullibz,fullbz641}_2026-06-16/`.

## 2026-06-16: Milestone A — screened-charge bispinor COHSEX (+ bare Breit) validated [C]

Rebased `lorrax_C` to `main` `e85be60`. Mapped the bispinor GW pipeline
(`reports/bispinor_screened_gw_state_2026-06-15/`): today's bispinor Σ is **bare DHF + bare-Breit
exchange only**; the path to full screened bispinor W is to invert `(δ−Vχ)` in the channel μ-basis.
**Milestone A** (screened charge channel + bare Breit; the χ_μν = χ⁰⁰-only reduction →
W⁰⁰=scalar screened, W^{ij}=bare, W^{0i}=0) turned out to be **already wired** in the one-shot
driver (`gw_jax.py:359` `compute_cohsex_sigma(do_screened=True, …, bispinor args)`), so A reduced
to a validation.

**Validated** (MoS₂ 3×3, nband=32, full-BZ, 64 s, 1 node;
`runs/MoS2/C_60Ry_bispinor_cohsex_2026-06-15/`,
`reports/bispinor_screened_a_validation_2026-06-16/`): screened bispinor COHSEX runs end-to-end;
screening reduces exchange (bare Σ_X −37.09 → screened Σ_SX −22.14 eV, n=0); bare-Breit Σ^B
coexists (diag −0.153/−0.153/−0.144 eV, in-plane (1,1)=(2,2) per z-mirror, Hermitian, Σ_tot^B
≈−0.52 eV); Kramers degeneracy preserved; COHSEX gap 2.40 eV (reduced model). Charge-block
reduction (bispinor Σ_SX/Σ_COH == scalar) holds **by construction** (`cohsex_sigma.py:211-212`
untouched by the bispinor branch); empirical scalar cross-check pending (allocation occupied by a
concurrent 16-GPU pool job).

**Blocker found + worked around:** IBZ cascade crashes the bispinor zeta fit on current main
(`B.shape[0]=9 != Nq=5`, full-BZ Z_q vs IBZ L_q; parent ran pre-cascade so was immune).
`LORRAX_FORCE_FULL_BZ=1` workaround (identical physics). Logged in KNOWN_SANDBOX_ERRORS.

**Remaining for A:** wire Σ^B through `compute_sigma_xc`/`sc_iteration` (+ transverse U_qp
rotation) for the SC path; then B = the genuine supermatrix (un-trace χ → channel δ−Vχ → LU →
screened Σ_c^B).

## 2026-06-15: CrI3 6×6 80Ry GN-PPM max-bands-before-OOM — bottleneck + ceiling [coord]

First **CrI3 GN-PPM** run in the sandbox (all prior CrI3 runs were x-only / COHSEX).
Rebased an old-main checkout (`lorrax_A`, branch `agent/cri3-ppm-maxbands`) to current
main `e85be60` (9-commit JAX-0.9/CPU-MPI delta; GPU memory model byte-identical, TRS-fix
+ IBZ-cascade already in both). 800-band NSCF (8 IBZ k, ngkmax=59990) → WFN.h5; reused
80Ry SCF. Run dir `runs/CrI3/6x6_80Ry_gnppm_maxbands_2026-06-15/`.

**Memory bottleneck = `C_fit_one_rchunk`** (ISDF pair-density transient
`slots·16·nk·ns²·μ·r_chunk/p_xy`), NOT the V_q buffer. Validated live: the production
driver printed `HWM=56.00 GB/dev, bottleneck=C_fit_one_rchunk` at nb150, matching a
standalone `plan_gflat_chunks` sweep. Mechanism: planner shrinks r_chunk to pin HWM=56
until r_chunk floors at `max_chunks=64`; past that HWM climbs with μ → OOM.

**EMPIRICAL ceilings (16× A100-80GB, μ=10·nband, both set at the ζ-fit):**
- **Non-bispinor: 700–798 bands** — nb700 (μ6986) FITS (ζ-fit + IBZ V_q in ~24 min),
  nb798 (μ7980) OOMs at the ζ-fit (60.5 GB alloc). The planner (~640) is **conservative**:
  nb700 fits at planner HWM=113% of budget (it assumes full-BZ V_q but runtime is IBZ
  n_q=8, and over-charges Peak C).
- **Bispinor: 200–300 bands — ~3× LOWER** — nb200 (μ2000) clears the binding charge ζ-fit,
  nb300 (μ2992) OOMs at it (`fit_zeta`, 90.8 GB). The 4 ζ channels (charge + 3 transverse)
  give a ~3.8× steeper Peak C slope (in-driver r_chunk=19536/58ch vs non-bisp 73824/16ch at
  nb150). The in-driver planner tracks bispinor well (nb300@145% OOMs, nb200@97% fits); only
  the *standalone* `plan_gflat_chunks(is_bispinor=True)` doesn't — worth a follow-up.

Enabling the sweep required a source fix + a workaround: **`--prune-mem-gb` flag** added to
`kmeans_cli` (threads `memory_per_device_gb` to the prune's Gram FFT — fixes the
cuFFT-scratch OOM); large-μ centroids then generated with **`--oversample 1.0`** (skips the
prune's still-unchunked 173 GB candidate-Gram). Full QP completion (eqp0.dat) still blocked
by the dipole.h5 box OOM (downstream of the ζ-fit, so it does NOT affect the ceiling result);
fix is to k-chunk `get_dipole_mtxels` via its existing `k_range`. Sandbox issues logged to
KNOWN_SANDBOX_ERRORS.md. Report: `reports/cri3_gnppm_maxbands_2026-06-15/report.md`.
Allocations JID 54541850 (session 1) + 54544991 (session 2).

## 2026-05-20: full COHSEX (do_screened=true) validated on CPU MPI [coord]

End-to-end full-COHSEX (`x_only=false, do_screened=true, screening=cohsex`)
on CPU n=4 (Milan node) at Si μ=384 production config completes **on the
first attempt with no additional source patches required**. Tests Σ_SX +
Σ_COH (static screened-exchange + Coulomb-hole) on top of the bare Σ_X
that the prior x_only validation exercised.

| | wall | Σ at (k=0, band=1) | eqp0.dat vs GPU |
|---|---|---|---|
| CPU 4 ranks × 8 threads | ~50 s | −16.478 eV (was −8.915 for x_only — δ ≈ −7.56 eV is Σ_C) | byte-identical except timestamp |
| GPU 4×A100 hbm80g | comparable | identical to CPU | reference |

pytest on lorrax_B against modules we patched: **63 passed, 15 skipped,
0 failed**.

Branch ready for merge to origin/main. The five-commit branch tip is
89690f0; see lorrax_B `git log origin/main..agent/jax-09-cpu-compat`.

## 2026-05-20: `_to_host` refactored to metadata dispatch (lorrax_B 89690f0) [coord]

Subagent design review of the 4-case shape-switch `_to_host` introduced
in 565750a flagged it as accidental complexity: 3 of 4 branches were
defending against return shapes that the JAX 0.9 source provably cannot
produce for the inputs `_to_host` actually receives. Empirically
characterized the sharding inventory at every gather call site (debug
run, Si μ=384 x_only CPU n=4) — confirmed:

* Every NamedSharded `mesh_xy` array (gflat_acc, G0_all, V_qmunu, etc.)
  is non-fully-addressable → `process_allgather(tiled=True)` Path (B)
  returns shape exactly `A.shape`.
* The lone Path-(D) failure case (`enk_full`) is `SingleDeviceSharding`
  with `is_fully_replicated=True`, which can be short-circuited via
  `A.addressable_data(0)` without ever calling `process_allgather`.

Replaced the 50-line 4-case switch with a 30-line 2-case dispatch on
stable `jax.Array` metadata (`is_fully_replicated`). Folded the inline
gathers in `gw_init.py:1121` and `isdf_fitting.py:2685` into `_to_host`
calls; dropped their legacy `shape[0] == 1` post-process guards (dead
code under the current geometry — G0 is 2D, gflat_acc is 3D, the guards
checked for 5D/4D leftovers from an old V_q bispinor layout).

Net: −24 LOC, no dead branches, no shape arithmetic, dispatches on
documented public API. End-to-end Si μ=384 x_only CPU n=4 `eqp0.dat`
**byte-identical** to the 565750a baseline (timestamp only differs).
Max rank-0 RSS unchanged at 26.6 GB.

Design review report:
`reports/memory_model_nonbispinor_kgrid_2026-05-18/PROCESS_ALLGATHER_DESIGN_REVIEW_2026-05-20.md`

## 2026-05-20: CPU port + planner backend-aware pair_density_slots [coord]

End-to-end CPU MPI port of the GW driver on Si 4×4×4 μ=384 non-bispinor.
Three JAX-0.9 strictness fixes were needed to get past the multi-process
code path (lorrax_B branch `agent/jax-09-cpu-compat`, commit `c7e6695`):
`cholesky_2d.py` panel_init `lax.pcast(('x','y'), to='varying')`,
`_slab_io_allgather.py` + `isdf_fitting.py` `tiled=True`. Backend-agnostic
fixes — GPU back-compat verified byte-identical (HWM 20.12, peak 20.13,
−0.05% — same as 2026-05-19 reference).

Planner finding (lorrax_B `5c2dae7`): CPU XLA's BufferAssignment schedules
**4 concurrent pair-density slots** in `fit_one_rchunk` where GPU XLA
schedules 3. Per-slot bytes match the existing `_bytes_c128(nk, ns², mu,
r_chunk, /p_xy)` formula exactly on both backends; only the slot count
differs. The +30% RSS excess over HWM_pred on CPU is **exactly this one
extra slot**. New helper `_default_pair_density_slots()` in
`gflat_memory_model.py` resolves the value via `jax.default_backend()` at
function-call time.

HLO evidence — robust across:
* Scalar non-bispinor (`module_0342.jit_fn`): 4 × 5.70 GiB
* Bispinor charge (`module_0360.jit_fn`): 4 × 16.92 GiB
* Bispinor transverse (`module_0413.jit_fn`): 4 × 16.92 GiB
* FFT-scratch hypothesis tested + REJECTED: at band_chunk ∈ {32, 64, 120}
  slot count + per-slot bytes invariant; FFT-box shapes alias into slots
  but don't size them.

Post-fix predictor accuracy on CPU at Si μ=384:
* n=1: HWM_pred 73.92 vs RSS 71.89 → **+2.8%** over
* n=2: HWM_pred 52.39 vs RSS 53.06 → **−1.3%** under
* n=4: HWM_pred 26.24 vs RSS 26.64 → **−1.5%** under

(was −24% to −33% across the same configs before the fix.)

Profiling stack: `scripts/profiling/pf.py` (+40 LOC) gains a CPU-backend
branch — psutil RSS fallback when `device.memory_stats()` returns None,
peak_rss_bytes tracking, pre-import of `jax.profiler` to dodge a
JAX-0.9 lazy-import race that crashes the sampler on CPU. GPU path
unchanged. New `skills/profiling_stack/cpu_addendum.md` documents the CPU
launch recipe + what's empty + HLO naming conventions.

Reports:
* `reports/memory_model_nonbispinor_kgrid_2026-05-18/CPU_VALIDATION_2026-05-20.md`
  — initial CPU port (n=1/2/4 RSS vs planner)
* `reports/memory_model_nonbispinor_kgrid_2026-05-18/CPU_OVERHEAD_DECOMP_2026-05-20.md`
  — subagent decomposition of the +6.5 GB excess into HLO-evidenced
  contributors; identified the 4-slot finding
* `reports/memory_model_nonbispinor_kgrid_2026-05-18/CPU_PLANNER_LANDED_2026-05-20.md`
  — this session's FFT-test + bispinor confirmation + planner-landed note

Run dirs:
* `runs/Si/NONBISPINOR_CPU_2026-05-20/{mu384,mu384_decomp,mu384_fft_probe,mu384_bispinor}/`
* `runs/Si/NONBISPINOR_BUDGETSWEEP_2026-05-20/` — earlier today: planner
  budget-fill behavior at memory_per_device_gb ∈ {25, 35, 50, 70} GB at
  both μ=384 (single-chunk) and μ=1200 (multi-chunk). Picker fills 80%
  of budget when binding, sits at single-chunk floor when loose, never
  exceeds budget; mem_stats peak tracks HWM_pred within +1% across the
  sweep.
* `runs/Si/NONBISPINOR_PROD_2026-05-19/` — 2026-05-19 GPU production
  redo + this session's GPU back-compat smoke test.

Allocations released (CPU 54411765, GPU 54411976).

## 2026-05-19: Non-bispinor planner audit — production-config redo, planner is faithful [coord]

Re-did the 2026-05-18 non-bispinor audit at the production configuration
(`noncolin=.true., lspinorb=.true.`, FR-ONCVPSP PBE pseudo, `bispinor=false`,
cuSOLVERMp default-on, hbm80g + BFC+0.95) after the 2026-05-18 sweep was
flagged as scope-erroneous (Agent A built `nspinor=1`; agents disabled
cuSOLVERMp instead of using `hbm80g` per env docs).

Two μ values matching the bispinor sister sweep on JID 53207377:

| μ | r_chunk × n_chunks | HWM_pred | mem_stats peak | %-err |
|---|---|---|---|---|
| 384 | 13824 × 1 | 20.12 | 20.13 | **−0.05%** (bit-exact) |
| 1200 | 13468 × 2 | 55.99 | 55.74 | **+0.45%** (slightly conservative) |

Both inside (and on the optimistic side of) bispinor's [−0.5%, −10.8%] band.

**Falsifies the all_gather-slab planner refinement** Agent C proposed at the
scope-erroneous config: it would have shown up here as a measurable
under-prediction in either μ data point — neither does. **Stand down on
the planner edit.**

The 2026-05-18 scope-error addendum: `nspinor=1` is unsupported (production
always uses FR pseudo + noncolin=true → nspinor=2); `cusolverMp status=7`
under BFC+0.95 is hbm40g vs hbm80g, not a sandbox bug (documented in
`ENVIRONMENT_COMPREHENSIVE.md` §3.2 + §8.3). The two `nspinor=1` loader
"fixes" landed on `agent/si-nonbispinor-mu-sweep` (`8c18925`, `dc0b254`)
were reset out — branch back to `origin/main`. Same for the
`d4cb599` cherry-pick on `agent/si-band-sensitivity`. Misleading
KNOWN_SANDBOX_ERRORS.md entries WITHDRAWN with cross-refs to the real docs.

Report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/REDO_PROD_2026-05-19.md`.
Run dir: `runs/Si/NONBISPINOR_PROD_2026-05-19/` (qe/ symlinks to
`runs/Si/05_si_4x4x4_sym/qe/`).

## 2026-05-18: Memory-model non-bispinor + k-grid robustness — SYNTHESIS [coord]

Cross-cut synthesis of three parallel sub-agents (A: μ-sweep + HLO at scalar
ns=1; B: k-grid 2³→6³ scaling; C: nb=100 vs nb=200 sensitivity) on Si non-bispinor
against the bispinor-calibrated planner from `memory_model_refit_2026-05-17`.

**Headline.** Planner constants survive ns ∈ {1, 2, 4} bit-exact (A HLO),
per-term kgrid scaling matches analytic prediction within 5–6% on all 4 kgrids
(B), no leaks across r-chunks/bc-chunks/sym-channels/kgrids. Production-scale
non-bispinor configs (Si 4×4×4 nb=100 −0.8%, Si 6×6×6 nb=100 −6.1%, Si 4×4×4
nb=200 −3.2%) sit inside bispinor's [−0.5%, −10.8%] under-prediction window.

**Three biases identified, each with a quantitative mechanism:**
1. ~5–8 GB CUDA/JIT/NCCL framework floor dominates whenever the algorithmic
   peak is small (Si 2×2×2 → +96.5%, Si 3×3×3 → +52.9%, Si 4×4×4 μ=192 → +185%).
   Additive, NOT multiplicative — already user-deferred per §6.2.
2. Single-r-chunk degenerate configs over-predict by ~25% because the planner
   reserves 3 pair-density slots but only ~2 are concurrently live when
   n_chunks=1 (Si scalar μ≥768 in A).
3. **NEW**: Si 3×3×3 nb=200 breaks the bispinor window at −13.9%. Root cause
   identified — unmodeled `c128(nk, band_chunk, ns, r_chunk/p_y)` all_gather
   slab on `psi_l_X`/`psi_r_X` inside `z_q_from_psi_sm._local`. Documented in
   `docs/MEMORY_MODEL.md` §R-Chunk but absent from `_peak_C_fit_one_rchunk`.
   Same shape as bispinor Si μ=768 −10.8% and CrI3 80Ry −8.5% gaps.

**Highest-leverage open work (synthesis §5.1):** add `M_all_gather_slab` to
Peak C; Agent C estimates it lands the 3×3×3 nb=200 outlier at −5.8% and
likely cuts CrI3 80Ry from −8.5% to −5%. Needs HLO calibration of slab
coefficient first (1× vs 2×, with/without aliasing).

**Latent ns=1 bugs fixed in-branch.** Agent A: `unfold_psi` and
`WfnLoader._ensure_phdf5_static` both silently broadcast nspinor=1 → 2 via
2×2 spinor-rotation einsums. Commits `8c18925`, `dc0b254` on
`agent/si-nonbispinor-mu-sweep` (lorrax_A), 30 LOC, all 44 loader/unfold
tests pass at ns=2 — clean merge candidates for origin/main.

**New sandbox bug.** `cusolverMpPotrf` returns status=7 INTERNAL_ERROR under
BFC + PREALLOCATE=true + MEM_FRACTION=0.95 on a 2D mesh. Workaround:
`cusolvermp_charge=off, cusolvermp_lu=off` in cohsex.in. Logged at
`KNOWN_SANDBOX_ERRORS.md:117`.

**Bottleneck-flip risk.** B_CCT_chol hits 71% of binding Peak C at Si 6×6×6 μ=1348.
Will flip to bottleneck at larger μ or under bispinor 4-channel cascade. No
r-chunk knob to mitigate — remedy is smaller μ or larger mesh.

Synthesis: `reports/memory_model_nonbispinor_kgrid_2026-05-18/SYNTHESIS.md`.
Per-agent reports + JSON data + run dirs preserved under same dir + `runs/Si/{MU,KGRID,BANDS}_nonbispinor_2026-05-18/`.

## 2026-05-18: TRUE scalar (nspinor=1) Si non-bispinor μ-sweep + HLO calibration [agent-A]

Mirrored `agent_t_si_bispinor_sweep.md` at the **opposite** extreme — actually
scalar Si (`noncolin=false, lspinorb=false, nspinor=1, nspin=1`) at 4×4×4
25 Ry, 4 GPUs on hbm40g (2×2 mesh), μ ∈ {192, 408, 756, 1176, 1764}
(orbit-pruned counts from requested {192, 384, 768, 1200, 1800}).

**Two latent ns=1 loader bugs surfaced and fixed in-branch** (commits
`8c18925` `unfold_psi` eager path + `dc0b254` `WfnLoader._ensure_phdf5_static`
phdf5 path): both blindly built 2×2 spinor rotation matrices regardless of
the WFN's nspinor, causing silent einsum broadcasting from ns=1 input to
ns=2 output. Pre-existing `KNOWN_SANDBOX_ERRORS.md` 2026-05-18 entry
"scalar nspinor=1 Si WFN.h5 trips kmeans_cli" is now marked FIXED. All 44
loader/unfold tests still pass.

**HLO calibration at ns=1**: `pair_density_slots = 3` and `fft_box_factor_D = 2.0`
both confirmed bit-exact at μ=408 + μ=1176, per-slot bytes match
`_bytes_c128(nk=64, 1, 1, mu_padded, r_chunk=13824, shard=p_xy=4)` exactly.
The invariant 3-slot count now holds across ns=1 (this work), ns=2 (M4),
ns=4 bispinor (M1) — `pair_density_slots` is a structural constant of
`fit_one_rchunk`'s scan-INSIDE-shard_map, NOT an ns-dependent count.

**Planner faithfulness pattern is structurally different from bispinor**:
* μ=192:  HWM_pred=2.81  vs mem_stats peak=8.02 → +185 % UNDER (CUDA-context floor wins)
* μ=384:  HWM_pred=5.99  vs mem_stats peak=8.03 → +34 % under (floor still in play)
* μ=768:  HWM_pred=11.17 vs mem_stats peak=8.38 → −25 % OVER (planner conservative)
* μ=1200: HWM_pred=17.50 vs mem_stats peak=13.04 → −25 % over
* μ=1800: HWM_pred=26.51 vs mem_stats peak=19.60 → −26 % over

vs bispinor's monotone −0.5 % to −10.8 % range. Two new effects identified:
(1) a ~8 GB CUDA-context / JIT-cache / NCCL-buffer-pool floor invisible to
the planner (dominant at small Peak C); (2) the 3-slot prediction is
**conservatively** correct in multi-chunk plans but **over-counts** when
only 1 r-chunk runs (slot 3's lifetime is contained in the OUTPUT slot via
aliasing). Both `pair_density_slots=3` and `fft_box_factor_D=2.0` are still
the right constants — they correctly characterize the structural worst-case.

**Other findings**:
* cusolverMpPotrf returns status=7 INTERNAL_ERROR under BFC + MEM_FRACTION=0.95
  on 2×2 mesh (works on 1×4 in bispinor sweep because `sharded_cholesky`
  path is selected for 1D meshes). Worked around with
  `cusolvermp_charge = off, cusolvermp_lu = off` in cohsex.in. Documented
  in `KNOWN_SANDBOX_ERRORS.md`.
* No OOMs, no crashes, no leaks. r_chunk = 13824 = n_rtot at every μ; the
  Si scalar load is so small the planner always picks a single chunk.

Run dir: `runs/Si/MU_nonbispinor_2026-05-18/`
Report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/agent_a_si_nonbispinor_mu_sweep.md`
Branch: `agent/si-nonbispinor-mu-sweep` on lorrax_A, tip `dc0b254`.

## 2026-05-18: Non-bispinor Si band-count sensitivity of gflat_memory_model planner [agent-C]

Stress-tested the planner's nb-scaling at fixed (kgrid, μ) by sweeping
nb=100 → nb=200 on Si non-bispinor (ns=2 noncolin=true, no SOC; cohsex.in
`bispinor=false`) at two k-grids: 3×3×3 (μ=408 snapped from 384, 1 r-chunk)
and 4×4×4 (μ=816 snapped from 768, 4 r-chunks). 40 Ry ecutwfc (bumped from
25 Ry so nbnd=200 fits the smallest-k sphere on 3×3×3), `cusolvermp_charge=off`
+ `cusolvermp_lu=off` to dodge agent-A's BFC bug, 28 GB budget per device.

**Headline:** planner stays within Agent T's `-0.5% to -10.8%` window at 3
of 4 (kgrid, nb) pairs but **3×3×3 nb=200 breaks out to -13.9% under-prediction**
(HWM_pred 15.70 vs mem_stats peak 18.24 GB/dev). Other 3: 3×3×3 nb=100 -0.5%,
4×4×4 nb=100 -1.2%, 4×4×4 nb=200 -3.2%. The 13.6%-point %-err jump nb=100→nb=200
at 3×3×3 is the largest in any A/B/C sweep so far at non-bispinor scale.

**Per-term nb-scaling:** every planner term scales exactly as predicted —
`centroids_persist` doubles (×2.00 measured ≈ predicted ×2.00 from
`4·c128(nk, ns, μ, nb_total)/p_xy` with `nb_total = nb_left + nb_right =
2·nb_cohsex`); all flat-in-nb terms (P_pair, zeta_out, gflat_acc, L_q)
match within 1%. **Peak C HWM_pred itself moves only +0.07 GB nb=100→200
at 3×3×3** (centroids-only delta), but mem_stats peak grows +2.53 GB —
36× more than the planner predicts.

**Where the missing +2.32 GB lives:** live_arrays at `after_fit_one_rchunk`
grew by only +0.21 GB nb=100→200 at 3×3×3, but `peak − live_total` (the
XLA preallocated-temp budget) grew by **+2.32 GB**. That excess is
unmodeled in-jit transient inside `z_q_from_psi_sm._local`, candidate
shape `c128(nk, band_chunk, ns, r_chunk/p_y)` — the per-rank post-all_gather
slab on `psi_l_X` + `psi_r_X` (×2). Predicted slab diff at 3×3×3
nb=100→200: 2 × 27 × 128 × 2 × 13500 × 16 = 3.0 GB (no aliasing); observed
+2.32 GB is consistent with one slab × 2 with ~25% aliasing discount.

**Proposed planner refinement:** add an `M_all_gather` term to Peak C
persistent base. HLO calibration of `z_q_from_psi_sm._local` at 3×3×3
nb=100 vs nb=200 needed to nail the coefficient (1× vs 2× slabs, and
aliasing). Expected to fix the 3×3×3 nb=200 gap from -13.9% to roughly
-5%, AND explain agent-T's worst-case bispinor μ=768 -10.8% gap (CrI3
production's -8.5% is also probably this all_gather slab not NCCL
overhead alone).

**Cross-r-chunk leak check (4×4×4, 3 r-chunks measured):** zero growth in
live_total across consecutive r-chunks at either nb=100 or nb=200 —
bc-scan correctly aliases its slots, no nb-correlated cross-chunk leak.

**Sandbox infra:** Agent A's `unfold_psi` nspinor=1 fix (commit `8c18925`)
cherry-picked as `d4cb599` on lorrax_C so future truly-scalar runs aren't
blocked (also fixed the nspinor=1 phdf5-loader issue per A's `dc0b254`
landing — A's pair of fixes is now on `origin/agent/si-nonbispinor-mu-sweep`).
Even with the unfold fix, the production loader still pads ns=1 → ns=2 via
`load_wfns.py:psi_G_flat` `ns_pad`, so the planner runs ns=2 for this
sweep too (consistent with agent-A and agent-B).

Run dir: `runs/Si/BANDS_nonbispinor_2026-05-18/`
Report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/agent_c_si_band_sensitivity.md`
Branch: `agent/si-band-sensitivity` on lorrax_C, tip `d4cb599`.

## 2026-05-18: Non-bispinor Si k-grid scaling of gflat_memory_model planner [agent-B]

Stress-tested the planner on scalar (`bispinor=false`, ns=2 non-SOC) Si across
2×2×2 → 6×6×6 k-grids at fixed μ/nk_full ≈ 6 (μ=48, 192, 432, 1348 orbit-
unfolded). Held nb=100 logical bands, 25 Ry, 24³ FFT box. Each kgrid measured
under both `platform_false` (production allocator) and `bfc_pre95` (BFC +
preallocate=true + MEM_FRACTION=0.95) for OOM-relevant `mem_stats` peaks.

**Headline:** 4×4×4 −0.8%, 6×6×6 −6.1% — both inside bispinor's
[−0.5%, −10.8%] window from agent_T. Small kgrids (2×2×2 −96.5%, 3×3×3 −52.9%)
have huge fractional %-err but it's a constant ~5–8 GB CUDA/JAX/cuFFT/NCCL
floor, NOT a multiplicative scaling failure. Δ = peak − HWM_pred is roughly
constant (~5 GB) across kgrids.

**Per-term scaling**: every component within 5% of analytic predicted exponent
(nk^0, nk^1, nk^2, nk^3 classes all match — slopes 0.0, 0.9-1.18, 2.06-2.08,
2.86-2.87 respectively). `sphere_idx_replicated` stays at 1 buffer across all
kgrids — Round-6 canonical-accessor fix holds. `B_CCT_chol` becomes the 2nd-
largest peak at 6×6×6 (17.58 GB = 71% of C=24.82 GB) — at larger μ or with
bispinor cascade it could flip the bottleneck from C to B.

**Sandbox bug found**: `nspinor=1` (true scalar) is blocked by `get_spinor_rotations`
always returning (n_sym, 2, 2). Logged in KNOWN_SANDBOX_ERRORS.md 2026-05-18.
Worked around by using `noncolin=true, lspinorb=false` (nspinor=2 no SOC) —
same per-rank planner formulas exercise, but truly nspinor=1 would require
a fix in `sources/lorrax_B/src/common/symmetry_maps.py:1220` to special-case
the trivial spinor.

Run dir: `runs/Si/KGRID_nonbispinor_2026-05-18/`
Report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/agent_b_si_kgrid_scaling.md`
Branch: `agent/si-kgrid-scaling` on lorrax_B (no LORRAX-source modifications).

## 2026-05-17: HLO calibration of planner constants pair_density_slots / fft_box_factor_D [agent-D]

Production-scale bispinor 80 Ry CrI3 HLO dumps + analysis to empirically
calibrate the two free constants of `gflat_memory_model.py`'s Peak C / Peak D.
JID 53075115 (4 nodes / 16 GPU / 4×4 mesh).

**Result:** `pair_density_slots = 3` and `fft_box_factor_D = 2.0` both confirmed
exactly. M1 (bispinor 4-channel, r=24576, b=32, gflat=360): 3 pair-density slots
× 20.04 GiB each (charge) / 19.83 GiB (transverse) in `fit_one_rchunk`'s 60 GiB
preallocated-temp.  M2: accumulate kernel shows 2 FFT-box slots × 6.03 GiB
(factor_D=2, not 4).  M3 (gflat=1 sanity): all 4 channels complete cleanly,
Peak D drops to 4.35 GiB ≈ planner prediction (< 1% error).  M4
(non-bispinor cross-check): 3 slots × 14.79 GiB (ns=2) matches.

The lorrax_B `agent/bispinor-ibz` working-tree edits to `gflat_memory_model.py`
(`fft_box_factor_D=2.0` + `pair_density_slots_{charge,transverse}=3`) are
empirically validated and safe to commit.  Old `fft_box_factor=4.0` was 2×
over-conservative, leaking ~13.8 GB of phantom Peak D budget at cs=360.

Run dirs (HLO dumps preserved for re-analysis):
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_2026-05-17/`
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_gflat1_2026-05-17/`

Report: `reports/memory_model_refit_2026-05-17/agent_d_hlo_calibration.md`.

## 2026-05-16: CrI3 6×6 **80 Ry** bispinor 16-GPU gate — INCOMPLETE (wall budget) [agent]

Attempted the 80 Ry production-scale Σ^B internal-consistency gate on JID 53057076
(4 nodes / 16 GPU / 4×4 mesh, 2:30 alloc).  Setup complete and on disk:
`runs/CrI3/M_6x6_80Ry_2026-05-07/{0X_lorrax_bispinor_fullbz_16gpu_2026-05-16,
0Y_lorrax_bispinor_ibz_16gpu_2026-05-16}/`.  Centroids regenerated:
charge `centroids_frac_1508.txt` (existing), transverse
`centroids_frac_1504_current.txt` (new, orbit-aware, ~3 min on 2 GPUs).  Both
sets pass orbit-closure under CrI3's 6 spatial sym ops.

**Findings & fix landed**:
1. Auto-planner picked `gflat_chunk_size = 717` at 80 Ry / mesh=16 — cuFFT
   batched-plan scratch allocator fails (`Failed to create cuFFT batched plan
   with scratch allocator`, 12 GiB scratch on top of 12.91 GB FFT box).
   **Worked around**: set `gflat_chunk_size = 360` in cohsex.in → per-iter FFT
   box 6.48 GB/rank, plan creation succeeds.  *Bug class: planner's
   `fft_box_factor=4.0` undercounts cuFFT's actual plan-side scratch at
   large `cs · n_rtot`; should land a correction in
   `gflat_memory_model.py` next session.*

**Why the gate didn't complete**: bispinor ζ-fit per-r-chunk at 80 Ry is
~14 s × 138 chunks ≈ 32 min per channel, × 4 channels (charge + 3 μ_L) =
**~128 min for ζ-fit alone**, plus V_q + Σ^B ≈ ~10–15 min.  Run A total ~140
min vs ~125 min alloc remaining at first ζ-fit start.  Doubling
`band_chunk_size` from 16 → 32 did **not** measurably speed up the inner
chunk (still 13.5 s/chunk; bottleneck is pair-density × cuFFT at
n_rmu=1508 not the bc-loop count).  Run A killed twice, gate output never
written; 30 Ry 51 µeV verdict stands as the strongest evidence to date.

**Next session prerequisites**: same branch `agent/bispinor-ibz` HEAD `d96aa46`,
same run dirs (left intact with cohsex.in + manifest + recon driver), **fresh
4 h allocation** (~3 h Run A, ~2 h Run B, ~10 min recon).  `lxalloc` with
`--time=04:00:00 --constraint="gpu&hbm80g"`.

## 2026-05-16: CrI3 6×6 bispinor IBZ 16-GPU end-to-end gate PASSES (51 µeV / 4×4 mesh) [agent]

Final driver for the bispinor IBZ-cascade end-to-end gate at production scale.
Run A (full-BZ reference, `LORRAX_FORCE_FULL_BZ=1`) and Run B (IBZ cascade) both
completed Σ^B + `v_q_bispinor.h5` write on a 4-node / 16-GPU / 4×4 mesh allocation
(JID 53054263).  Reconstruction via `reconstruct_sigma_b.py` rebuilt ψ once and
diffed Σ^B[k, m, n] between the two `v_q_bispinor.h5` files.

**Verdict: PASS at 0.0512 meV** (gate threshold 1 meV; bit-identical to the 2-GPU
gate's 51 µeV).  `Σ_X scalar` (charge channel) is bit-identical (Δ=0); the 51 µeV
residual lives entirely in Σ^B.  Per-tile trace shifts reproduce the 2-GPU
"Lorentz-mixing-cancels" signature — (μ_L=1,1)↔(1,2) tiles each shift by ~600 meV
in the IBZ cascade leg, with the cross-shifts cancelling in the channel sum.

Wall times: Run A ~356 s to V_q (n_q_solve=36), Run B ~180 s to V_q (n_q_solve=8,
~2× speedup from IBZ cascade), reconstruction ~190 s.

Source: branch `agent/bispinor-ibz` advanced from `9956dff` to `d96aa46` over 5
src/ commits.  The load-bearing one for 16-GPU was **`4930dab`
`fix(v_q_bispinor,reader): mesh-padded sharded tile reads`** — `BispinorVqReader.get_tile`
was reading transverse tiles at the on-disk logical extent (n_rmu_T=298), failing
`_validate_block_divisible` under any mesh with a sharded axis ≥3.  Now rounds μ
extents up to `gx*gy` and passes the padded shape as `shape=` plus the logical
extent as `valid_shape=` — mirrors the write-side `_round_up_to_mesh` at
v_q_tile.py:1116-1118.  Concurrent fixes: `d96aa46` (NamedSharding scope shadow in
gw_jax.py main()), `eb4a1e0` (band_chunk floor at world_size), `2de70eb`/`4b927dc`
(WFN loader band-pad past mnband).

Both runs crashed in *post-Σ* artifact writers (`write_qp_wfn_h5` for Run A,
`write_results` for Run B) — both `nk` vs `nk_irr` indexing bugs in the
bispinor + full-BZ vs IBZ-cascade combination, not exercised by the 2-GPU
gate.  Σ^B and `v_q_bispinor.h5` were emitted before either crash, so the
gate observable is unaffected.  Deferred to a separate writer-fix initiative
before any downstream BSE / WFN_qp consumer is wired up.

Artifacts:
- `runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/sigma_b_gate_16gpu_v2_{A,B}.npz`
- `runs/CrI3/.../03_lorrax_bispinor_fullbz_16gpu_2026-05-16/tmp/v_q_bispinor.h5`
- `runs/CrI3/.../04_lorrax_bispinor_ibz_16gpu_2026-05-16/tmp/v_q_bispinor.h5`
- `recon_sigma_b_gate_16gpu_v2_2026-05-16/recon.out`

## 2026-05-16: CrI3 6×6 bispinor IBZ 16-GPU retest v2 — ABORTED (two new bugs at 4×4 mesh) [agent]

Re-attempted Run A on `agent/bispinor-ibz @ 9956dff` (band-pad fixes for
`load_centroids_band_chunked` and `psi_G_store._populate_from_loader` in place) on the
shared 4-node urgent allocation JID 53054263. **Two new bugs surface at 4×4 mesh; both
hit the failure-mode protocol's "STOP, do NOT patch" rule.** No Σ^B produced; cascade leg
(Run B) never launched.

- **Step `.0`** (HEAD as handed off): JAX init + COHSEX header OK, then phdf5 kchunk-union
  reader fails with `HDF5-DIAG: H5Dread failed` and `INTERNAL: phdf5
  read_kchunk_union: H5Dread failed`. Cause: replicated `counts` table doesn't clamp the
  tail rank's `(offset, count)` to the on-disk band extent when `mnband=86` is not
  divisible by `world * bands_per_rank=16 * 6`. SLURM cancelled the step after 41 min.
- **Step `.1`** (after an in-tree "PHDF5 FIX" + "MU FIX" added per-rank clamped counts to
  `src/file_io/wfn_loader.py` and `src/ffi/phdf5/read.py`; not committed): all 16 ranks
  raise `ValueError: empty band range: (86, 86)` in `WfnLoader.load:750`. Call site:
  `psi_G_store._populate_from_loader:225` requests the pad-only sub-block `(86, 86)` (the
  final, entirely-past-mnband band-chunk). The patched `WfnLoader.load` now rejects empty
  windows up-front, overriding the zero-fill path that commit `9956dff` added downstream.
  Followed by segfaults; SLURM cancelled the step.

Sandbox-errors log gained two new entries:
- `LORRAX_NGPU` is per-node in `lxrun`, not total — task specs that set
  `LORRAX_NGPU=16` (total) need `LORRAX_NNODES=4 LORRAX_NGPU=4` instead.
- Shifter env passthrough — `export LORRAX_FORCE_FULL_BZ=1` from the shell does not reach
  the container; the launch must add `--env=LORRAX_FORCE_FULL_BZ=1` to `LORRAX_SHIFTER`.

JID 53054263 still RUNNING at the time of write-up (~1 h 13 min left, 4/4 nodes idle).
No source modified by this session; both new bugs deferred to the orchestrator.

## 2026-05-16: CrI3 6×6 30Ry bispinor IBZ gate FAILS at 16 GPUs — band-axis world-size pad outruns WFN file [agent]

Re-running the prior 2-GPU bispinor IBZ end-to-end gate on the production
16-GPU / 4×4 mesh (`runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/03_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
and `04_lorrax_bispinor_ibz_16gpu_2026-05-16/`, lorrax_B `agent/bispinor-ibz`
@ `82520a1`) hit the **suspected small-`nbnd` band-sharding death mode**
(see `~/.claude/.../project_cri3_small_nbnd_band_sharding_suspect.md`)
on all 16 ranks, before any V_q tile, kernel, or HLO compile.

Mechanism (root cause identified, not patched):

- `common/meta.py:107-109` sets `b_id_4 = round_up(nband_user=84, world_size=16) = 96`.
- `gw/wavefunction_bundle.py:83` exposes `full_range = (b0, b4) = (0, 96)`.
- `gw/gw_init.py:1205-1209` → `common/load_wfns.py:437-439` → `loader.load(bands=(0, 96), ...)`.
- `file_io/wfn_loader.py:678-681` rejects `b_hi=96 > self.nbands=86` (NSCF `nbnd=86`).
- Error: `band range (0, 96) out of [0, 86); use bands_pad_to-style external padding for over-file requests`.
- The `meta.py:100-117` comment promises zero-fill past `b_id_4_user` in
  `load_centroids_band_chunked`; the actual call path doesn't slice or pad
  externally before the loader sees the over-file extent.

Mesh sensitivity (`nband_user=84`, NSCF `nbnd=86`):

| world_size | `round_up(84, ws)` | vs file `mnband=86` |
|------------|-------------------|---------------------|
| 2 (prior gate) | 84 | OK |
| 4 | 84 | OK |
| 8 | 88 | FAILS |
| **16 (this gate)** | **96** | **FAILS** |

So 8 GPUs would already fail; the production 16-GPU mesh fails by 10 slots.

Per task contract: no source patch, no commit, no retry on fewer GPUs.
Allocation 53050082 (`hbm80g`, 4 nodes) released. Full failure analysis +
recommendation in the gate report (returned to the user as text — no
report.md file written per subagent conventions). Run 04 (IBZ-cascade
leg) was skipped because both legs run the same `Meta.from_system` +
`prepare_isdf_and_wavefunctions` code path before either IBZ branch is
taken; the bug would fire identically.

## 2026-05-16: bispinor IBZ Σ^B gate PASS at 51 µeV on CrI3 6×6 30Ry [agent]

End-to-end internal-consistency gate for the new bispinor IBZ cascade
(`lorrax_B agent/bispinor-ibz` @ `882ed4a`, 3-vector Lorentz mixing on
TT tiles). Two paired LORRAX runs on the same CrI3 6×6×1 30 Ry SOC
QE/centroid reference under `runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/`
differing only in `cohsex.in: bispinor_use_ibz`:

- `01_lorrax_bispinor_fullbz_ibz_gate_2026-05-16/` — reference (false)
- `02_lorrax_bispinor_ibz_2026-05-16/` — new IBZ cascade (true)

Per-(k, n) Σ^B reconstructed via `reconstruct_sigma_b.py` (calls
LORRAX as library; no source mod) and diffed in `analyze_gate.py`:

- `max |Δ Σ^B[k, n]|` = **0.0507 meV** over (36 k) × (84 sigma bands)
- mean = 0.0027 meV; RMS = 0.0059 meV
- Gate threshold 1 meV → **PASS** by 20×
- Scalar Σ_X (charge channel): **bit-identical** between A and B (Δ = 0)

Per-tile traces of (1,1), (2,2), (1,2), (2,1) each shift by ~600 meV
under in-plane proper rotations with opposite signs that cancel in
the contracted Σ^B — positive evidence the 3-vector Lorentz mixing
`V^{ij} = R^{iα} R^{jβ} V^{αβ}_{IBZ-unfold}` is acting unitarily.

V_q kernel wall: 26.28 s (A) → 6.28 s (B), **4.18× speedup** on the
V_q stage (4.5× IBZ shrink on 36→8 q's for P-3). Total wall is
dominated by ζ-fit on this 6×6 case (~6 min) which is unchanged.

Both runs are blocked at QP output by the pre-existing kin_ion
crash (`KNOWN_SANDBOX_ERRORS.md` 2026-05-14) but Σ_X printing
completes first under `x_only = true`.

Plot: `reports/bispinor_ibz_e2e_gate_2026-05-16/sigma_b_gate_scatter.png`.

## 2026-05-15: CrI3 sym-vs-nosym L-phase + perm-direction fix [agent]

Two-bug fix on `agent/trs-aware-sym-fix` commit `0735c2a`:

1. **Missing per-centroid umklapp phase** `exp(2π i q · (L_μ − L_ν))` in
   `unfold_v_q`. `L_μ = floor(S r_μ + τ)` is now captured by
   `compute_centroid_sym_perm` (which returns `(sym_perm, L_table)`) and
   threaded through `_resolve_ibz_q_list` → `unfold_v_q` in both
   `gw/v_q_g_flat.py` and `gw/compute_vcoul.py`.

2. **Wrong centroid permutation direction** in `unfold_v_q`. The previous
   code used `inv_perm = argsort(sym_perm)` (= π⁻¹). Correct direction is
   `sym_perm` directly (= forward π). For involutive ops (MoS2 σ_h, Si
   cubic) the two coincide — that's why MoS2 + TRS passed at 0.090 meV
   while CrI3 C3/S6 sat at 4 eV.

Unit test: `reports/trs_sym_audit_2026-05-14/test_production_unfold_v_q.py`
closes V_q to ISDF noise floor (rel 8.73e-6 ≈ 22 eV out of |V|=2.5e6) on
all 36 q's of the CrI3 6×6 30 Ry dump, including non-involutive ops AND
umklapp shifts (kg0 ≠ 0). 13/13 pytest tests in centroid/unfold domain.

Convention reference document:
`reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md`. Empirically
verified that `wfn.sym_matrices = U` (forward direct-space sym), NOT K
(reciprocal rotation) as a prior agent claimed; the user's existing DFT
degeneracy tests across multiple systems relied on this convention.

Run: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym_lphase_fix_2026-05-15/`
(pending e2e gate; expected to drop max |ΔΣ_X| from 6 eV → <1 meV).

## 2026-05-14: CrI3 sym-vs-nosym PR3 validation — FAIL (third sym-handling bug) [agent]

Run dir: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/`. Source: lorrax_B
`agent/trs-aware-sym-fix` @ `8504994` + `a45f039` + `69ab42c`. Task #30 mirror
for a second inversion-containing system, complementary to MoS₂ (PASS, 0.090 meV)
and Si (FAIL, 160 eV — τ-phase bug).

Pipeline: regenerated nosym NSCF (ntran=1, nrk=36) from existing CrI3 SCF charge
density → 2 LORRAX cohsex runs (x_only=true, do_screened=false, bispinor=false,
bare_coulomb_cutoff=30.0) sharing the existing 300 orbit-closed centroids from
`M_6x6_30Ry_bispinor_2026-05-14`: `run_sym/` (ntran=6, P-3 spatial group with
inversion, IBZ cascade fires n_q_disk=8 → 36 full-BZ unfold) vs `run_nosym/`
(ntran=1, direct full-BZ).

Result: **max |ΔΣ_X(k, n)| = 6022 meV ≈ 6.02 eV** across all 36 k × 84 bands.
Uniform ~5 eV residual at every k-point (no clean k); worst rows at valence-top
d-bands (b=60-61, 56-57, 64-65); systematic mean −2046 meV (sym more negative
than nosym). NOT a PR3 firing — CrI3 has spatial inversion in mtrx ⇒ 0 TRS-fold
k-pts ⇒ PR3 (iσ_y·conj and τ-phase) is a strict no-op for this system, which
this test experimentally confirms. The 6 eV residual exposes a **third, distinct
sym-handling bug**: broken IBZ→full V_q (or ζ) cascade unfold for groups
containing C3 + improper rotations (S6, −I). The MoS₂ pass (E + σ_h only) was
insufficient to detect this; the Si τ-phase bug (non-symmorphic Fd-3m) is a
separate failure mode (CrI3 is symmorphic with τ=0 for all 6 ops).

Triage targets (next session):
1. Bisect against `9e644e9` (pre-Phase-2) on the CrI3 test bed to confirm
   pre-existing — mirroring the Si triage. Likely pre-existing.
2. Suspect: SIGN or CONJUGATE flip missing in V_q (or ζ) unfold when sym op
   has det=−1, OR wrong G-vector mapping under improper rotations / C3.
3. Fix pre-existing nrk=48 vs n_unfold=36 crashes in `qp_wfn.write_qp_wfn_h5`
   (line 137 shape check) and `gw_output.write_results` (line 288 indexer)
   — both fire AFTER sigma_freq_debug.dat is written and so didn't block this
   validation, but they should use `meta.nkpts_unfolded` not `wfn.nkpts`.

Report: `reports/trs_sym_audit_2026-05-14/cri3_sym_vs_nosym_pr3.md`.
Comparison: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/compare_sigma_x.{py,log}`.
Total cost ~8 GPU-min.

## 2026-05-14: sym-vs-nosym PR3 e2e validation gate — PASS [agent]

Run dir: `runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14/`. Source: lorrax_B
`agent/trs-aware-sym-fix` @ `8504994` + `a45f039` + `69ab42c`. Task #30 — the
load-bearing end-to-end gate for the Phase-2 sym refactor.

Pipeline: 1 kmeans run on `00_mos2_3x3_cohsex/qe/nscf/WFN.h5` (sym, ntran=2)
→ 399 orbit-closed centroids → 2 LORRAX cohsex runs (`x_only=true`,
`do_screened=false`, `bispinor=false`, `bare_coulomb_cutoff=30.0` explicit)
sharing those centroids: `run_sym/` (sym WFN, exercises PR3 unfold_psi +
iσ_y·conj on TRS k {1, 3, 4, 5}) vs `run_nosym/` (`02_mos2_3x3_nosym/qe/nscf/WFN.h5`,
ntran=1, IBZ cascade trivial).

Result: **max |ΔΣ_X(k, n)| = 0.090 meV** across all 9 k-pts × 56 bands.
Pass gate was ≤ 1 meV; observed residual is 11× below threshold and is
essentially the DFT-eigenvalue ULP-offset (0.069 meV mean) between two
independent SCF runs propagated through the same Σ_X kernel. TRS-group
mean |ΔΣ_X| = 0.028 meV vs non-TRS-group 0.030 meV — indistinguishable,
which is exactly the signature of a correct sym implementation. The PR3
audit (`audit_pr3.md` R3) had shown PR3 shifts Σ_X by ≤95 meV on this same
test bed (bug was firing); this gate confirms PR3's fix produces the
*physically correct* answer (matches direct nosym evaluation to ULP).

Report: `reports/trs_sym_audit_2026-05-14/sym_vs_nosym_pr3_validation.md`.
Comparison: `runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14/compare_sigma_x.{py,log}`.
Total cost ~3 GPU-min.  PR1+PR2+PR3 cleared from this gate.

## 2026-05-14: testbed_mos2_3x3_soc — PR3 ψ-side TRS-fix validation baseline [agent]

Run dir: `runs/MoS2/03_mos2_3x3_soc_2026-05-14/`. Source: lorrax_B
`agent/trs-aware-sym-fix` @ `796c043` + a one-line `dft_operators.py` migration
fix (see below). Goal: bring up a **non-inversion SOC** test bed so PR3's
ψ k-unfold iσ_y rotation + TRS-row Gkk τ-phase fix has a non-trivial signal.

Pipeline: QE SCF + NSCF + NSCFq (3×3×1, 30 Ry, noncolin+lspinorb, nbnd=58) →
BGW epsilon + sigma cohsex (`number_bands 56`, `bare_coulomb_cutoff 30.0` explicit) →
LORRAX kmeans (orbit-closed, 399 centroids from 206 reps) → dipole + kin_ion →
LORRAX `gw_jax` cohsex on 2 GPUs (nval=26 divisible).

Symmetry probe (`sym_analysis.log`):

```
ntran = 2  (E + σ_h; no_t_rev=.true. + SOC kills rotations)
len(sym.sym_mats_k) = 4
#k via TRS  = 4  (full-BZ {1, 3, 4, 5})
#q via TRS  = 4  (full-BZ {2, 6, 7, 8})
has_inversion = False  ← suitable PR3 test bed
```

Σ_X finite at every (k, n). Group-mean Σ_X (post-PR3, band 19..30 window):
  - TRS k group     N=48  mean = -17.585 eV
  - non-TRS k group N=60  mean = -17.523 eV

By the time this task was launched, PR3 (`8504994`) had just landed.  To
produce a real pre-PR3 baseline, ran cohsex AGAIN with src/ at `796c043` +
the same dft_operators fix.  PR3 ψ-side TRS-fix diff (`pr3_diff_summary.log`,
band-19..30 × 9-k window):

|       | Δx_bare max\|Δ\| | Δx_bare rms | Δeqp0 max\|Δ\| | Δeqp0 rms |
|-------|-----------------|-------------|----------------|-----------|
| TRS k | 59 mΩ           | 16 mΩ       | 64 mΩ          | 17 mΩ     |
| non-TRS k | 49 mΩ       | 12 mΩ       | 53 mΩ          | 13 mΩ     |

Δx_head and Δcoh_head are bit-equal (scalar head untouched by PR3).
Non-TRS k diffs are non-trivial because the wrong ψ at TRS-folded k
pollutes χ_0(q), hence W(q), hence Σ at every k through q = k − k′.
The 64-mΩ max-Δ_eqp0 sits in the 10-100 meV PR3 prediction band from the
task spec; TRS rows are ~20% larger than non-TRS rows.

**Source-code fix** (commit `a45f039` on `agent/trs-aware-sym-fix`):
`sources/lorrax_B/src/psp/dft_operators.py::generate_gvectors_k` was still
calling the post-P5-removed `sym.get_gvecs_kfull` API; patched to dispatch
through `WfnLoader.gvecs(k="full_bz")`, mirroring the pattern in
`psp/get_DFT_mtxels.py::_gvecs_full_cache`. This unblocks both
`psp.get_dipole_mtxels` AND `gw.kin_ion_io` on this branch (closes a
KNOWN_SANDBOX_ERRORS entry).

## 2026-05-14: testbed_cri3_6x6_30Ry_bispinor — bare Σ_X end-to-end on lorrax_B `agent/trs-aware-sym-fix` [agent]

Run dir: `runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/`. Source: lorrax_B
`agent/trs-aware-sym-fix` @ `a00722d` (post-PR2 V_q IBZ→full unfold lift).

Pipeline: QE SCF/NSCF (30 Ry, 6×6×1, noncolin+lspinorb, 86 bands) →
BGW epsilon + sigma (Σ_X reference) → LORRAX kmeans (scalar 300 +
current-density 298) → LORRAX cohsex (`x_only=true`, `bispinor=true`).

Result: finite bare Σ_X printed for the bispinor configuration. Off-diagonal
Σ^B tile traces ~7% of diagonal; spin-doubled degeneracy holds to 4 ULP;
tile hermiticity holds to 4 ULP. Cascade did not fire (bispinor mode
disables IBZ-only ζ writes by design in `gw_init.py:650`; CrI3 also has
`-I` so TRS folds = 0). Downstream QP analysis blocked by a separate
`kin_ion_io` `SymMaps.get_gvecs_kfull` bug (see `KNOWN_SANDBOX_ERRORS.md`).

Important sandbox surface area that this run uncovered (all logged in
`KNOWN_SANDBOX_ERRORS.md` 2026-05-14):
- `centroid.kmeans_isdf` is **not** a CLI module; use `centroid.kmeans_cli`.
- Bispinor V_q requires a second `--density-mode current` kmeans run AND a
  `centroids_file_current = ...` entry in `cohsex.in`; otherwise the
  bispinor branch silently falls back to scalar V_q and then crashes on a
  full-BZ vs IBZ ζ shape mismatch.
- This run config OOMs on 40GB A100 even with `band_chunk_size=2`,
  `r_chunk_size=8192` — `--constraint="gpu&hbm80g"` is required.

Caveat (documented in `README.md`): CrI3 has spatial inversion, so this
test bed validates only the bispinor V_q tile pipeline / cascade
machinery layout, NOT the iσ_y·conj TRS-spinor patch (Agent 1 sites
#5/#6/#7). A non-inversion bispinor system (1H-MoSe₂ + SOC, BiI₃, or
CrI3 + E_perp) will be needed when PR3 lands.

## 2026-05-13: zeta-fit memory model follow-up — 2nd HLO dump + Path D scaffolding on LORRAX_B [agent]

Reports: `reports/zeta_rchunk_memory_model_2026-05-13/{agent_1_hlo_verify,agent_2_structural_fix,hlo_findings}.md`.
Branch: `sources/lorrax_B` at `agent/zeta-bc-scan-shardmap` (commit `cdd0fba`).

Same allocation as the previous entry, follow-up work on the morning
commit `ff5873c` (LORRAX_A).

Two agents re-engaged via the tmux team:
- Agent 1 (HLO verification): independently re-read `module_0408`,
  confirmed `pair_density_slots = 3` (0.01% accuracy), `S_fft ≈ 3`
  (formula matches observed bytes at 0.1%), `psi_Y_full` aliases
  cleanly.  Flagged two errors in `hlo_findings.md` §2/§4: "band_chunk
  is the lever against W_wfn" was wrong (band_chunk is band-axis-
  invariant for this term — only `psig_k_chunk_size` and `(nb_L+nb_R)`
  move it); mesh dimension wasn't explicit.  Both fixed in
  `hlo_findings.md`.
- Agent 2 (structural fix design): evaluated 5 candidate paths
  (`fori_loop`+donation, `scan`+axis-naming, streaming einsum,
  bc-loop-inside-shard_map, `donate_argnums`).  Recommended **Path D**
  — push the bc-loop INSIDE the shard_map body via `lax.scan`.  The
  scan carry is per-rank-local; SPMD has already stopped at the
  shard_map boundary, so the WhileOp/SPMD trap that killed
  `solve_zeta`'s fori_loop attempt (cited in `isdf_fitting.py:1119-1141`)
  does not apply.  Expected: 58 → ~3 FFT-box slots.

2nd HLO dump at `psig_k_chunk_size=1`:
`runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_k1_2026-05-13/`.
Planner picked `r_chunk=17 568, band_chunk=16, gflat_chunk_size=320`;
predicted HWM = 47 GB/rank on a 35 GB budget (model now over-
conservative); XLA actually allocated **23.42 GiB/rank** in
`module_0307.jit__kernel.sm_8.0-memory-usage-report.txt`.  Reduction
from the OOMing run: 196 → 22 GiB (8.85× vs the model's 6×
prediction).  Extra factor came from `r_chunk` also dropping ~4×
(W_zeta term collapse).  Run completed the fit_zeta loop end-to-end
with no OOM, confirming `ff5873c` plus `psig_k_chunk_size=1` is a
working empirical config.

Commit `cdd0fba` on LORRAX_B lands the **Path D scaffolding** — two
read-only helpers that are independently testable and unlock the
larger 4c kernel rewrite:
1. `common.wfn_transforms.to_rchunk_inner` (Path D §4b): per-rank-
   local body of `to_rchunk` without the enclosing shard_map.
   Callable from inside another shard_map's body or a `lax.scan`
   body.  Numerical contract verified by three new tests at floating-
   point precision against `to_rchunk` on a 1×1 mesh.
2. `common.psi_G_store.PsiGStore._slice_local_tile_bc` (Path D §4a):
   host-tile slicer that takes a TRACED `bc_idx`, returns a padded
   `(nk, _bpd_max, ns, ngkmax)` array so `io_callback` inside a
   `lax.scan` body sees a static return shape.  Added `_bpd_max`
   field to `__init__`.

15 wfn_transforms tests pass (3 new), 12 psi_g_store + rchunk_gflat
tests pass (no regressions).

Path D 4c-e (rewrite `c_q_from_psi_sm` / `z_q_from_psi_sm` with the
`lax.scan` over bcs inside their shard_map bodies — the load-bearing
kernel rewrite, ~200-250 LOC across `isdf_fitting.py`) deferred to a
focused session.  Implementation sketch + validation plan are at
`reports/zeta_rchunk_memory_model_2026-05-13/agent_2_structural_fix.md`
§4c-e + §5.

Notes recorded for the future implementer:
- The current planner is now **over-conservative** (predicted HWM 47 GB,
  reality 22 GB).  This is a side-effect of the simple "everything
  lives concurrently" model; XLA aliases more aggressively when per-
  slot bytes drop.  Refining the planner to match reality is lower
  priority than Path D, which collapses the slot count entirely.
- The current feasibility-check raise is a sound lower bound
  (`band_fft_pool > budget` ⇒ definitely infeasible) but not tight.
  Folded into Path D's follow-up because Path D removes the need for
  this gate entirely.

## 2026-05-13: zeta-fit r-chunk memory model — 4-agent synthesis + HLO-verified bug fixes on LORRAX_A [agent]

Reports: `reports/zeta_rchunk_memory_model_2026-05-13/{consensus,hlo_findings}.md`
Branch: `sources/lorrax_A` at `agent/zeta-r-chunk-fixes-2026-05-13` (commit `ff5873c`).

Spawned a 4-agent independent study (4-pane tmux + Opus 4.7) of the
GWJAX zeta-fit memory model.  Two rounds: from-scratch v1 drafts
(`agent_{1..4}.md`, ~6k words each) → cross-reading v2 with disagreements
+ open questions (`agent_{1..4}_v2.md`, ~3.5k words each) → orchestrator
synthesis (`consensus.md`, 3.5k words).  Consensus identified two source
bugs everyone agreed on and three HLO-only disputes.

Empirical resolution on a CrI3 6×6 80 Ry HLO dump
(`runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/`,
planner-free at the report.md §7 60 GB / `band_chunk=16` config):
- Planner-free pick was `r_chunk=73 328` (16 chunks), HWM-estimated 52 GB,
  not the 12 500 cited in `report.md §7` — the original number was an
  unverified estimate.
- XLA actually requested 200.35 GiB per rank → `RESOURCE_EXHAUSTED` OOM
  at 196.30 GiB on 40 GB A100s.  3.85× model miss attributable entirely
  to the previously-unmodeled band-FFT pool: 58 concurrent live
  `c128[k_chunk, band_chunk, ns, nx, ny, nz]` slots at 3.22 GiB each,
  materialised UNSHARDED on every rank.  XLA cannot alias them because
  the Python-unrolled bc-loop in `c_q_from_psi_sm` / `z_q_from_psi_sm`
  has overlapping lifetimes between iterations.

Commit `ff5873c` lands two fixes on `agent/zeta-r-chunk-fixes-2026-05-13`:

1. `_bytes_centroids_LR` helper — fixes the `centroids_persist` term:
   replace `nk`-typo with `nb_total`; replace `shard=p_xy` with
   per-axis division for the L+R copies that live on disjoint mesh
   axes.  ~4× correction on a balanced 4×4 mesh.  Applied at the
   three sites (lines 184, 316, 350).
2. Add `band_fft_unsharded` term + structural feasibility check in
   `plan_gflat_chunks`.  Total cost is
   `nb_total · S_fft · psig_k_chunk_eff · ns · n_rtot · 16`
   per rank — **band_chunk-independent**; only `psig_k_chunk_size`
   reduces it linearly.  When the pool alone exceeds budget the
   planner raises a structured `ValueError` listing three mitigations
   (lower `psig_k_chunk_size`, narrow `nval/ncond`, or the structural
   `lax.fori_loop` rewrite).  `cfg.memory.psig_k_chunk_size` is now
   threaded through `gw_init.fit_zeta`.

Tests: `tests/test_aot_memory.py` + `tests/test_rchunk_gflat_pair.py`
13 passed / 3 skipped after the fix.  CPU smoke confirmed: planner
correctly refuses the previously-OOMing CrI3 config at
`psig_k_chunk=6` and picks a feasible plan
(`band_chunk=64, r_chunk=21 808`) at `psig_k_chunk=1`.

Follow-ups not in this commit, in priority order:
- Peak A's centroid-load FFT box is likely also unsharded (single
  slot, smaller impact).  Needs Peak A HLO confirmation.
- Wire `common/fft_helpers.query_fft_peak_bytes` per call site to
  replace the global `fft_box_factor=4.0`.
- Structural: convert `c_q_from_psi_sm` / `z_q_from_psi_sm` bc-loop
  to `lax.fori_loop`.  Would alias the `n_bc · S_fft` slots into one
  and recover most of the band-FFT pool's memory cost.

## 2026-05-13: WFN rchunk construction communication profile on LORRAX_D [agent]

Report: `reports/wfn_rchunk_profile_2026-05-13/report.md`

Inspected newest `sources/lorrax_D` WFN loading / `psi_n_XYk(rchunk)` path and
mined the freshest 4-GPU D-usage HLO profile while the new allocation remained
pending.  The source path intends rank-local behavior after
`PsiGStore.fetch_psi_rchunk` pulls each rank's host tile, but the HLO shows
large unwanted all-gathers in the fused per-rchunk kernel: repeated
`506.25 MiB` `all-gather-start` operations attributed to
`common/wfn_transforms.py:109`, gathering local
`c128[4,4,9,24,24,80]` FFT-box shards into full
`c128[16,4,9,24,24,80]` buffers.  `psi_G_store.populate.loader_load` is only
~0.84-0.86 s for five band chunks and `shard_to_host` is ~10 ms, so the
communication problem is not the host tile copy; it is the JAX/SPMD boundary
around G-flat gather / FFT-box materialization.  Recommended next edit: keep
the full `to_rchunk` pipeline inside a single `shard_map` region rather than
only wrapping the FFT.

Follow-up implementation trial on the same branch added an opt-in
`LORRAX_PSIG_RCHUNK_SHARDMAP=1` path:

- `src/common/wfn_transforms.py`: new `to_rchunk_shard_map` keeps G-flat
  gather, local IFFT, r-slice, and Bloch phase inside one `shard_map` region.
- `src/common/psi_G_store.py`: `fetch_psi_rchunk` dispatches to that variant
  when the env var is set.
- Fresh 4-GPU run:
  `runs/MoS2/00_mos2_3x3_cohsex/D_wfn_rchunk_shardmap_2026-05-13`.
  It completed end-to-end; `eqp0.dat` numeric rows match the prior D-usage
  profile exactly (timestamp differs).
- HLO result: the `wfn_transforms.py:109` `506.25 MiB` all-gathers disappear
  from `collectives_details.txt`.  Top remaining collectives are now V_q-side
  `gw/v_q_tile.py:717/718` all-gathers at ~183-188 MiB.
- Targeted validation passed on CPU:
  `24 passed` for `test_wfn_transforms.py`, `test_rchunk_gflat_pair.py`, and
  `test_psi_g_store.py`.  Full CPU-forced suite remains not clean due unrelated
  failures (`gw_jax` regression subprocess OOMed on GPU selection, plus the
  known `make_v_munu_chunked_kernel(... mesh_xy)` API drift in two V_q tests).

## 2026-05-13: GWJAX FFI/JAX boundary profile on LORRAX_A [agent]

Moved the profiling investigation to `sources/lorrax_A` on branch
`agent/ffi-boundary-profile-a`, based on `origin/agent/zeta-ibz-header`.
No LORRAX_A source files were modified.

Report: `reports/ffi_boundary_profile_a_2026-05-13/report.md`

Ran two 4-GPU MoS2 3x3 `gw.gw_jax` profiles with the sandbox profiling stack:

- Baseline A:
  `runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_profile_2026-05-13`
  (`path=sharded_cholesky` for charge, JAX/CUDA LU for transverse).
- cuSOLVERMp A:
  `runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_cusolvermp_profile_2026-05-13`
  with `LORRAX_USE_CUSOLVERMP_CHARGE_FACTOR=1` and
  `LORRAX_USE_CUSOLVERMP_LU=1`.

Key findings:

- End-to-end `run_module:gw.gw_jax` wall time was essentially unchanged:
  baseline `101.25 s`, cuSOLVERMp `101.40 s`; profiled totals were
  `85.422 s` vs `85.844 s`.
- HLO custom-call counts show only 3 `lorrax_cusolvermp_batched_potrf`,
  3 `lorrax_cusolvermp_batched_potrs`, and 9
  `lorrax_cusolvermp_batched_solve_lu` calls in the full cuSOLVERMp run, so
  direct Python/JAX-to-CustomCall boundary count is not the main wall limiter.
- cuSOLVERMp reduced HLO modules/compile count slightly (`1077 -> 1033`,
  XLA compile `21.5 s -> 16.4 s`) but greatly increased low-level GPU trace
  activity (`12k -> 242k` GPU events, `8 -> 654` compute streams).
- The strongest avoidable JAX overhead is first-run orchestration:
  roughly 600 cache misses in both runs, led by local `_per_rank` jitted
  closures in `src/file_io/_slab_io_ffi.py` read/write paths and repeated small
  primitives in `wfn_loader.py`, `gamma_matrices.py`, and `fft_helpers.py`.
- The largest steady-state levers are still high-level GWJAX collectives and
  data motion: `V_q_compute` at ~38.5 s, zeta fits at ~36-37 s total, repeated
  2.47 GiB all-gathers in `wfn_transforms.py`, and all-reduces in the
  factorization path.
- Follow-up: fast-forwarded `lorrax_A` branch `agent/ffi-boundary-profile-a`
  to the D usage stack (`lorrax_D/agent/cusolvermp-ffi-profile`, commit
  `c21d855`) and reran the same profile in
  `runs/MoS2/00_mos2_3x3_cohsex/A_rebased_D_usage_profile_2026-05-13`.
  Compile/retrace metrics improved (`582/562 -> 478` XLA compiles and
  `629/602 -> 525` cache misses), but wall time regressed to `150.30 s`
  (`133.750 s` profiled) because zeta HDF5 write/close time ballooned.
  The top cache misses still include `_slab_io_ffi.py` `_per_rank` factories
  (`20` read, `17` write), so the first caching pass is incomplete from JAX's
  callable-identity perspective.
- q-loop acceleration probe: prototyped an opt-in CUDA Graph replay path for
  the existing full-mesh `cusolverMpPotrf` q-loop using stable ctx-owned staging
  buffers.  The code built, but `cusolverMpPotrf` failed during stream capture
  with status `7` at `q=0` on all ranks under cuSOLVERMp 0.7.2 / NCCL 2.26.3.
  Baseline potrf for the same shape was `9.523 ms` median.  Removed the failed
  prototype and rebuilt the FFI shared library from the reverted source.

## 2026-05-12: cuSOLVERMp FFI 4-GPU profiling harness + Nsight traces [agent]

Branch `agent/cusolvermp-ffi-profile` on `lorrax_D`.

Added a dedicated 4-GPU benchmark/profiling driver near the cuSOLVERMp FFI:
`sources/lorrax_D/src/ffi/cusolvermp/profile_batched.py`.  The harness runs
under the normal `lxrun`/Shifter/JAX-distributed path, defaults to the
MoS2-3x3-like shape (`nq=9`, `n=640`, `mrhs=640`, `complex128`, `2x2` mesh),
prebuilds donated inputs outside the timed range, and supports
`nsys --capture-range=cudaProfilerApi`.

Report: `reports/cusolvermp_ffi_profile_2026-05-12/report.md`

Code/profile changes:
- Added NVTX step ranges to `batched_potrf_ffi.cc` and `batched_potrs_ffi.cc`
  for cross-stream waits, copies, descriptor setup, buffer-size query,
  workspace ensure, per-q cuSOLVERMp calls, and descriptor teardown.
- Added the CUDA toolkit target include directory to the FFI CMake include
  path so `<nvtx3/nvToolsExt.h>` resolves in the container.
- Captured rank-local Nsight Systems traces under
  `runs/FFI/cusolvermp_batched_profile_2026-05-12/nsys/`.

Findings:
- `potrf` at `nq=9, n=640, c128, 2x2` is ~9.45 ms median; the `nq` sweep
  is nearly linear (`nq=1`: ~1.95 ms, `nq=18`: ~17.86 ms).
- Combined `potrf_potrs` is ~28-29 ms median, with `potrf` ~9 ms and `potrs`
  ~18 ms in the trace.
- Descriptor creation, buffer-size query, workspace ensure, and destroy are
  single-digit microseconds per FFI call on this build; caching them is not
  the first high-impact optimization here.
- Cross-stream waits are also tiny (~5 us median per bridge). The dominant
  overhead is the serial per-q cuSOLVERMp call queue and its internal
  NCCL/cuBLAS/cuSOLVER kernels.
- Quick NCCL env checks (`NCCL_PROTO=Simple`, `NCCL_PROTO=LL128`,
  `NCCL_MAX_NCHANNELS=1`) all regressed this shape, so defaults are currently
  best.
- Verification: FFI C++ rebuild passed, `profile_batched.py` py-compiled,
  and a 4-GPU `potrf_potrs` smoke run passed. Full `uv run python -m pytest -q`
  failed outside the touched FFI/profiling files (`make_v_munu_chunked_kernel`
  test API drift, k-means label tie mismatches, and one CUDA OOM regression
  subprocess); see the report for details.

## 2026-05-12: cuSOLVERMp FFI profiling orientation [agent]

Oriented on `lorrax_D` branch `agent/zeta-ibz-header` for the distributed
linear-algebra FFI overhead investigation.  Read the sandbox skills,
profiling stack, LORRAX_D agent guide, current branch state, and the
cuSOLVERMp/cuBLASMp/SLATE/phdf5 FFI docs and hot paths.

Report: `reports/cusolvermp_ffi_profile_orientation_2026-05-12/report.md`

Notes:
- Active hooks found: `lxalloc`, `lxrun`, `lxshell`, `lxpre`; no `lxattach`
  hook was present in the searched sandbox/source paths.
- `batched_potrf_ffi.cc`, `batched_potrs_ffi.cc`, and cuBLASMp batched GEMM
  all recreate descriptors and re-query workspaces per FFI call; a shared
  descriptor/workspace-size cache on `LorraxCusolverMpCtx` is the clearest
  first optimization after adding NVTX ranges.
- Logged sandbox bookkeeping issues in `KNOWN_SANDBOX_ERRORS.md` for the
  `sources/lorrax` vs `sources/lorrax_D` doc mismatch and missing D-variant
  manifests.

## 2026-05-12: accumulate_rchunk_to_gflat μ-axis chunking [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

`accumulate_rchunk_to_gflat` now chunks the **μ axis (axis 1)**
inside the FFT-batch scan, replacing the previous n_q-axis chunking.
This bounds the per-rank FFT-box transient at CrI3 J_3x3 scale
(n_q=9, n_rmu_padded=1504, FFT grid 45×45×120) without OOM, and
crucially handles the n_q=1 case (Γ-only debug runs) that n_q
chunking could not.

### Why this change

The n_q chunking required `n_batch_chunks | n_q`.  CrI3 J_3x3 has
n_q=9 in the IBZ which is only divisible by {1, 3, 9}; MoS2 has
n_q=9 with the same restriction.  More importantly, single-q
debugging runs (n_q=1) had no way to chunk at all.  The μ axis is
always large (n_rmu_padded = several hundred to a few thousand) and
is already μ-sharded across ranks, so chunking it aligns naturally
with `with_sharding_constraint` decomposing the per-chunk
intermediates across ranks.

### Code changes

- `common/wfn_transforms.py: accumulate_rchunk_to_gflat`:
  - Replaced `_q_chunk` with `_mu_chunk = n_rmu_padded /
    n_batch_chunks`.
  - Scan body now `dynamic_slice_in_dim(..., axis=1)` on rch, with
    `_shard3` / `_shard5` constraints on pad_buf / box / G_box so
    each per-rank chunk is `(n_q, _mu_chunk/p_prod, ...)`.
  - qvec_frac path simplified: qvec is per-q (n_q, 3), broadcasts
    the same way for every μ chunk — no per-chunk slicing.
  - Sphere gather uses the shared `_gather_sphere` helper for both
    one-shot and chunked paths (per-q sphere broadcasts across the
    μ-chunk; no per-chunk sphere slicing needed since axis 0 is
    intact).
  - Divisor check: `n_batch_chunks | n_rmu_padded`.
- `common/isdf_fitting.py`: updated comment + default chunk
  selection (largest divisor of `n_mu_local` ≤ `num_chunks`).
- `tests/test_rchunk_gflat_pair.py`: tightened test (n_rmu = 6,
  divisible by every parametrised chunk count {1, 2, 3}) and
  updated the indivisible-rejection test to match the
  n_rmu_padded check message.

### Numbers (vs the 2026-05-11 baseline, both on the same git tree)

MoS2 3×3 bispinor end-to-end (4 ranks A100, full COHSEX):

| | 2026-05-11 baseline | 2026-05-12 μ-chunked | Δ |
|-|--|--|--|
| Σ^B(μ_L=1,ν_L=1) | -0.242923 eV | -0.242923 eV | 0 |
| Σ^X band 1 k=Γ | -40.0326 eV | -40.0326 eV | 0 |
| eqp0 max abs Δ vs baseline | — | 0 eV | bit-exact |
| Total wall | 47.3 s | 47.95 s | +0.65 s (~1.5%) |
| Per-chunk FFT box | n/a (one-shot) | 9 q × 40 μ × 46080 = 0.27 GB | — |

CrI3 J_3x3 G-flat (8 ranks A100, x-only):

| | 2026-05-11 baseline | 2026-05-12 μ-chunked |
|-|--|--|
| Per-chunk FFT box | n/a (one-shot OOMed at 98 GB) | 9 q × 47 μ × 243000 = 1.64 GB |
| ζ-fit | OOM at 98 GB | succeeds |
| V_q[CC] trace at q=0 | n/a | 120459594.32 |

(Run fails downstream at `kin_ion.h5` — same pre-existing
followup as the 2026-05-11 attempt; not a regression in this
refactor.)

Run dirs:
- `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_shardmap_2026-05-12/`
- `runs/CrI3/D_gflat_cri3_3x3_muchunk_2026-05-12/`

## 2026-05-11: Bispinor V_q orchestrator on G-flat — end-to-end MoS2 3×3 [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

The 7-tile bispinor V_q^{μ_L, ν_L} hot loop is now end-to-end on
the G-flat ζ disk format.  All seven unique tiles (CC + 3 TT
diagonal + 3 TT off-diagonal) run through the new per-q +
G-chunked kernel.

Run directory: `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_2026-05-11/`
Report:        `reports/gflat_e2e_bispinor_mos2_3x3_2026-05-11/report.md`

### Code changes

- `gw/v_q_g_flat.py`: factored a private `_compute_V_q_g_flat_one_tile`
  helper (~250 LOC) that drives one tile end-to-end.  Charge wrapper
  `compute_all_V_q_g_flat` reduced to a ~30-line bare-Coulomb
  v_per_G builder + helper call.  Kernel parametrized over
  `(n_rmu_L, n_rmu_R)` with separate L/R buffers; the same_zeta
  path still aliases L=R inside the jit.
- `gw/v_q_bispinor.py`: added `compute_V_q_bispinor_g_flat_to_h5`
  (~120 LOC).  Loops over `UNIQUE_TILES`, builds per-tile
  `v(q+G)` via new `_make_per_q_v_builder_for_tile` (CC = bare
  Coulomb; TT = bare · `(δ_ij − K̂_i K̂_j)`), calls the shared
  helper, streams each tile to HDF5.  Reuses the existing
  `tile_dataset_name`, `UNIQUE_TILES`, `HERMITIAN_PAIRS`,
  `BispinorVqReader` (output format is unchanged).
- `gw/gw_init.py`: bispinor dispatch reads the charge ζ's
  `isdf_header.zeta_layout` and routes to the new orchestrator
  on G-flat (opens 4 `ZetaReader` handles).  Legacy r-space path
  preserved as fallback.  Also: copy `sys_dim` onto `meta_curr`
  (dataclasses.replace strips dynamic attrs — caught by the
  bispinor shakedown).

### Numbers (vs the legacy bispinor smoke A_bispinor_smoke_2026-05-08)

ζ disk-shrink (per file, all in `tmp/`):

| File              | Legacy r-space | G-flat new | Ratio |
|-------------------|----------------|------------|-------|
| `zeta_q.h5`       | 4.0 GB         | 177 MB     | 23×   |
| `zeta_q_mu1.h5`   | 2.6 GB         | 181 MB     | 14×   |
| `zeta_q_mu2.h5`   | 4.2 GB         | 181 MB     | 23×   |
| `zeta_q_mu3.h5`   | 4.2 GB         | 181 MB     | 23×   |
| **Total ζ**       | **15.0 GB**    | **720 MB** | **~21×** |
| `v_q_bispinor.h5` | 446 MB         | 424 MB     | 1.05× |

`v_q_bispinor.h5` size is unchanged by design — V_q has
(μ × μ) axes, no G-axis.

V_q wall: 4.2 s for all 7 tiles on 4× A100 (extrapolated ~6×
faster than the legacy μ × ν tile driver from the charge-only
shakedown; total bispinor pipeline 47.3 s).

### Numerics

Bare Σ_X print at k=Γ matches the legacy r-space baseline to
**0.01 eV** band-by-band (-40.0326 new vs -40.0277 legacy at
band 1; matching delta of ~5 meV across all sampled bands).  The
residual is per-q sphere ⊂ shared sphere drop-out of cutoff-edge
G's, as designed.

Bispinor unit tests in `tests/test_compute_V_q_bispinor_g_flat.py`
(committed in `ac735cc`):
* 7 tiles agree with a per-q einsum reference V^{μ_L, ν_L}[μ, ν]
  = Σ_G conj(ζ_L) · v_q^{μ_L, ν_L} · ζ_R to 1e-10;
* CC tile from the bispinor orchestrator is bit-identical to the
  charge-only orchestrator on the same ζ_C file (confirms genuine
  code path sharing, not just structural duplication).

### Diagnostic (3-way Bare Σ_X check)

Ran a third comparison to disentangle the V_q rewrite from
code-drift between May 8 and today: legacy r-space writer +
legacy μ × ν V_q driver, on the SAME git rev as the G-flat run.

```
                                            Bare Σ_X k=0, band 1
G-flat new (today)                            -40.0326
Diagnostic — legacy path, same git rev        -40.0325     ← <100 μeV
May 8 baseline (A_bispinor_smoke)             -40.0572     ← 25 meV drift
```

So the V_q rewrite is **bit-equivalent** to the current legacy
code (1e-5 relative on every sampled band).  The 25 meV vs May 8
is intervening fixes to the legacy bispinor path (`_make_K_cart`
qvec_frac convention, IBZ unfold, Bloch-phase unification), not
my rewrite.  The new G-flat code never had those bugs because it
builds K_cart from per-q components with already-divided
fractional q.

xx/yy symmetry sanity (`||V_TT_11 − V_TT_22|| / ||sum||`):
May 8 baseline 0.97 (broken), today's legacy 0.50 (fixed), today's
G-flat 0.50 (matches).

### Notes

- Each new kernel compile emits ~8 `Involuntary full
  rematerialization` SPMD warnings (disk read at `P(None, ('x','y'),
  None)` reshards to `P(('x','y'), None)` inside the jit; XLA does
  full rematerialization instead of an all-to-all).  Non-fatal;
  ~20 MB per q per rank lost to the copy, dwarfed by the kernel
  time.  Followup: have the loader expose a directly-sharded read.

- `eqp0_bisp.dat` (full bispinor Σ^B reading the TT tiles) was
  not emitted by either the baseline or this run for this config —
  appears to need an output flag we haven't enabled.  Separate
  followup.

## 2026-05-11: G-flat ζ + new V_q orchestrator — end-to-end MoS2 3×3 [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

End-to-end shakedown on Perlmutter (1 node × 4× A100, 4-rank).
Writer ran in G-flat mode (`LORRAX_WRITE_G_FLAT_ZETA=1`), V_q via
the new `gw.v_q_g_flat.compute_all_V_q_g_flat` orchestrator, Σ
through the existing path.  No code changes to the kernel since
yesterday's swap commit — only orchestration patches caught in
shakedown.

Run directory: `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_xonly_2026-05-11/`
Report:        `reports/gflat_e2e_mos2_3x3_2026-05-11/report.md`

### Numbers (vs r-space baseline 02_lorrax_xonly)

| Quantity                 | r-space   | G-flat     | Ratio |
|--------------------------|-----------|------------|-------|
| `zeta_q.h5` size         | 2.3 GB    | **101 MB** | **23×** |
| Total wallclock          | 17.2 s    | **11.4 s** | 1.5× |
| `zeta_fit.close_io`      | 3.8 s     | 0.1 s      | **~38×** |
| `V_q_compute`            | 4.4 s     | **0.7 s**  | **6.3×** |
| Σ stage                  | 3.1 s     | 2.9 s      | 1.07× |

sigma_diag agreement: **5 decimals vs r-space baseline** at every
k, band sampled (per-q sphere is a strict subset of the legacy
shared sphere; the few cutoff-edge G's drop out by design since
`v(q+G) → 0` past `zeta_cutoff_ry`).

### Shakedown fixes (commit 6ebfc3e)

- `compute_all_V_q` dispatcher: async prefetch default OFF (env
  `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH=0`).  The worker-thread G-flat
  read deadlocks against the PHDF5 FFI collective in production
  (NCCL kernel collectives interleave with the MPI read collective
  via the GIL in ways that hang).  Sync loop is already 6.3× faster
  than the legacy driver — async is a future opt-in.
- `v_q_g_flat.compute_all_V_q_g_flat`: caller in `gw_init.py`
  passes `ZetaReader`, not `ZetaLoader` (the unit tests use the
  loader).  Orchestrator now detects which API is on hand and
  dispatches accordingly.
- Per-q progress print on the sync path (`read=…s, kernel=…s`):
  one line per q so a stuck JIT compile or NCCL hang is visible
  in `tail -F run.log`.
- `gw_init.py` ζ-peek diagnostic: reads `'zeta_q_G'` on G-flat
  files (was hard-coded to `'zeta_q'`).
- `compare_bgw_gwjax.py` (sandbox top-level): replaced the stale
  `common.wfnreader.WFNReader` import with raw h5py k-list read.

### Followups (unchanged)

- BGW agreement: 0.5 eV gap at band 19 for MoS2 3×3 x-only is
  **pre-existing in the r-space baseline** — not introduced by
  this rewrite.  Worth a separate dig.
- Async prefetch re-enable (NCCL ↔ MPI interleave).
- Bispinor 7-tile orchestrator
  (`gw.v_q_bispinor.compute_V_q_bispinor_to_h5`).

## 2026-05-11: V_q driver swap — new G-flat orchestrator [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

The G-flat V_q hot loop is now end-to-end: ζ̃ read off disk in WFN.h5
per-q sphere layout → v(q+G) built at the per-q Miller components →
G-chunked contract → dynamic_update_slice into (V_acc, g0_acc) →
IBZ-to-full unfold.

- `gw/v_q_g_flat.py` (NEW, ~280 LOC) — `compute_all_V_q_g_flat`.
  Replaces the legacy ``compute_V_q_tile`` / ``_choose_v_q_chunks``
  pipeline for the G-flat-on-disk case.  μ × ν tiling, the chooser,
  the in-V_q FFT, and the shared-sphere conversion all collapse:
  per q, one read + one ``compute_v_q_per_q_g_chunked`` call.  Async
  prefetch (single-step) overlaps the next q's read with the current
  q's contract (borrowed from `v_q_tile`).
- `compute_vcoul.compute_all_V_q` now dispatches on
  ``zeta_io.zeta_layout``: G-flat → new orchestrator; r-space →
  legacy `v_q_tile` path (kept as fallback).
- Tests in ``tests/test_compute_all_V_q_g_flat.py``: synthesised
  G-flat ζ file → orchestrator output bit-matches a one-shot
  einsum reference; async vs sync identical; r-space loader is
  rejected with a clear error.

### Followups

- Larger profile (Si 4×4×4 / MoS2 3×3×1) with the new path enabled
  to validate the disk-shrinkage + I/O-overlap wins claimed for
  the writer.
- Bispinor V^{μ_L, ν_L} 7-tile driver
  (`gw/v_q_bispinor.compute_V_q_bispinor_to_h5`) still uses the
  legacy r-space path; swap follows the same pattern (1 q at a time,
  G-chunked, per-tile signed v).

## 2026-05-11: per-q G-chunked V_q kernel + ζ-cutoff separate from V_q cutoff [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

### Two cutoffs, separate plumbing

`cfg.head.zeta_cutoff` is now an independent knob from
`cfg.head.bare_coulomb_cutoff`.  Both default to `ecutwfc` and cap at
`ecutrho`; `bare_coulomb > zeta` is a hard error (V_q would need ζ̃
values the writer never stored).  The on-disk per-q sphere is built
at `zeta_cutoff`; V_q's `sqrt_v(q+G)` mask uses
`bare_coulomb_cutoff`.

- `gw_config.HeadConfig.zeta_cutoff` (new field).
- `gw_init.fit_zeta`: shared `_resolve_cutoff` helper validates ≤
  ecutrho, raises on `bare > zeta`.
- `isdf_fitting.fit_zeta_to_h5(zeta_cutoff_ry=)` builds the per-q
  sphere at that cutoff and writes it to
  `isdf_header/zeta_cutoff_ry` (renamed from
  `bare_coulomb_cutoff_ry`).
- `ZetaReader` / `ZetaLoader.zeta_cutoff_ry` surfaces it.

### Per-q, G-chunked V_q kernel

New `compute_vcoul.compute_v_q_per_q_g_chunked(zeta_q_L, zeta_q_R,
v_q, g_chunk=...)` evaluates

    V_q[μ,ν] = Σ_G  conj(ζ̃_μ(G)) · v(q+G) · ζ̃_ν(G)

at a single q with the G-axis chunked into `g_chunk` slices.  Each
chunk is a GEMM-shape einsum `'mG,nG->mn'` on
`(n_rmu, g_chunk)` blocks — contiguous G access, no FFT, no
shared-sphere conversion.  Accumulator is donated so repeated calls
(e.g. one per q) sum in place under jit.

The companion `compute_v_q_per_G(q_irr_frac, gvec_components, ...)`
builds `v(q+G)` at the writer's per-q Miller list (matches the
legacy kernel's full-FFT-grid `get_sqrt_v_and_phase` output at the
sphere positions for both 2D slab and 3D bulk — tested).

This is the kernel the rewritten V_q driver will call once swapped
over; the existing `compute_V_q_tile` driver in `v_q_tile.py`
remains in place for the current production hot path.

### Tests

- `tests/test_v_q_per_q_g_chunked.py` (NEW, 9 tests):
  - kernel matches one-shot einsum (3 g_chunk sizes);
  - bispinor off-diagonal (L ≠ R, signed/complex v);
  - pad-slot invariance (ζ̃ = 0 at j ≥ ngk[q] ⇒ zero contribution
    regardless of v(G) there);
  - accumulator donation across multiple kernel calls;
  - alignment-error path (ngkmax not divisible by g_chunk);
  - `compute_v_q_per_G` ≡ legacy `get_sqrt_v_and_phase` at the
    sphere positions, for `sys_dim ∈ {2, 3}`.

### Followups

- Swap `compute_V_q_tile` / `_choose_v_q_chunks` over to the new
  per-q kernel.  The chooser shrinks to G-chunk + memory model
  (q-batching gone in this scope; comment marks the seam for a
  future opt-in).
- Sigma readers that consume V_q[μν] are unchanged — V_q's μ × ν
  output shape is identical to the legacy kernel's.

## 2026-05-11: G-flat ζ on-disk with WFN.h5-style per-q sphere padding [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

Writer now produces ``zeta_q_G(n_q, n_rmu, ngkmax)`` instead of
``zeta_q(n_q, n_rtot, n_rmu)`` when ``LORRAX_WRITE_G_FLAT_ZETA=1`` is
set, with per-q WFN.h5-style sphere components stored alongside.

- **`sources/lorrax_D/src/common/coulomb_sphere.py` (NEW)**
  - `compute_bare_coulomb_sphere_idx(...)` — shared single sphere
    used by V_q kernel.  Extracted from inline code in
    `compute_vcoul.py:246-263` so the writer and consumer share one
    source of truth.
  - `compute_per_q_bare_coulomb_components(...)` — per-q sphere
    `{G : |q+G|² ≤ cutoff}` for every IBZ q, padded uniformly to
    `ngkmax = max_q ngk[q]` with sentinel Miller index
    `(-nx/2, -ny/2, -nz/2)`.  Returns `sphere_idx_padded`,
    `gvec_components_padded`, `ngk_per_q`, `ngkmax`.
  - Fixed `_q_max_cart` bug: enumerates the actual BGW-wrapped
    q-list instead of using the `±0.5/kgrid` half-BZ corners
    (under-bound for the real q-list — even kgrid leaves q=K/2 at
    `q_frac = 1/2` outside the Wigner-Seitz cell).
- **`sources/lorrax_D/src/gw/compute_vcoul.py`**
  - Inline sphere construction at lines 246-263 replaced by a call
    to `compute_bare_coulomb_sphere_idx`.
- **`sources/lorrax_D/src/common/wfn_transforms.py`**
  - `accumulate_rchunk_to_gflat` accepts a 2-D per-q
    `sphere_idx (n_q, ngkmax)` in addition to the legacy 1-D shared
    sphere.  Uses `jnp.take_along_axis(mode='promise_in_bounds')`
    to dodge the XLA x64+shard_map verifier bug.
- **`sources/lorrax_D/src/file_io/isdf_header.py`**
  - New fields: `gvec_components (n_q, 3, ngkmax)`, `ngk_per_q (n_q,)`,
    `bare_coulomb_cutoff_ry`.  Required by `IsdfHeader.build` when
    `zeta_layout == 'G_flat'`; legacy r-space files read with these
    fields set to `None`.
- **`sources/lorrax_D/src/common/isdf_fitting.py`**
  - `fit_zeta_to_h5(..., vcoul_cutoff_ry=...)` accepts the bare
    cutoff; builds the per-q sphere, allocates
    `gflat_acc(n_q, n_rmu_padded, ngkmax)`, gathers per-q after each
    chunk's FFT, masks pad slots to zero post-loop, and persists
    components + ngk + cutoff in the isdf_header.
- **`sources/lorrax_D/src/gw/gw_init.py`**
  - Plumbs `vcoul_cutoff_ry` into both `fit_zeta_to_h5` call sites
    (scalar charge + bispinor transverse μ_L=1,2,3).
- **`sources/lorrax_D/src/file_io/zeta_loader.py` / `zeta_reader.py`**
  - Expose `gvec_components`, `ngk_per_q`, `bare_coulomb_cutoff_ry`,
    `ngkmax_zeta`.
  - Loader: G-flat-on-disk reads `zeta_q_G` directly via the new
    `_read_g_flat_disk` helper.  `layout='r_space'` raises
    `NotImplementedError` against a G-flat file (would need IFFT).
  - Reader: G-flat path raises `NotImplementedError` for the
    "narrow to shared sphere" sub-case (per-q → shared scatter not
    yet wired into the V_q hot loop); raw slab returns work.
- **Disk-size win** (`n_G_sph / n_rtot`, smaller is better):
  - MoS2 3×3×1, cutoff=30 Ry: **11.5%** of r-space (~8.7× shrinkage).
  - Si 4×4×4, cutoff=30 Ry: **16.9%** (~5.9× shrinkage).
  - Si 4×4×4, cutoff=120 Ry (=ecutrho): 94.4% (near full FFT box at
    the rho cutoff — expected).
- **Tests**
  - `tests/test_per_q_sphere.py` (NEW, 6 tests): helper correctness
    vs direct `(q+G)` enumeration, shared-sphere ⊇ per-q-sphere
    invariant, per-q accumulate matches reference FFT+gather,
    header round-trip + validation errors.
  - `tests/test_zeta_loader.py`: bumped 1 test's `IsdfHeader.build`
    call to supply the new required G-flat fields.
- **Validation**
  - 33/33 new + existing G-flat tests pass.
  - Full non-GPU pytest sweep: 181 passed, 20 skipped.  Pre-existing
    `test_kmeans_sharded` failures unchanged (independent of this
    branch).  GPU regression needs a CUDA job allocation (login-node
    cuSolver init fails — same as before).
- **Followups**
  - Wire the per-q → shared-sphere scatter into the V_q wrapper so
    the kernel can consume the new G-flat on-disk format.  Until
    then, the kernel keeps using r-space ζ files.

## 2026-05-11: chunk-capable local FFT helpers + slab-only phase helpers [agent]

Branch `agent/fft-batch-chunks` on `lorrax_A`, rebased onto `origin/main`
`92cbd83`.

- **`sources/lorrax_A/src/common/fft_helpers.py`**
  - added `apply_local_fft(...)`, a reusable device-local FFT helper with
    optional `fft_batch_chunks=` batching over all non-transform axes
  - threaded `fft_batch_chunks=` through
    `make_sharded_{f,if}ftn_3d`, `make_flat_k_fft`, and
    `query_fft_peak_bytes`
  - default remains `fft_batch_chunks=1`, so current production callers keep
    today’s one-shot FFT behavior unless a future refactor opts in
- **`sources/lorrax_A/src/common/wfn_transforms.py`**
  - added generic flat-r helpers:
    `extract_flat_rchunk`, `embed_flat_rchunk`,
    `apply_bloch_phase_flat_points`, `apply_bloch_phase_flat_rchunk`
  - `to_rmu(..., kvecs_frac=...)` now phases only the gathered centroid
    points instead of the whole FFT box
  - `to_rchunk(..., kvecs_frac=...)` now slices the flat-r slab first and
    applies the Bloch phase only on that retained slab
  - `to_rbox` / `to_rmu` / `to_rchunk` now also accept `fft_batch_chunks=`
    for future opt-in use
- **`sources/lorrax_A/src/file_io/zeta_reader.py`** and
  **`sources/lorrax_A/src/file_io/zeta_loader.py`**
  - threaded `fft_batch_chunks=` into the `G_flat` zeta read path so the
    upcoming `rchunk <-> G_flat` zeta/V refactor can reuse the same helper
    without reopening reader internals
- **Tests**
  - `tests/test_fft_helpers.py`: new chunked-helper coverage and
    chunk-aware `query_fft_peak_bytes` coverage
  - `tests/test_wfn_transforms.py`: new phased `to_rmu`,
    phased/chunked `to_rchunk`, and flat-r helper coverage
- **Validation**
  - `uv run python -m pytest -q tests/test_fft_helpers.py` → `5 passed`
  - `uv run python -m pytest -q tests/test_wfn_transforms.py` → `16 passed`
  - `uv run python -m pytest -q` → `182 passed, 20 skipped, 4 failed`
  - remaining failures are unchanged from `main`:
    - `tests/test_gw_jax_regression.py::test_gw_jax_matches_reference`
      (`write_qp_wfn_h5` shape mismatch)
    - `tests/test_kmeans_sharded.py::test_refactored_matches_naive[fcc-avec1]`
    - `tests/test_kmeans_sharded.py::test_refactored_matches_naive[skew-avec2]`
    - `tests/test_kmeans_sharded.py::test_pbc_distance_scan_matches_naive_fcc`
- **Report**
  - `reports/fft_helper_unification_2026-05-11/report.md`

## 2026-04-21: analytic chunk chooser + γ-calibrated AOT memory model [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.  Closes Phase 6 of the
AOT memory-model initiative — the chooser now predicts runtime peak
bytes to ≤1% error after γ calibration.

- **`src/gw/aot_memory_model/chooser.py`**: new
  `choose_chunks_analytic(sys, mesh, budget)` — regroups the 7 memory
  primitives into 4 scaling classes via a `PRIMITIVE_CLASSES` dict on
  the kernel (`const`, `cr`, `bc`, `crbc`).  The feasibility bound
  ``peak ≤ M`` is linear in chunk_r at fixed bc, so
  ``chunk_r_max(bc) = (M − α₀ − α_bc·bc) / (α_cr + α_crbc·bc)`` is a
  closed-form inversion.  Chooser 1-D searches over bc candidates, no
  2-D grid.  Adds an optional `fft_launch_overhead_flops` knob for
  calibrating the "small-bc performance hit" post-hoc.
- **`src/common/isdf_fitting.py`**: replaced the
  `jax.devices()[0].memory_stats()` peak tracker (returns `None` on
  this CUDA PJRT) with a single `nvidia-smi` sample at the end of the
  r-chunk loop.  Per-chunk sampling inside the Shifter container was
  observed to hang on some Perlmutter node types.
- **`src/gw/gw_init.py`**: prints `γ = runtime_peak / aot_predicted`
  at the end of `fit_zeta` whenever both numbers are available.  Also
  corrected the `aot_sys.n_b` passed to the predictors: use the union
  range (`nb_full`) not `nb_L + nb_R`; the cost primitive's factor of
  2 handles the L+R sum.
- **`fit_one_rchunk__current__fit.json`**: records **γ=0.510**
  calibrated at MoS2 3×3 nosym (runtime nvidia-smi = 3.06 GB vs AOT
  worst-case = 6.00 GB).  `Fit.gamma` is applied by both
  `predict_peak` and the analytic chooser's `_group_alpha`, so
  chooser-predicted peaks now match runtime to within measurement
  noise.
- **Validation** — with `memory_per_device_gb=4` and
  `use_aot_chunk_chooser=true` at MoS2 3×3:

  ```
  AOT chooser: chunk_r=46080 band_chunk=80 (1×1 jits,
      peak=3.06 GB / 3.88 GB = 79%, total=7.3 GF)
  ```

  Chooser-predicted 3.06 GB matches the earlier runtime nvidia-smi
  measurement (3.06 GB).  Budget is genuinely hit.  Bit-identical
  eqp0 vs baseline `c8fc139fb22d2653d585874fe19c72a7`.
- **Follow-ups**: (1) widen the bc DoE axis — current 11-sample fit
  collapses bc-sensitivity into zero β.  (2) γ calibration at Si 4×4×4
  60Ry — may need mesh-scaling γ rather than a global scalar.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — Phase 6 section
  with α decomposition, γ measurement, budget-hit validation table.

## 2026-04-21: phdf5 on-demand G-space during ISDF fit [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.  Follow-on to same-day
"jit the r-chunk body" work.

- **`src/common/isdf_fitting.py`**: new `use_phdf5_gspace: bool`
  parameter to `fit_zeta_chunked_to_h5`.  When True, the driver skips
  the device-resident G-space cache (`load_gspace_for_bands`) and
  instead calls `PhdfWfnReader.coeffs_gspace(band_range)` fresh per
  r-chunk per band-chunk.  The tuple is `del`'d right after the
  `fit_one_rchunk` jit returns — nothing persists between r-chunks.
- **`src/gw/gw_config.py` + `gw_init.py`**: `use_phdf5_gspace` surfaces
  as a `cohsex.in` flag and threads into `fit_zeta`.
- **Duck-type**: `PhdfWfnReader.coeffs_gspace` already returns
  `(n_k, nb_pad, n_s, nx, ny, nz)` with
  `P(None, ('x','y'), None, None, None, None)`, matching the cached
  path's shape/sharding contract exactly.  No FFI-reader signature
  changes were needed; the driver-side factory is four lines.
- **Validation** (MoS2 3×3, `use_phdf5_gspace=true`):
  - single r-chunk + `use_ffi_io=true`:
    md5 `c8fc139fb22d2653d585874fe19c72a7` ✓
  - multi-chunk (5×10000 + remainder 6080) + `use_ffi_io=false`:
    same md5 ✓
  - Multi-chunk + `use_ffi_io=true` fails with concurrent HDF5 MPI-IO
    errors in the async zeta_q writer.  Pre-existing interaction —
    PhdfWfnReader + SlabIO-FFI race on MPI-IO state on the same ranks.
    When both flags are needed, use `use_ffi_io=false`.
- **Memory win**: zero persistent GPU footprint for the per-band-chunk
  G-space cache between r-chunks.  ~265 MB per rank saved at MoS2 3×3
  (small); multi-GB at Si 10×10×10 1000+ bands, where it pushes the
  pre-rchunk CCT/cholesky stages back under budget.
- **Timing**: +0.2 s total at MoS2 3×3 multi-chunk (4.3 s vs 4.1 s).
  Negligible under phdf5; would be slow under legacy h5py (keep flag
  opt-in).
- **AOT model**: per-r-chunk peak unchanged — `psi_bc_G_tuple` is
  still a jit input, so `argument_size_in_bytes` is identical.  Phase
  1b benefits show up in the *between-rchunk* GPU residency, which the
  AOT kernel doesn't currently measure.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — new Phase 1b
  section with validation matrix and the FFI-writer conflict note.

## 2026-04-21: jit the r-chunk body + AOT-model fit_one_rchunk [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.

- **`src/common/isdf_fitting.py`**: new
  `_make_fit_one_rchunk_kernel` factory + `fit_one_rchunk` entry point.
  The full per-r-chunk body (FFT+reshard per band-chunk, streamed
  spin-traced pair-density accumulate, ZCT, Z→col reshard, Cholesky
  solve) is now one jitted kernel.  `fit_zeta_chunked_to_h5` calls it
  once per r-chunk.  Two compile variants per run (full + remainder).
- **`src/common/load_wfns.py`**: `get_sharded_wfns_rchunk_slice`
  signature refactored `(r_start, r_end)` → `(r_start, r_chunk_size)`
  so `r_start` can be a tracer inside an outer jit.  Callers in
  `iter_psi_rchunk_bandwise` updated.
- **`src/gw/aot_memory_model/kernels/fit_one_rchunk.py`**: new
  composite AOT kernel mirroring the production factory.  Captures the
  driver-level memory peak including coexisting buffers that per-stage
  kernels can't see.  Primitives: `Pacc`, `PrBc`, `psiBc`, `psiBcY`,
  `psi_cent`, `L_q`, `psiG_total`.
- **`src/gw/aot_memory_model/core.py`**: `SysDims` gains an optional
  `fft_grid` field + `fft_shape` property for kernels that need both
  k-grid and real-space FFT box.
- **`src/gw/aot_memory_model/presets.py`**: `points_fit_one_rchunk`
  for `mos2_3x3` and `si444_60Ry`.
- **`src/gw/gw_init.py`**: logs the AOT-predicted driver peak
  alongside the existing per-stage heuristic — sanity-check-only, does
  not override `chunk_r` yet.
- **Validation**: MoS2 3×3 COHSEX single-chunk (46080 pts) and
  multi-chunk (r_chunk_size=10000, 5 chunks + remainder 6080) both
  produce `md5sum eqp0.dat == c8fc139fb22d2653d585874fe19c72a7` matching
  the reshard-fix baseline.
- **NNLS fit** (11 DoE points, residual RMS 0.23 GB on ~3 GB peaks):
  β[PrBc]=1.03, β[L_q]=5.02, β[psiG_total]=1.65.  Saved at
  `src/gw/aot_memory_model/artifacts/fit_one_rchunk__current__{fit,samples}.json`.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — new "Update
  2026-04-21" section with primitives table, fit coefficients, next
  steps.
- **Next**: (1) host-resident `cached_gspace` via phdf5-like duck type
  — attacks the `psiG_total=1.65` primitive directly; (2) switch
  `compute_optimal_chunks` to use AOT prediction for the
  pair+zct+reshard+solve sub-loop; (3) γ-calibrate at Si 4×4×4 60 Ry.

## 2026-04-20: phdf5 FFI — independent writes by default, Cray MPICH now viable [agent A]

Branch `agent-A/independent-writes-default` on `lorrax_A`.

- **`src/ffi/phdf5/cpp/ctx.h`**: split `use_collective` into
  `use_collective_read` (default `true`) and `use_collective_write`
  (default `false`).  `coll_metadata` now defaults to `false`.
- **`src/ffi/phdf5/cpp/context.cc`**: new env-var surface.
  `LORRAX_PHDF5_INDEPENDENT=1` still forces reads independent too (power
  user override).  New `LORRAX_PHDF5_COLLECTIVE_WRITES=1` to opt writes
  back into collective (do NOT set on Cray).  New `LORRAX_PHDF5_COLL_META=1`
  to re-enable collective metadata.
- **`src/ffi/phdf5/cpp/write_ffi.cc` + `read_ffi.cc`**: dxpl selection
  now uses the per-direction flag.
- **Why**: the Cray MPICH collective write driver
  (`ad_cray_write_coll.c:669`) OOMs at ≥ 1 GB/rank regardless of
  `cb_*`, `stripe_*`, `alloc_time`, or `cray_cb_write_lock_mode` knobs.
  The fix that prior investigation missed was the combination of
  `H5FD_MPIO_INDEPENDENT` writes AND non-collective metadata ops —
  both are needed to fully bypass the buggy driver.  Independent
  writes are neutral on OpenMPI at our measured sizes; collective
  reads are preserved (ROMIO two-phase is optimal on both stacks).
- **Regression data** (1 node / 4 GPUs, post-fix defaults):

  | workload | OpenMPI | Cray MPICH |
  |---|---|---|
  | MoS2 3×3 `phdf5_multi_offset_test` | PASS | PASS |
  | MoS2 3×3 `phdf5_profile` (45 MB WFN) | 18.1 ms (was 18.3) | **17.9 ms** |
  | MoS2 3×3 `phdf5_profile --centroids` (gw_jax load) | 26.2 ms | 26.4 ms (parity) |
  | n=16384 C128 `phdf5_read_bench` (4.29 GB) | 3.04 GB/s | **3.79 GB/s** (was CRASH) |

  Cray now works at all scales and beats OpenMPI at large scale; at
  MoS2 3×3 scale the two stacks are within noise.  Unification around
  Cray for cross-cluster portability is viable.
- **Docs**: `src/ffi/phdf5/ARCHITECTURE.md` env-var table and
  `src/ffi/PORTING.md` Option B write-up refreshed.

## 2026-04-20: flat-k FFT helper — one wrapper for kx/ky/kz across the GW pipeline [agent C]

Branch merged to `main` as commit `c9bd801`.

- **`src/common/fft_helpers.py`** — new `make_flat_k_fft` /
  `make_flat_k_ifftn` / `make_flat_k_fftn`.  Callers hand it flat-k
  `(nk, *trail)` arrays, the helper does
  `reshape → with_sharding_constraint → custom-partitioned 3-D FFT →
  reshape back`.  `kgrid` and the 3-D PartitionSpec are closure state;
  the 3-D form never appears in caller code.
- **Call sites wired through**:
  - `gw/w_isdf.py` chi0 minimax — three FFT closures collapsed to
    helper calls (`Gv_ifftn`, `Gc_fftn`, `chi_fftn_local`).
  - `gw/ppm_sigma.py` `_sigma_kij_kernel` — `_fft_flat_G` /
    `_fft_flat_V` closures replaced.
  - `gw/gw_jax.py` — `_make_fft_pair` factory removed in favor of
    direct helper calls for G and V.
  - `common/isdf_fitting.py` — `CCT_LR`, `CCT_LR_spin_matrix`,
    `ZCT_LR`, `ZCT_LR_spin_matrix` all refactored to take flat-k
    input end-to-end.  The pre-ZCT `reshape → with_sharding_constraint`
    in the r-chunk loop was deleted (ZCT now takes flat-k directly).
    `donate_argnums` re-enabled on CCT_LR (0, 1), ZCT_LR `_left_ifft_conj`
    (0) and `_right_ifft_mul_fft` (0, 1), and on the spin-matrix
    variants — a handful of one-shot trace-time XLA aliasing warnings
    remain (rank-3 → rank-5 intermediate), but there are no per-call
    donation failures.
- **Validation** (`runs/Si/C_flatk_si10/`, Si 10×10×10 mem12, 4 GPUs):
  - `eqp0.dat` **byte-identical** to `C_stream_si10_transposed`
    baseline (0-byte diff).
  - Total runtime 304.7 s → **275.9 s** (9.5 % faster).  Savings
    concentrated in `zeta_fit.chunk_loop` (30 s → 25.5 s) from
    pair-density donation; `close_io` unchanged (65.9 s → 63.2 s);
    Σ computation unchanged.
- **Why it matters**: single point where the 3-D FFT happens, so the
  NUFFT substitution the user has in mind is a one-file change with
  no call-site churn.

## 2026-04-18 (midday): scissor-shift for out-of-grid bands + Si 4×4×4 pseudobands end-to-end [agent A]

Branch `agent-A/scissor-shift-sc-gw` on `lorrax_A`
(commits `dfc880c`, `9b0e666`).  Full write-up in
`reports/scissor_shift_2026-04-18/report.md`.

- **`src/gw/scissor.py`** — `ScissorFit` dataclass, `fit_scissor` (numpy
  OLS, separate valence / conduction lines), `extrapolate_delta_e`, and
  `add_diag_to_H_kmn` (shard_map-based diagonal add onto a
  `P(None,'x','y')`-sharded Hamiltonian, ready for the future SC loop).
  Smoke-tested 4-GPU: fit recovers synthetic slopes to <6e-6, sharded
  diagonal add bit-identical to numpy (maxabs 0.0), P(None,'x','y')
  output sharding preserved, divisibility check raises cleanly.
- **`src/gw/gw_jax.py`** — G0W0 PPM post-processing now honors the
  `sigma_at_dft_extrapolate` config knob: out-of-grid bands get the
  fitted affine QP correction instead of the static-COHSEX fallback.
  Fixed two adjacent bugs while wiring:
  - `E_qp_ev * ryd2ev` unit double-count in the original `in_grid`
    mask — every state looked out-of-grid, so the diagonal Sigma
    fixed-point was silently discarded for all bands.
  - in-grid test must use `E_DFT`, not `eigvalsh(H_qp)`, because
    pseudobands' non-unit norms scale `<n|H|n>` by the pseudoband
    weight and produce garbage eigenvalues for compressed states.
- **First end-to-end test** — `runs/Si/A_06_si_4x4x4_scissor/`.
  Si 4×4×4 nosym, BGW-convention pseudobands via
  `psp.run_nscf --pseudobands` (50 windows × 2 pseudobands = 8 prot +
  98 pseudo = 106 bands).  Σ(ω) grid ±5 eV so all pseudobands
  (onset ~+10 eV) are out-of-grid.  GN-PPM G0W0 + scissor on 4×A100:
  run wall 30 s, 668/6784 in-grid, valence fit
  α=-0.44, β=-6.24 eV (RMSE 1.07 eV), conduction fit
  α=-0.64, β=-0.61 eV (RMSE 2.37 eV).  E_QP vs E_DFT is a single
  smooth line across the full 0→330 eV bandrange with no jump at the
  in-grid / out-of-grid boundary — continuity goal met.  Magnitudes
  over-correct at the high-E tail (E_QP ≈ 0.36·E_DFT → highest
  pseudoband lands at ~120 eV vs DFT 330 eV); expected for a line fit
  over 10 eV extrapolated to 300 eV, worth revisiting with a softer
  A + B/E tail or a damped law later.
- **Known issue documented**: `psp.get_dipole_mtxels` crashes on a
  pseudobands WFN (`vnl_velocity_matrix` hits a `None` `dZ`).  Worked
  around in this run by routing the q→0 head through the BGW
  `eps0mat.h5` from `runs/Si/02_si_4x4x4_nosym/01_bgw_gnppm/` with
  `wcoul0_source = epshead` in `cohsex.in`.

Validation: `uv run python -m pytest -q` on `lorrax_A` → 13 passed,
1 pre-existing reshard failure (unchanged from prior state).

## 2026-04-18 (overnight): sigma_ppm cleanup + compile-cache trims + zeta_fit probe [agent C]

Branch `agent/C-sigma-ppm-cleanup` on `lorrax_C`. Full write-up in
`reports/session_2026-04-18_async_probe/report.md`.

MoS2 3×3 / 4-GPU run_module wall: **47.3 s → 34.7 s (−27 %)**, eqp0.dat
bit-identical at every commit (16 substantive + 1 TEMP profiling).

- **Reduce-scatter in `_sigma_kij_kernel`**: `projection_kernel.project_ri`
  tail replaced by a shard_map'd local einsum + `psum_scatter × 2` (m on x,
  n on y).  σ^τ now emerges `(m_X, n_Y)`-sharded; every downstream ω-kernel
  multiply + accumulate is rank-local.  HLO diff shows 4× `all-reduce
  c128[9,2,2,320,80]` flipping to `reduce-scatter c128[2,9,40,2,320]` per
  τ step, same byte volume but output is now sharded.
- **σ^τ as a (re, im) tuple** from the shard_map — removes the
  `sigma_ri[0]/[1]` indexing pjits and the `is_fully_addressable` assert
  that a multi-process tuple-unpack of a sharded (2, …) stacked array
  would trigger.
- **New `_ReduceScatterGpuAccumulator`** is the default buffered path;
  Σ_c(ω, k, m, n) is held sharded on GPU so it's n_b²/p² per rank instead
  of replicated.  `_BufferedGpuAccumulator` deleted (was redundant).
- **lax.scan τ-loop infrastructure** landed as `_get_sigma_tau_scan_kernel`
  + `_ReduceScatterGpuAccumulator.run_window_scan`, **off by default** —
  regresses at MoS2 3×3 scale (fewer overlap opps with big fused module +
  per-window compile).  Reconsider at padded-τ or larger mesh.
- **Physics visibility pass**: module docstring states the quadrature
  formula directly; `_iter_branches` NamedTuple with comments deriving
  kernel_sign / scale flips; `_run_sigma_branch` reads like a physics
  outline; `_combine_coeff_with_sigma_tau` documents the re/im split and
  drops the dead "real" branch; `_convolve_sigma_branch_kij` →
  `_run_sigma_branch`; 'channel' scrubbed from factory names.
- **Dead-param / dead-class purge**: `omega_sign_flip` (always +1), the
  unused `_BufferedGpuAccumulator`, the one-line wrapper
  `_accumulate_tau_into_window`.  −203 / +102 lines in one commit.
- **Compile-cache trims — numpy for tiny host-side helpers**:
  `get_enk_bandrange`, `fft_integer_axes`, `exp_ikr_fftbox`, `_build_Gij`,
  `_build_occ`.  Each had emitted ~8–16 standalone pjits at trace time
  for pure host bookkeeping.  TRACING CACHE MISS 313 → 269 (−44).
  `wavefunction_setup` section **1.79 s → 0.18 s** (the old `jnp.zeros_like`
  + `.at[].set` on sharded input had a non-trivial runtime tied to
  cross-device scatter).
- **zeta_fit chunk loop**: dropped the per-chunk `sync_global_devices`
  (the allgather is itself a collective; one rendezvous at the end is
  enough).  Investigated async-allgather paths and confirmed JAX has no
  async `process_allgather`-to-host API; the 1.95 s first-collective
  NCCL setup is the floor without pre-warming or the phdf5 FFI path.

Future work documented in-tree (heavy comments at each extension point):
  τ batching, m-chunking, `_CollectiveFlushSlabIoAccumulator` (FFI SlabIO
  collective-write variant for multi-process streamed output).  `zeta_fit`
  remains the dominant cost bucket (47.6 % of total) and is the natural
  next target.

## 2026-04-17 (pm): k-means ISDF — parallelism refactor + 4-GPU sharding prototype [agent B]

Branch `agent/kmeans-sharded` on `lorrax_B`. Full write-up in
`reports/kmeans_sharded_2026-04-17/report.md`.

- **Refactored `centroid/kmeans_isdf.kmeans_update_step`** to eliminate the
  double (P, K, 3) tensor materialization: segment-sum over labels replaces
  the one-hot-mask weighted mean; a `lax.scan` over K-chunks replaces the full
  pairwise distance tensor (peak (P, `k_block`, 3) instead of (P, K, 3)).
  PBC minimal image and metric tensor behavior are byte-compatible with the
  old implementation (new regression test covers orthorhombic / FCC / skew
  cells and the cross-boundary minimum-image case).
- **Added `make_sharded_kmeans_update`** — `shard_map`-based parallel Lloyd
  step. P sharded on mesh axis `'x'`, centroids replicated, one `lax.psum`
  per iteration on the (K, 3) / (K,) accumulators. Verified bit-identical
  single-GPU vs 4-GPU on Si 4×4×4 (matching md5 on `centroids_frac_128.txt`),
  same 71-step trajectory.
- **Fixed latent `alat`-vs-`Å` mislabel in `main()`.** BGW WFN.h5 stores
  `avec` in alat units and `alat` in Bohr; the old code treated `|avec row|`
  as Å, which silently inverted a ~2× grid upsample into a ~0.6× downsample.
  `main()` now converts to Å once via `wfn.alat * BOHR_TO_ANG`; the kmeans
  function docstring states distances inherit the caller's avec units.
- **Multi-process bootstrap.** Added the standard `_maybe_init_jax_distributed`
  to the module so `srun -n N>1` works (matches `psp/run_nscf.py`, `gw/gw_jax.py`).
  Prototype uses the simpler single-process-4-GPU path.
- **New tests** (`tests/test_kmeans_sharded.py`): 5 cases, all pass. Full
  suite: 18 pass, 1 pre-existing failure in `test_reshard_all_to_all.py`
  unrelated to this branch.
- **Sandbox doc hardening.** `skills/execute_workflow/SKILL.md` now says
  explicitly: never export a `SLURM_JOBID` you did not allocate yourself; the
  interactive-allocation section documents the background `salloc` +
  `-J lorrax_X_agent` naming pattern. Matching memory pointer at
  `memory/feedback_never_share_allocation.md`.
- **Run**: `runs/Si_B/00_si_4x4x4/` — fresh 4×4×4 Si sym-reduced QE (8 IBZ
  k-pts, 24³ FFT, 16 bands) → 48³ kmeans grid. Three sub-dirs hold the
  baseline, refactored-single-GPU, and sharded-4-GPU centroid outputs for
  the equivalence check.

## 2026-04-17: Three parallel LORRAX checkouts (A/B/C) for concurrent agents

Consolidated the previous per-sandbox LORRAX clones into three sibling
checkouts at `$HOME/software/lorrax_{A,B,C}`, symlinked into the sandbox
as `sources/lorrax_{A,B,C}`. Each agent session claims one letter and
touches only its own checkout. Shared Shifter stage trees remain at
`/pscratch/sd/j/jackm/lorrax_{nvhpc,phdf5_openmpi}` (read-only in the
container), so the three variants share bind-mounted deps but build
their own `src/ffi/common/cpp/build/liblorrax_ffi.so`.

- `config/perlmutter/install.sh`, modulefile template: new
  `LORRAX_MODULE_NAME` variable lets each checkout install its own
  modulefile (`lorrax_A`, `lorrax_B`, `lorrax_C`). `family("lorrax")`
  makes variants mutually exclusive in a single shell; across shells
  they are fully independent. Landed on `main` (LORRAX feature branch
  `agent/multi-checkout`, fast-forwarded).
- Sandbox `AGENTS.md`: new "Which agent are you?" section at the top,
  revised source-code table, non-negotiable rule #7 ("only edit your
  assigned checkout"). `execute_workflow`, `checkpoint`,
  `profiling_stack` skills updated to say `sources/lorrax_X` /
  `module load lorrax_X`.
- Deleted stale sandboxes: `lorrax_sandbox_fresh`,
  `lorrax_sandbox_profiling`, and their backing clones
  `$HOME/software/lorrax_{bse,profile_ppm}`.
- `pyproject.toml`: dropped the sandbox-level `lorrax` editable
  dependency; the path no longer resolves to a single variant. Host
  Python that imports LORRAX should run inside Shifter via `lxrun`, or
  `uv run` from inside a specific `sources/lorrax_X`.

## 2026-04-17: WFNReader full-zone symmetry wrappers

- Audited the raw wavefunction reader usage after the nonsymmorphic-phase work.
  In active `src/`, raw `get_cnk()` / `get_cnk_batch()` are only consumed by
  `SymMaps.get_cnk_fullzone*`; there was no active path pairing unfolded
  `get_gvecs_kfull()` output with raw irreducible-zone coefficients.
- Clarified the API in both `src/common/wfnreader.py` and
  `src/file_io/wfnreader.py`: raw `get_cnk*` / `get_gvec_nk` remain explicit
  irreducible-zone readers, and new `get_gvecs_kfull`,
  `get_cnk_fullzone`, and `get_cnk_fullzone_batch` wrappers now route full-BZ
  access through `SymMaps` so the non-symmorphic `tau` phase is applied in the
  safe path by construction.
- Switched active consumers in `src/common/load_wfns.py`,
  `src/bandstructure/htransform.py`, and
  `src/centroid/get_charge_density.py` to the new WFNReader full-zone
  wrappers.
- Verified on Si `4x4x4` symmetry-vs-nosym WFNs that for all `44` k-points
  unfolded with nonzero `tau`, `get_gvecs_kfull()` matches the nosym G-list as
  a set, confirming `tau` does not act on the integer G-list itself.
- Added wrapper regression coverage in
  `tests/test_symmetry_maps_nonsymmorphic.py`.
- Validation: `uv run python -m pytest -q tests/test_symmetry_maps_nonsymmorphic.py`
  passed (`4 passed`), and full `uv run python -m pytest -q` passed
  (`15 passed, 1 warning`).

## 2026-04-16 (pm): cuSOLVERMp FFI unblocked — NCCL-backed cal_comm_create works

Follow-up to the earlier "cuSOLVERMp WIP" entry.  The SIGFPE in
`cusolverMpSyevd_bufferSize` was a communicator-plumbing bug:
NVIDIA's sample passes `ncclComm_t` directly to a `cal_comm_t`-typed
API, which works under `MPI_Init` but *not* in a JAX-only C++ process
(C implicit pointer-conversion quietly becomes a bug in C++).  The
documented non-MPI CAL path — `cal_comm_create` with user
allgather/req_test/req_free callbacks — routes through NCCL cleanly.

### Result (job 51659364, nid001033, 1 node × 4×A100)

| Path | n   | type        | max \|evals − ref\| |
|------|-----|-------------|---------------------|
| cuSOLVERMp (multi-proc, NCCL)   | 128 | F64  sym    | 9.1e-13 |
| cuSOLVERMp (multi-proc, NCCL)   | 128 | C128 Herm   | 5.7e-13 |

Both on a 2×2 process grid with `NamedSharding(P('x','y'))`.

### Source changes (branch `agent/ffi-cusolvermp`, commits 22ed74a, 7c716a7)

- `src/ffi/cusolvermp/cpp/ctx.h`: add `CalNcclShim` (NCCL comm + stream
  + persistent device scratch buffer), add `cal_comm_t` field on Ctx.
- `src/ffi/cusolvermp/cpp/context.cc`: three static callbacks
  (`cal_nccl_allgather` = H2D→`ncclAllGather`→D2H→stream-sync,
  `cal_nccl_req_test/free` trivial since we're synchronous).  Replace
  `reinterpret_cast<cal_comm_t>(ncclComm)` with a real
  `cal_comm_create(params, &ctx->cal_comm)`.  Teardown extended.
- `src/common/cusolvermp_eigh_test.py`: `gather_to_numpy` now uses
  `multihost_utils.process_allgather(x, tiled=True)` so each rank
  can verify the full logical array in multi-process mode.
- `src/ffi/AGENTS.md`: mark cuSOLVERMp status as working, document the
  three required env vars and why.

### Required runtime env

```
CUSOLVERMP_FORCE_NCCL=1              # route libcal's runtime collectives via NCCL
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5   # leave headroom for cuSOLVERMp workspace
XLA_PYTHON_CLIENT_PREALLOCATE=false  # allocate on demand, not up front
```

Without the first flag, libcal's internal reduce goes through UCC and
trips `Failed to parse ib device list` in the container.  Without the
memory settings, `cudaMalloc(scratch)` inside UCC fails because JAX's
modulefile default reserves 95% of VRAM up front.

### Why this is the real scaffold for ELPA

The multi-process NCCL bootstrap is the hard part; everything else
(`XLA_FFI_DEFINE_HANDLER_SYMBOL` in C++, `jax.ffi.ffi_call` wrapped in
`shard_map` in Python, ctypes-loaded .so, bind-mounted host libs, JAX
KV-store unique-id broadcast) transfers to ELPA 1-for-1.  ELPA takes
an MPI communicator instead of NCCL; that swaps `ncclGetUniqueId` /
`ncclCommInitRank` for their MPI equivalents but does not change the
control flow.

### Report

`reports/ffi_cusolvermp_nccl_2026-04-16/report.md`.

## 2026-04-16: JAX FFI scaffolding — cuSOLVERMg eigh working on 4 GPUs; cuSOLVERMp WIP

New directory `sources/lorrax/src/ffi/` with pluggable scaffolding for
calling compiled parallel-LA libraries from JAX via the XLA FFI.  No
pybind/nanobind; the `.so` is plain C ABI loaded with `ctypes.CDLL` and
its XLA handlers wrapped via `jax.ffi.pycapsule` — the pattern from
NVIDIA's JAX FFI tutorial.

### Working — `ffi.cusolvermg` (single-process, multi-GPU)

- `src/ffi/cusolvermg/cpp/eigh_mg_ffi.cc`: XLA FFI handler that owns a
  lazy `cusolverMgHandle_t` + pairwise peer access, scatters the
  device-0 input into cuSOLVERMg's column-tile layout via
  `cudaMemcpyPeerAsync`, runs `cusolverMgSyevd` across all visible GPUs,
  and gathers `Q` back to device 0.
- `src/ffi/cusolvermg/eigh.py`: `eigh_mg(A, tile_size=32, max_gpus=0)`.
- `src/common/cusolvermg_eigh_test.py`: 4-GPU Python test.

Validation on 1 node × 4×A100 (job 51656242, nid001164):

| n    | tile | max \|evals − ref\| | wall (post-warmup) |
|------|------|---------------------|--------------------|
| 128  | 32   | 9.1e-13             | 57 ms              |
| 2048 | 256  | 2.2e-11             | 509 ms             |

Eigenvector residuals `‖A q_i − λ_i q_i‖∞` ≈ 7e-14 (F64).

### WIP — `ffi.cusolvermp` (multi-process, multi-GPU/multi-node)

Everything builds, links, and runs up to the solve.  NCCL bootstrap
works via `jax.distributed.global_state.client` KV-store broadcast of
a 128-byte `ncclUniqueId` (note: `multihost_utils.broadcast_one_to_all`
silently promotes `uint8 → uint64` under `jax_enable_x64=True` and
scrambles it — workaround is documented in the code).  `cusolverMpCreate`,
`CreateDeviceGrid`, `CreateMatrixDesc` all succeed.  `cusolverMpSyevd_bufferSize`
then SIGFPEs (integer divide-by-zero) at a constant offset inside
`libcusolverMp.so`.

Most likely cause: NVIDIA's `mp_syevd.c` sample passes `ncclComm_t`
directly to an API typed `cal_comm_t` — the C implicit pointer
conversion plus an MPI-initialised libcal recognises the wrap; our
JAX-only process never calls `MPI_Init` so libcal's NCCL-detection path
never arms, `cal_comm_get_size` returns 0, and bufferSize divides.
Preserved as branch `agent/ffi-cusolvermp`; follow-ups
documented in `src/ffi/AGENTS.md` and the report.

### Build + runtime environment

- Container: `nvcr.io/nvidia/jax:25.04-py3` (CUDA 12.9, JAX 0.5.3.dev,
  `libcusolver*`, `libcusolverMg*` in-container).
- NVHPC (for cuSOLVERMp ONLY): `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/`
  staged to `/pscratch/sd/j/jackm/lorrax_nvhpc` and bind-mounted into
  Shifter at `/lorrax_nvhpc` — the Mg path needs nothing outside the
  container.
- Build: `src/ffi/common/cpp/build.sh` via
  `src/ffi/common/cpp/run_shifter.sh`.
- `LORRAX_NTASKS` env added to `run_shifter.sh` so single-process
  multi-GPU runs (1 task × N GPUs) are as easy as multi-process
  (N tasks × 1 GPU each).

### Report

`reports/ffi_cusolvermg_2026-04-16/report.md`.

### Regression

`uv run python -m pytest -q` → 12 passed, 1 OOM failure (GPU contention
with the interactive 4-GPU alloc; not a regression).

## 2026-04-16: JAX profiling stack — skill, helpers, k-parallel run_nscf

New sandbox-level `skills/profiling_stack/` and `scripts/profiling/` that
turn an unfamiliar LORRAX module into a ranked punch-list of bottlenecks
in one command. Four categories covered: memory, compute time, sharding,
compilation.

### Deliverables
- `scripts/profiling/pf.py` — helper library (`setup_env`, `trace_profile`,
  `region`, `annotate`, `snapshot_memory`, `aot_report`, `attach_compile_log`).
  Handles jax.distributed bootstrap, JAX_ENABLE_X64 latching, and the
  per-rank perfetto-trace race that broke multi-process runs.
- `scripts/profiling/run_profiled.py` — one-shot launcher wrapping
  `python -m <module>` with the whole env (XLA_FLAGS dump, JAX_LOG_COMPILES,
  IR dump, xprof trace, pprof snapshot).
- `scripts/profiling/analyze_hlo_dump.py` — XLA dump → ranked
  `hlo_summary.{md,json}` (Memory, Compute + custom calls, Sharding
  collectives, Rematerialization warnings, Retrace groups).
- `scripts/profiling/analyze_compile_log.py` — JAX compile log → ranked
  `compile_summary.{md,json}` (wall-clock totals, cache misses by source
  location, persistent-cache misses).
- `skills/profiling_stack/` — SKILL.md (entry point) + four category docs
  (memory / compute_time / sharding / compilation) + aot_reports.md +
  cookbook.md. All docs lead with "read the ranked summaries first, drill
  into source second" — per-function inspection is the secondary tool.

### LORRAX code change — branch `agent/run-nscf-kpar` (`4617f6e`)
- `src/psp/run_nscf.py`: module-level `_maybe_init_jax_distributed()`
  (same pattern as `gw.gw_jax`); Davidson k-loop strides over
  `jax.process_index()`; `process_allgather` of evals + packed coeffs;
  only rank 0 writes WFN.h5.

### Validation — Si 2×2×2 / 60 Ry / 12 bands
- `runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/00_davidson_only/`:
  1 GPU, Davidson 7.91 s (1 rank), evals[0]=-0.418717 Ry.
- `runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/02_davidson_4gpu/`:
  4 GPU k-parallel, Davidson 6.99 s (4 ranks). **WFN.h5 bit-identical to
  1-GPU** (eigenvalue maxabs diff 0.0, coefs maxabs diff 0.0).
- Analyzer on 4-GPU run surfaces 4 collectives (all-gather-start on
  f64[1,8,12] evals + c128[1,8,12,2,2120] coeffs, 31 MiB each) — the
  expected multihost_utils payloads.
- `uv run python -m pytest -q` → 14 passed when login-node GPU not saturated.

### Report
`reports/profiling_stack_2026-04-16/report.md` — deliverables, validation,
top-3 bottlenecks found from the very first profile (memory in
`jit__apply_H_sparse`, 33 % of wallclock spent in XLA compile, 163 cache
misses localised to `solvers/davidson.py` + `psp/vnl_ops.py`).

### Next steps
- A communication-heavy smoke test (multi-GPU `gw.gw_jax`) would exercise
  the Sharding + Rematerialization view at scale — `run_nscf` is
  embarrassingly k-parallel so only holds single-digit MiB collectives.
  Waiting on direction for the next target module.
- Collapse the `jit_multiply` x58 / `jit_broadcast_in_dim` x45 retrace
  groups by wrapping the Davidson k-loop body in one outer jit (or
  `lax.scan`).

## 2026-04-16: Symmetric Si 2x2x2 failure traced to SymMaps index conflation

- Reproduced the current symmetry-path failure directly from
  `runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscf/WFN.h5`:
  `SymMaps(WFNReader(...))` raises
  `IndexError: index 8 is out of bounds for axis 0 with size 8`.
- Root cause is in `sources/lorrax/src/common/symmetry_maps.py`:
  `create_kpoint_symmetry_map()` stores **symmetry-operation indices** in
  `kpoint_map`, but `kpoint_map_irrbz_ids()` later treats those values as
  **full-/irreducible-k indices** and indexes `full_kpts[idx]`.
- For the Si `2x2x2` WFN this is fatal because `nk_full=8` but the stored
  symmetry ids include `8` and `12`; the symmetric `4x4x4` path only
  appears to survive because its mistaken symmetry ids remain `< 64`.
- Compared against BerkeleyGW `Sigma/genwf_mpi.f90` and
  `Common/find_kpt_match.f90`, which keep irreducible-k index and symmetry
  index as separate state. This is the active bug; time reversal is only a
  secondary latent concern for future TR+nonsymmorphic cases.
- Fixed on source branch `agent/symmetry-maps-fix`:
  `create_kpoint_symmetry_map()` now stores irreducible-k ids rather than
  symmetry ids, and `kpoint_map_irrbz_ids()` now validates that direct map
  instead of reinterpreting it as a full-grid index.
- Added `src/common/symmetry_test.py`, a debug checker that validates both
  atomic-position invariance under the stored spatial symmetries and full-grid
  k-point unfolding from the irreducible wedge.
- Validation:
  `uv run python -m pytest -q` → `14 passed, 1 warning`;
  `uv run python -m common.symmetry_test .../Si_pseudobands/.../WFN.h5`
  → `48/48` symmetries and `8/8` k-points valid;
  `uv run python -m common.symmetry_test .../Si/05_si_4x4x4_sym/.../WFN.h5`
  → `48/48` symmetries and `64/64` k-points valid.

## 2026-04-15: Bare Σ_X invariance analysis — ISDF quality confirmed OK

### Bare exchange is nearly invariant (17 meV shift, BGW: 0 meV)
- Added bare Σ_X diagnostic print to gw_jax.py
- Ran 4 COHSEX calculations with the diagnostic: baseline (400c, 2000c), V1 PB, V2 PB
- Result: bare X shifts only 17-20 meV with pseudobands
- Centroids don't affect bare X (400c vs 2000c identical)
- ISDF quality for exchange is acceptable

### Decomposed comparison vs BGW (using CH' = exact static, per BGW sigma_hp.log)
- LORRAX absolute X differs from BGW by 5.5 eV — nk convention (8 vs 4 k-points)
- PB screening shifts: LORRAX ΔCH ≈ -1.4 to -1.7 eV, BGW ΔCH' ≈ -1.1 to -1.8 eV — within 20%
- Baseline CH offset (LORRAX -6.77 vs BGW -8.46) is k-grid dependent: 1.7 eV at 2×2×2, 0.6 eV at 4×4×4
- No evidence of COHSEX implementation regression from recent refactors

## 2026-04-15: Pseudobands v2 (Gauss-quadrature energies) — implemented, tested, V1 still wins

Branch `agent/nscf-clean-scaffold` (+6 commits).

### New module: `solvers/pseudobands_v2.py`
- **Shifted CJ boundaries** (δ = π/2M) for quadratic POU: Σw_j² ≈ 1 ± 0.04
- **Gauss quadrature** from windowed DOS moments (Stieltjes/Jacobi algorithm)
  gives per-band energies and weights. Numerically fragile for large n_eff;
  falls back to Ritz eigenvalues + uniform weight.
- **Davidson windows**: no-matvec Galerkin from stored eigenvalues
- **n_min = k floor** prevents pathologically narrow windows
- **Window placement** with automatic n_min enforcement
- Wired into `run_nscf.py` via `pb_version = 2` in nscf.in

### COHSEX comparison (Si 2×2×2, VBM)

| Method | sigTOT (eV) | Δ from 40-band |
|:--|:--:|:--:|
| Baseline 40-band | -12.824 | — |
| **V1 hybrid PB** | **-14.145** | **-1.32** |
| V2 Gauss PB | -14.428 | -1.60 |
| V2 Ritz energies | -14.419 | -1.60 |
| BGW reference | — | -1.18 |

**V1 remains the better scheme** (-1.32 vs -1.60 excess). The v2 shifted
boundaries and different window placement create 0.3 eV more over-screening.
Energy assignment (Gauss vs Ritz) has negligible effect (< 10 meV).

### Key findings
- Dominant error: ISDF quality degradation with pseudobands (89 meV sigSX shift)
- Energy assignment is NOT the bottleneck — Gauss vs Ritz ≈ same result
- The v2 infrastructure is complete and working, but the shifted boundaries
  need further investigation to understand why they increase over-screening
- `dos_cjwindows.py` diagnostic plots CJ window indicators on the full spectrum

### Test directories (runs/Si_pseudobands/00_si_2x2x2_60Ry/)
```
11_lorrax_pb_v2_k4_40win/    — v2 k=4, 41 windows (192 bands)
12_lorrax_pb_v2_k6_60win/    — v2 k=6, 59 windows (382 bands)
13_lorrax_cohsex_v2/          — COHSEX with v2 Gauss energies
14_lorrax_pb_v2_ritz_energies/ — v2 with Ritz energies
15_lorrax_cohsex_v2_ritz/     — COHSEX with v2 Ritz energies
```

## 2026-04-15: Hybrid stochastic/CJ-Ritz pseudobands — cross-window fix

Branch `agent/nscf-clean-scaffold` (+1 commit on top of prior work).

### Architecture change
- **Hybrid pseudobands**: three construction modes per window:
  - **Stochastic**: random-phase sums of exact eigenstates (for windows
    where CJ filter can't resolve — near conduction edge).
  - **CJ-Ritz**: Chebyshev-filtered Galerkin-Ritz (high-energy windows).
  - **CJ-0**: zero-weight placeholder (spectral gaps, CJ produces garbage).
- Det bands split into "protected" (below window start, included as-is)
  and "available" (consumed by stochastic construction). Extends Davidson
  deeper (nbnd=60) to provide exact eigenstates for transition zone.

### Bug fixes
- **Window start below det max**: E_cross was 1.31 Ry but det bands
  went to 2.23 Ry. First 3-4 windows were in the det manifold — after
  deflation, CJ produced noise. Now: stochastic for those windows.
- **Zero-norm NaN**: WFNReader clamped zero norms to 1e-30, ISDF divided
  by it → 10^30 → NaN in all zeta. Fixed: clamp to 1.0 (no-op division).
- **n_protected consistency**: fixed band count across k-points by passing
  n_protected from k=0 to subsequent k-points.

### Results (Si 2×2×2, 60 Ry)
- COHSEX pseudobands shift: **-1.32 eV** (was -1.77 eV broken, BGW ref -1.18 eV)
- Excess over BGW: **0.14 eV** (was 0.59 eV — 76% reduction)
- No more NaN output, no cross-window leakage

### Next
- Investigate remaining 0.14 eV excess (ISDF quality with pseudobands)
- Test with more centroids (5000+) to separate ISDF error from PB error
- Consider global QR for CJ windows to further reduce cross-window overlap

## 2026-04-14: NSCF refactor — clean scaffold, 2D Coulomb fix, module reorganization

Branch `agent/nscf-clean-scaffold` (14 commits).

### Bug fix
- **MoS2 2D Coulomb truncation**: `compute_V_H_and_V_xc` hardcoded `truncation_2d=False`
  for V_H Poisson solve. QE's `assume_isolated='2D'` now auto-detected from XML and
  applied to both V_loc and V_H. MoS2: 594 mRy → 0.013 mRy offset. Si unchanged.

### Module reorganization
- **`src/solvers/davidson.py`**: generic eigensolver (BSE-ready). `nspinor→n_channels`,
  `n_tgt→n_eig`, `nG→dim`. `psp/davidson.py` → shim.
- **`psp/pseudos.py`**: `load_pseudopotentials`, `symbol_to_Z`, `AtomPP` extracted from
  `get_DFT_mtxels.py` (-300 lines from the kitchen sink).
- **`psp/gvec_utils.py`**: `build_master_gvec_list`, `select_gvecs_for_k`, `compute_ngkmax`,
  `reorder_to_qe` consolidated.
- **`psp/radial/`**: `radial_jax.py`, `solid_harmonics.py`, `build_projectors_qe.py`.
- **`psp/upf/`**: `load_upf.py`, `normalize.py`, `upf_model_2_0_1/`.
- **`file_io/`**: `qe_save_reader.py` + `wfn_writer.py` joined `WFNReader` et al.
- **`dft_operators.py`**: now owns `poisson_potential_from_rhoG`, `generate_gvectors_k`,
  `build_G_cart` (moved from `get_DFT_mtxels` and `charge_density`).
- **Deleted**: `kpar.py`, `get_dipole_mtxels_chunked.py`, debug functions (~750 lines).
- **Archived**: `charge_density.py` (85% dead SCF code).
- **`get_DFT_mtxels.py`**: 1281 → 974 lines.

All three entry points (`run_nscf`, `get_DFT_mtxels`, `get_dipole_mtxels`) and GW drivers
now import shared routines from canonical locations. Validated: Si 0.001 mRy, MoS2 0.013 mRy.

## 2026-04-14: NSCF driver, WFN.h5 writer, k-parallel, MoS2 validation

### New modules
- **`psp/run_nscf.py`**: Full NSCF driver (QE .save → Davidson → WFN.h5)
- **`psp/kpar.py`**: K-point parallel diag via 2D mesh ('k', 'g')  
- **`compare_wfn.py`** (sandbox): Permanent WFN.h5 comparison tool

### WFN.h5 accuracy
- **Si 4×4×4**: 33/37 fields EXACT, eigenvalues 0.0009 mRy MAE, timing competitive with QE
- **MoS2 3×3×1**: 36/37 fields EXACT (all structural, G-vectors byte-identical after QE convention matching). Eigenvalues: 2.7 mRy MAE at Gamma, 1.0 mRy at other k-points.

### Bug fixes
- **bvec.T transpose bugs**: bdot, adot, atom_crys, G_cart — all hidden by cubic Si, exposed by hexagonal MoS2. Fixed in qe_save_reader.py, wfn_writer.py, ionic_gspace.py, charge_density.py.
- **QE G-vector ordering**: Matched exactly via `(round(|G|²×1e8), g1, g2, g3)` lexicographic sort
- **nosym symmetry convention**: ntran=1, identity only, zero-padded to 48
- **scipy_erf**: Replaces jax.scipy.erf in table construction (avoids Shifter PTX crash)

### MoS2 NSCF eigenvalue discrepancy — FIXED
**Root cause**: `compute_V_H_and_V_xc` hardcoded `truncation_2d=False` for V_H Poisson solve.
QE's MoS2 input uses `assume_isolated='2D'`, applying 2D Coulomb truncation to both V_H and
V_loc. LORRAX applied it to V_loc but not V_H, causing 594 mRy offset.

**Fix** (branch `agent/nscf-2d-truncation`): Added `truncation_2d` kwarg to
`compute_V_H_and_V_xc` and threaded from `run_nscf.py`. After fix: **0.013 mRy offset,
0.002 mRy MAE-no-offset** across all 9 k-points. Si unchanged at 0.001 mRy.

## 2026-04-13: Unified ionic G-space pipeline — 195s → 31s (setup: 177s → 5s)

Three changes on branch `agent/rho-core-table-interpolate`:

1. **Unified `build_ionic_and_core`** (new `psp/ionic_gspace.py`):
   - V_loc(r) and ρ_core(r) built in one pass via shared `lax.scan` primitives
   - `species_structure_factors` + `accumulate_species_on_G` — jittable, scannable
   - Cold: 2.37s. Warm: 0.01s. Previously V_loc=1.5s + rho_core=155s.

2. **SciPy CPU table construction** (`radial_jax._spherical_hankel_table_np`):
   - Replaced JIT-compiled `spherical_hankel_table_jax` for one-time setup
   - l=1 table build: 20.27s → 0.24s (84× faster, no JIT overhead)
   - JAX version kept for gradient computations

3. **VNL table reduction** (`vnl_ops.build_vnl_setup` n_q: 50000 → 4000):
   - Linear interpolation accurate to <1e-6 Ry at dq~0.001
   - vnl_setup: 21.5s → 2.6s

Full pipeline Si 4×4×4 nosym 64 k-points: **195s → 31s** total (26s is per-k JIT).
Setup (V_loc+NLCC+VNL): **177s → 5.0s**. Eigenvalues ≤0.0001 mRy.
Branch: `agent/rho-core-table-interpolate`, commits `8e50cbc`..`3c95c63`.
- **Next**: wire `build_ionic_and_core` into `test_dft_hamiltonian.py` callers,
  consider further per-k JIT reduction, merge to main.

## 2026-04-13: Active PSP callers migrated onto unified JAX VNL path

- Switched the remaining active preprocessing callers off the old
  `projector_pipeline` execution backend:
  `psp.get_dipole_mtxels`, `psp.get_dipole_mtxels_chunked`,
  `psp.get_DFT_mtxels`, and `gw.kin_ion_io_chunked` now build one
  `vnl_ops.build_vnl_setup(...)` and use per-k
  `build_vnl_kdata_from_kvec(...)` plus dense JAX contractions for `V_NL`.
- Added canonical sparse-G helpers to `psp.dft_operators` so the active caller
  scripts share one gather / `V_NL` matrix-element path rather than
  reimplementing host-side extraction logic.
- Preserved the custom JAX radial/spline/Bessel handling in one place:
  the migration still flows through `psp.radial_jax` and `psp.vnl_ops` for
  uniform-table interpolation, derivative tables, and stable spherical-Bessel
  behaviour.
- Archived the old CPU-heavy compatibility modules under `src/psp/archive/`:
  `build_projectors.py` and `projector_pipeline.py`.
- Validation:
  `uv run python -m pytest -q` → `13 passed, 1 warning in 19.27s`
  and real sandbox smokes both completed on local GPU:
  `gw.kin_ion_io_chunked` wrote `/tmp/kin_ion_migrated_smoke.h5`
  with shape `(64, 8, 8)` in `38.769 s`, and
  `psp.get_dipole_mtxels_chunked` wrote `dipole.h5`
  with shape `(3, 64, 60, 60)` from a temp staging directory.
- Revalidated both migrated preprocessors in the documented Perlmutter
  interactive-node Shifter environment on job `51487668` so profiling stays
  comparable to earlier sandbox runs:
  `gw.kin_ion_io_chunked` completed with `Total recorded: 17.793 s`
  and `real 30.31`, while
  `psp.get_dipole_mtxels_chunked --vnl-mode analytic` completed with
  `real 49.57`.

## 2026-04-12: Unified JAX radial backend for PSP setup path

- Added a shared source backend for radial transforms:
  [src/psp/radial_jax.py](/global/u2/j/jackm/software/lorrax/src/psp/radial_jax.py:1).
  This now owns the common spherical-Bessel kernels, uniform radial tables,
  interpolation, and radial integration weights used to form `V_NL`, `V_loc`,
  and NLCC/core charge.
- Switched the active production builders away from the old SciPy spline path:
  `vnl_ops.build_vnl_setup(...)`,
  `build_projectors_qe.build_local_ionic_potential_on_G_total(...)`, and
  `charge_density.build_core_density(...)` now all use the shared JAX/table
  backend.
- Simplified the autodiff `V_NL` channel extraction path in
  `dft_operators.py` so it consumes the same uniform tables rather than SciPy
  spline internals.
- Removed a duplicate spherical-Bessel implementation from
  `projector_pipeline.py` by importing the shared backend instead.
- Validation:
  `uv run python -m pytest -q` → `13 passed, 1 warning in 15.24s`
  and the canonical Si DFT-H reproducer still passes with
  `Max MAE: 0.0001 mRy = 0.00 meV`.
- Measured canonical launcher wall time after the refactor:
  `/usr/bin/time -p ./launch_test_dft_hamiltonian.sh` →
  `real 25.67`, `user 0.05`, `sys 0.04`.
- Followed up with a terminology cleanup in the active path so plan/bundle
  fields now prefer `radial_tables` over `splines`, reducing conceptual drift
  after the backend swap.
- Added report:
  [reports/jax_unified_psp_radial_2026-04-12/report.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/reports/jax_unified_psp_radial_2026-04-12/report.md)

## 2026-04-12: Standalone psp DFT-H validation now documented and runnable

- Fast-forwarded `sources/lorrax` again from `f7bc2e2` to `273a7d8`, picking up
  the new upstream reproducer `src/psp/tests/test_dft_hamiltonian.py` and the
  expanded `src/psp/dev_status.md`.
- Logged a new sandbox mismatch in `KNOWN_SANDBOX_ERRORS.md`: the local
  `runs/Si/04_si_4x4x4_davidson/00_davidson/` helper scripts still pointed at
  deleted `psp` setup helpers, so they were no longer a valid entrypoint.
- Added a sandbox-side canonical entrypoint for the standalone DFT path:
  [runs/Si/04_si_4x4x4_davidson/00_davidson/README.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/runs/Si/04_si_4x4x4_davidson/00_davidson/README.md)
  and
  [runs/Si/04_si_4x4x4_davidson/00_davidson/launch_test_dft_hamiltonian.sh](/global/homes/j/jackm/scratchperl/lorrax_sandbox/runs/Si/04_si_4x4x4_davidson/00_davidson/launch_test_dft_hamiltonian.sh),
  both using this sandbox's real paths and the Shifter environment that
  includes `$SANDBOX/sources` for `jax_xc_local`.
- First launcher run exposed a real upstream test bug: `test_dft_hamiltonian.py`
  passed `CrystalData` into `vnl_ops.build_vnl_setup(...)`, but the current
  implementation needs the `WFNReader` for its k-dependent G-vector scan.
  Patched locally on source branch `agent/test-dft-hamiltonian-fix`.
- Re-ran the canonical test on interactive job `51470500` and obtained:
  `Max MAE: 0.0000 mRy = 0.00 meV`
  and
  `PASS: all k-points match QE to < 0.01 mRy`
  for all 8 Si `4x4x4` IBZ k-points.
- Added report:
  [reports/dft_hamiltonian_validation_2026-04-12/report.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/reports/dft_hamiltonian_validation_2026-04-12/report.md)

## 2026-04-12: Si 4x4x4 no-sym COHSEX output-format rerun

- Created `runs/Si/02_si_4x4x4_nosym/16_lorrax_cohsex_rerun_4gpu_repeat/` as a fresh clone of variant `15` and reran GWJAX on interactive job `51470500` (1 node / 4 GPUs) so the updated logging/output-writing behavior would land in a new `gw.out` without overwriting prior outputs.
- Run completed end to end in `26.661 s`; artifacts written successfully: `gw.out`, `eqp0.dat`, `qp_wfn_rotations.h5`, and `tmp/isdf_tensors_480.h5`.
- The new `gw.out` differs materially from variant `15`: no initial `srun` step line, denser chunked-ISDF setup summary, progress-bar style zeta/V_q status lines, a new `STATIC HEAD TERMS` block, and inline XLA rematerialization warnings captured in the file.
- `eqp0.dat` from variant `16` is not byte-identical to variant `15`, so this should be treated as more than a cosmetic logging-only rerun.

## 2026-04-12: Housekeeping sync

- Fast-forwarded `sources/lorrax` on local `main` from `b0b02f9` to `f7bc2e2` to match `origin/main`.
- Logged a sandbox inconsistency in `KNOWN_SANDBOX_ERRORS.md`: the newest report directory (`reports/mos2_kgrid_gnppm_head_convergence_2026-4-10/`) does not contain the documented `report.md`.
- Added sandbox-local `jax_xc_local` wiring for the standalone `psp` DFT path:
  `sources/jax_xc_local -> /global/u2/j/jackm/software/jax_xc_local_lorrax_sandbox`
  and `sources/jax_xc -> /global/u2/j/jackm/software/jax_xc`.
  Verified `jax_xc_local.pbe` and `psp.dft_operators.compute_V_H_and_V_xc` import and execute under the documented Shifter flow when `PYTHONPATH` includes `$SANDBOX/sources`.
- Pulled the current Si Davidson/NSCF test drivers from `../lorrax_sandbox_fresh` into
  `runs/Si/04_si_4x4x4_davidson/00_davidson/` and updated `run_direct_diag_v2.py`
  to the current `origin/main` `psp` API (`setup_H_k`, `build_matrix_k`, `vnl_ops.build_vnl_setup`).
- First live Perlmutter/Shifter validation now works end-to-end for the direct-diag rung.
  `run_direct_diag_v2.py` reaches all 8 IBZ k-points and reports:
  diagonalized occupied-band MAE `94.890 mRy`, offset `-94.890 mRy`, MAE-no-offset
  `19.943 mRy`, max `153.478 mRy`. Nontrivial k-points show `H` non-Hermitian warnings
  (`~1e-4` to `4.5e-4`) and pathological Rayleigh quotients, which is the clearest
  current testing signal before Davidson wall-time work.

## 2026-04-12: Major code clarity refactor

Session focused on making gw_jax.py main() read like a physics outline.

**Screening pipeline surfaced at top level:**
- `compute_chi0(wfns, quad, meta, mesh_xy)` and `solve_w(V_q, chi0_q, meta, mesh_xy)` now visible in main() for both COHSEX and PPM paths
- `build_static_quadrature` / `build_imag_quadrature` are clean one-liners for quadrature setup
- `fit_gn_ppm(W_q, Wiwp_q, V_q, omega_p, mesh_xy)` extracted from monolithic PPM builder

**ppm_sigma.py (-347 lines):**
- PPM arrays stored as flat-q (nq,μ,μ) — eliminated transpose round-trip
- Fixed _mu_nu_sharding (was 5D for dead k-last layout)
- Fixed _build_single_sigma_window missing mask_B args (would crash on kernel_sign=-1)
- Stripped all profiling boilerplate; replaced verbose prints with per-window summary
- _convolve_sigma_branch_kij takes wfns bundle (28→22 params)

**gw_jax.py (-267 lines from ISDF move, +gw_output.py):**
- ISDF pipeline moved to gw_init.py (fixes circular import), split: fit_zeta + compute_V_q
- Output formatting extracted to gw_output.py (GWResults dataclass + write_results)
- V_q/W_q naming used consistently everywhere (no more bare V/W aliases)
- solve_w_from_chi_q_jax → solve_w; print0= → print_fn= standardized

**w_isdf.py:**
- Fixed chi0 accumulator sharding for non-divisible k-grids: P(None,'x','y')
- Fixed Dyson solve padding order (pad before reshard)
- Both verified on 4×A100 with MoS2 3×3 (nk=9)

All changes GPU-regression-tested (MoS2 3×3 COHSEX, 4×A100-40GB, bit-identical).
COHSEX chi0_W timing dropped from 2.7s→1.7s (old path computed unnecessary PPM head terms).

## 2026-04-09: GWJAX pipeline refactor status

Primary initiative: remove non-jitted stages, eliminate incorrect host/replicated
materializations, and make the active no-symmetry GWJAX pipeline safe on multi-GPU
Si `4x4x4` and `10x10x10`.


## Current status

What is now in good shape:
- head corrections for sigma_{X,static SX-X/CH, GN-PPM cor}
- active multi-GPU minimax screening path
- active GN-PPM fit path
- active dynamic sigma path
- post-PPM tail safety on `10x10x10`
- one process per GPU execution

What still looks worth improving:
- `compute_sigma_c_ppm_omega_grid` dominates runtime on large grids
- post-PPM fixed-point / QSGW work is safer now, but not yet distributed over
  band tiles on the `XY` mesh. This is a significant issue.
- likely next architectural step is a band-sharded `sigma_mnk.h5` / post-PPM
  path over `(omega, k, m_X, n_Y)`

## Known environment notes

- For multi-GPU GWJAX on Perlmutter, use Shifter, not `uv run`.
- *Keep one MPI rank per GPU. Do not ever run one mpi rank per node with 4 GPUs or so forth.*

## 2026-07-01 — Regression-gate audit (lorrax_D, agent/docs-tighten == main e7b6c7d)
- Full audit: `reports/gw_refactor_map_2026-07-01/archive/GATE_AUDIT.md` (inventory, GPU run, coverage matrix, golden-gate recommendations).
- Suite run on 1×A100 (pool `lx-alloc-jackm`): **5 failed / 250 passed / 20 skipped in 10:48**.
- **The COHSEX e2e gate is RED on main**, twice over: (1) driver crashes in one-shot `write_qp_wfn_h5` (full-BZ U (9,30,30) vs wfn.nkpts=4; `debug.write_wfn_h5` default true) before the eqp compare; (2) with the writer disabled, sigSX/sigCOH drifted vs `eqp_ref.dat` (MAE 3.5 meV, max 12.5 meV sigTOT, all 270 rows > tol, VH exact) — plateau-shaped, W-side; candidates fc1602a/882ed4a. Needs bisect + either fix or reference re-freeze.
- Other 4 failures are environment mismatch (container JAX 0.5.3 vs pyproject jax>=0.9: reshard `jax.jit` kwargs form; aot_memory libcufft probe ×3).
- Coverage: only e2e value gate = static COHSEX / charge / 2D / single-GPU. GN-PPM, HL-PPM, real-axis Σ_C, bispinor Σ values, 3D, head-off, SC loop, multi-GPU driver: all ungated (unit layer strong on ζ-fit/V_q/unfolds). Recommended 7 golden gates with existing seeds under `runs/` — see report §4.
- Sandbox: base lorrax_{A..D} modulefiles' `LORRAX_FFI_*` defaults point at purged $SCRATCH paths → logged in KNOWN_SANDBOX_ERRORS.md (workaround: env overrides to $HOME/software before module load).

## 2026-07-10 — SLATE build hardening: scripted GPU + CPU builds (P2 of slate-linalg-ffi)
- New `src/ffi/slate/scripts/build_perlmutter.sh {gpu|cpu}`: reproducible SLATE builds under `$HOME/software/slate_builds/{gpu,cpu}/install`, source pinned @ v2025.05.28-1 (same as the untouched `$HOME/software/slate` eval build). GPU = cuda/sm_80 (PrgEnv-gnu + cudatoolkit/12.9 + craype-accel-nvidia80 + libsci); CPU = `gpu_backend=none`, CUDA modules explicitly unloaded so no `libcuda.so.1`/GTL in the link. The `-DSCALAPACK_LIBRARIES=""` gotcha is now explained in-script (ScaLAPACK lives inside wrapper-linked libsci; empty keeps the tester's `--ref` checks).
- Validation (all pass, machine precision): SLATE tester potrf/trsm/heev on GPU node (target=devices, 2×2 grid, `--ref=y` vs libsci ScaLAPACK) and on Milan CPU node (`none` build, target=host). Cuda build also runs host-target on both node types (Perlmutter CPU nodes ship libcuda.so.1 — site quirk, don't rely on it).
- FFI rebuilt against the new gpu install in a SEPARATE build dir (`LORRAX_FFI_BUILD_DIR` knob added to `build.sh`; runtime override `LORRAX_FFI_SO`): `common.slate_cholesky_trsm_test` + `slate_batched_test` PASS at ~1e-16 on 2×2. In-tree `build/` .so untouched.
- CPU verdict for the FFI: GPU-only today (CUDA-typed handlers, `platform="CUDA"` registration, .so dlopen fails at libnccl on CPU nodes). Port = host handler variants (`fromScaLAPACK` + HostTask) + CUDA-free `liblorrax_ffi_host.so` + per-platform loader, est. 1–2 days — documented in `reports/slate_linalg_ffi_2026-07-10/report.md` §P2, deferred.
- Sandbox: login-node shifter bind-mount transient failure logged in KNOWN_SANDBOX_ERRORS.md (workaround: build via compute node).

## 2026-07-11 — CPU-backend distributed-linalg timing matrix (lorrax_C agent/ffi-host-platform, MoS2 3×3 bispinor GN-PPM, 4 MPI ranks 2×2, Milan)
- Extends `runs/MoS2/C_bispinor_backend_timing_2026-07-11` with CPU variants 03–07 (own allocation JID 55793024). All ran e2e via the **shifter container on the CPU node** (raw srun + `--module=mpich`, `JAX_PLATFORMS=cpu`, `MPICH_GPU_SUPPORT_ENABLED=0`, `LORRAX_FORCE_FULL_BZ=1`); the documented native-venv CPU path (§3.5 / lxrun `LORRAX_PARTITION=cpu`) is **broken for multi-rank GN-PPM** under venv jax 0.9.1 — `process_allgather(tiled=False)` at `gw/minimax_screening.py:44` raises `ValueError: ... only supports tiled=True` in `fit_ppm` (after all ζ fits). Logged in KNOWN_SANDBOX_ERRORS.md with the working container recipe.
- Wall times (≈identical; solver cells are tiny at this scale): 03 intree 273 s, 04 slate 274 s, 05 scalapack 274 s, 06 slate+scalapack 271 s, 07 slate-repeat 272 s (GPU equivalents: 73–80 s). Dominant cost is `zeta_fit.chunk.z_q_build` (~44 s ×4 channels); charge cholesky itself: sharded 0.73 s vs slate host 1.56 s; transverse solve: per-q lu 4.4 s vs scalapack 3.1 s.
- **Correctness (vs 03 baseline, `shared/diff_sigma.py`)**: 05 scalapack_lu = **0.000e+00** (bit-identical to per-q lu — host ScaLAPACK LU validated); 07 slate-repeat = **0.000e+00**; but **04 = 9.484e-01 and 06 = 6.523e-01 eV (04 vs 06 = 1.598)** — the HOST SLATE cholesky (`Target::HostTask`, `liblorrax_ffi_host.so`) **intermittently corrupts exactly one q tile** (q idx 5: GN invalid-mode count 1372 healthy vs 11554/64486 corrupted; sigX identical, only screened W/sigC affected). GPU slate (variant 02) was bit-identical, and 1 of 3 host-slate runs (07) is clean → host-handler-specific flake (suspect threading/race in the host potrf/trsm batched loop), NOT a systematic layout bug. `distributed_cholesky=slate` on CPU should be considered UNRELIABLE until root-caused; `distributed_lu=scalapack` is good.
- mpi4py absent from every reachable env → `use_ffi_io=true` on CPU always falls back to `H5PY_ALLGATHER` (uniform across variants, io-only).
