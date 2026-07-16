# LORRAX BSE — refactor MAP (the spine)

> The BSE analogue of `reports/gw_refactor_map_2026-07-01/MAP.md`. This is the map a
> model should load to orient before touching BSE code: (0) the taxonomy, (1) the
> dataflow spine, (2) the concern×stage matrix, (3) the coupling cross-map, (4) the
> ranked refactor targets, (5) bug + gate reality, (6) the five missing-feature
> designs, (7) the recommended attack order.

Synthesized 2026-07-15 from an 18-agent deep-read pass over `src/bse/` (~7.7 kL) +
`src/solvers/` (~3.9 kL) + 207 adversarial verdicts (`archive/_raw_verdicts.json`)
+ two independent teleological sortings (`archive/_raw_sorts.json`) + 5 feature
designs (`archive/designs/*.md`). Checkout: `sources/lorrax_D`. Audit base
`e18d0e5` (`agent/slate-linalg-ffi`); the BSE+solvers trees are **byte-identical**
to current HEAD (`git diff e18d0e5..HEAD -- src/bse src/solvers` is empty) — the
BSE package was untouched by the entire GW refactor program. Per-file detail:
`archive/files/*.md`. Feature catalog: `archive/FEATURES.md`. Bug/dead ledger:
`archive/DEAD_CODE.md`.

**Headline state**: the BSE stack is one generation behind the GW refactor on
every axis it shares with it — and it has two showstopper findings the GW program
never had: (1) **every load path is broken against restarts written by current
gw_jax** (§5, B3/B4/B5 — the package only runs against legacy-era h5 files), and
(2) **the exchange kernel is k-block-diagonal in all four matvec implementations**
where the physical kernel is dense in (k,k′) (§5, B1 — triple-confirmed, including
a 2-k numerical diff against a hand-built dense reference). There is **zero
pytest-collected BSE coverage** (§5). Fix order matters: gates → loaders → physics
→ deletes → moves (§7).

---

## 0. Taxonomy (merged from both sortings)

Two independent sorters (26 and 19 categories) agree almost cell-for-cell; merged:

**Tier A — pipeline stages** (dataflow order):
`A0` GW-side producers (restart bundle, W0+heads, dipole.h5) · `A1` Ingest
(bse_io loaders, eqp, n_occ, head injection) · `A2` H·X matvec (kernel-apply) ·
`A3` Eigen/spectral solve · `A4` Spectra (ε₂/JDOS) · `A5` Output writers (BGW-spec)

*Architectural note (sorter 2): LORRAX BSE has **no separate "kernel build" stage**
— V_q0/W_q are ingested once (A1) and applied on the fly inside every matvec (A2).
There is no assembled-H object anywhere; "kernel build" and "matvec" are fused by
construction. This is the right shape for matrix-free fine-grid work and should be
preserved.*

**Tier B — variant axes** (cut across A-stages):
`B-TDA` TDA vs full-BSE (non-TDA exists only as `build_bse_ring_matvec_full`; its
W-including B-block has never executed — §5 B2) · `B-MV` matvec kind
(simple|ring|gather|serial) · `B-W` W source (W0_qmunu; **silent fallback to bare
V_qmunu** when `W0_ready=False`; Casida diagnostics + pseudopole W_c(ω) thread) ·
`B-EQP` energy source (DFT enk_full vs BGW eqp1.dat via SymMaps unfold) ·
`B-FUT` unbuilt axes with designs: fine-grid interpolation, finite-Q (§6)

**Tier C — infrastructure**:
`make_bse_shardings`/`create_mesh_2d` (bse_ring_comm.py:31-63 — the ONE sharding
vocabulary, 13 importers, top extraction target) · `common.fft_helpers`
sharded FFT · `common.symmetry_maps.SymMaps` · `runtime.padding.padded_mu_extent` ·
generic `solvers/` (lanczos, davidson, chebyshev, dos, quadrature) ·
`file_io.WfnLoader` + `runtime.init_jax_distributed`.
*Note: half of `solvers/` (pseudobands*, sternheimer_*, minres, projectors) serves
only `psp/`, not bse/ — foreign territory with no subpackage boundary to signal it.*

**Tier D — config surface**: 14 separate argparse surfaces, no shared config
object, and the only cohsex.in keys BSE reads (`wfn_file`, `vhead`, `whead_0freq`)
go through a **private parser in bse_io.py** that bypasses the canonical
`gw_config.read_lorrax_input`. The BSE-side twin of GW's old shadow-env-flag
disorder. Full flag catalog: `archive/files/flags_surface.md`.

**Tier E — diagnostics/sweeps**: feast_{sweep,zolo_sweep,ellipse_mixed_sweep},
bse_feast_dense_debug, pseudopoles_sweep, ring smoke/check CLIs,
`_main_random_demo`, test_bse/test_davidson_bse (CLI smoke scripts — NOT pytest),
`context/` design notes.

**Tier F — dead** (verified, `archive/DEAD_CODE.md §2`): the bse_jax module-level
matvec trio (67-160, uncallable `lax.psum` outside shard_map),
`apply_bse_hamiltonian_ring`, `symmetrize_W_q`, `BSEData`, 7 of 8
bse_preconditioner exports, test_bse's stale restart loader,
write_eigenvectors.py (soft-dead: superseded by the streaming writer, kept alive
only by test_bse).

---

## 1. Dataflow spine

Full trace with boundary-array table, function tables, and coarse-k bake-in
points: **`archive/files/kernel_dataflow_trace.md`** (the §1-equivalent document;
read it before touching any A-stage). Compressed:

```
A0 GW SIDE   gw_init.setup_isdf_tensors → tmp/isdf_tensors_{n_rmu}.h5
             (V_qmunu flat-q, psi_full_y = ψ at centroids (nk,nb,ns,μ), enk_full,
              G0_mu_nu = ζ(0,μ,G=0), kgrid; W0_qmunu ← persist_w0_and_head after
              compute_screening, + vhead/whead scalars)
             psp.get_dipole_mtxels → dipole.h5 (spectra only)
      ▼  boundary: the restart h5 IS the gw↔bse module boundary (no isdf/ imports)
A1 INGEST    bse_io: sharded loader (≥2 dev) | ring loader (1 dev)
             n_occ: --n-occ ▸ WFN ifmax ; val_idx v=0 = DEEPEST valence (internal)
             eqp1.dat (eV, IBZ) → SymMaps unfold → Ry full-BZ
             q→0 head: rank-1 (head/V_cell)·g0*⊗g0 via gw.head_correction (×2, dup)
      ▼  data dict: psi_{c,v}_{X,Y}, eps_c/v, V_q0 P('x','y'), W_q P('x','y',None³), g0_X/Y
A2 MATVEC    H X = D X + V X − W X   (TDA, Ry; W_R = ifftₖ(W_q) once per solve)
             D = (ε_c−ε_v)·X ; V = (1/Nk) M† V_q0 M X  [k-DIAGONAL — bug B1] ;
             W = FFT-convolution over q=k−k′ of ψψ*-encoded T against W_μν(q)
             kinds: simple (jit+einsum) | ring (ppermute) | gather | serial (1-dev ref)
             non-TDA: build_bse_ring_matvec_full S=[[A,B],[−B†,−A†]]  [B-block: bug B2]
      ▼
A3 SOLVE     default → bse_feast (FEAST windows+GMRES+Ritz; n_ritz=4 FIXED)
             --lanczos → solve_bse_sharded (block-Lanczos) ; --solver davidson
             --kpm-dos → bse_kpm (broken at HEAD: phantom kwarg)
             1 device → solve_bse (CLI knobs silently dropped — C2)
      ▼  eigenvalues (Ry, replicated), eigvecs (n_eig,1,nc,nv,nk)
A4 SPECTRA   absorption_haydock (CF recursion; ≥2-device requirement — C2) |
             absorption_eigvecs (sum-over-states) | eigvals_to_eps2 | davidson_absorption
A5 OUTPUT    bse_io.write_eigenvectors_stream → eigenvectors.h5 (BGW spec: eV,
             valence axis FLIPPED on write, use_tda=1 hardcoded)
             absorption_*_{b1,b2,b3}_eh.dat, eigenvalues_b*.dat (BGW 4-col formats)
```

Coarse-k bake-in points (where fine-grid interpolation must hook — trace §"bake-in"):
ψ exists only at coarse k AND only at centroids r_μ; nq ≡ nkx·nky·nkz enforced;
W(k−k′) as 3-D FFT over exactly the coarse grid; D from enk_full[:,idx]; dipole.h5
at coarse nk; **implicit global contract** that psi_full_y k-order == flat-q order
== C-order MP grid (silently load-bearing in the FFT convolution).

---

## 2. Concern × stage matrix

| Concern ↓ / Stage → | A1 ingest | A2 matvec | A3 solve | A4 spectra | A5 output |
|---|---|---|---|---|---|
| **B-TDA** | — | 4 TDA impls + 1 non-TDA fork (B-block broken) | FEAST/KPM non-TDA contours (RPA-only in practice) | TDA-only | `use_tda=1` hardcoded |
| **B-W source** | `W0_ready` silent V-fallback (bse_io:376,906) | W_R built per matvec factory (×3 dup) | bse_feast mutates `data["W_R"]` | — | — |
| **q→0 head** | rank-1 inject, **duplicated** sharded(463-513)/ring(804-836); cohsex.in overrides | — | — | vhead in ε₂ prefactor conventions | — |
| **B-EQP** | `apply_eqp_corrections` (SymMaps) | — | CLI eqp block **×3 dup** (B7: 2 of 3 copies buggy); FEAST/KPM get **no** --eqp (C3) | haydock copy shares B7 | — |
| **C sharding/pad** | μ re-pad on read; `_assert_local_block` | `make_bse_shardings` (ONE home, wrong file) | trial-X sized with **unpadded** counts (B6) | haydock ≥2-dev | streamed per-vector |
| **D config** | private cohsex.in parser | — | 14 argparse surfaces; 1-dev knob-drop (C2); FEAST/RPA default routing (C1) | own argparse ×4 | — |
| **BGW-compat** | valence-flip on READ (absorption_common:63) | — | — | prefactors = absh.f90 | flip on WRITE, Ry→eV |

The most-smeared concerns: **restart-layout knowledge** (three on-disk generations,
each reader/injector knowing a different subset — the direct cause of B3/B4/B5) and
**the eqp/band-window resolution block** (×3, two copies wrong). These two plus the
sharding-vocabulary extraction drive the refactor.

---

## 3. Coupling cross-map (contracts to preserve)

| A ↔ B | Mechanism |
|---|---|
| GW → BSE | `tmp/isdf_tensors_*.h5` written by `tagged_arrays.write_restart_state_to_h5` + `gw_output.persist_w0_and_head`. **The h5 is the entire module boundary** — bse/ imports nothing from isdf/ and only `head_correction.apply_q0_head_rank1{,_sharded}` from gw/. Layout is generation-sensitive (8-D → 6-D → rank-3 flat-q); readers must normalise FIRST, then operate (B5's lesson). |
| C sharding ↔ A1/A2/A3 | `make_bse_shardings`/`create_mesh_2d` (bse_ring_comm:31-63): 13 importers. X = P(None,'x','y',None) (c on x, v on y, k replicated); ψ kept in TWO copies (μ on x / ν on y); W_q k-axes replicated so ifftₖ is device-local. |
| B4 sym ↔ A1 | `SymMaps.irr_idx_k` is the only sym consumer (eqp IBZ→full unfold). Fine-grid/finite-Q designs route ALL new sym needs through the same canonical table (feedback_unified_sym_action). |
| k-order contract | psi_full_y k-order == V/W flat-q order == C-order MP grid (`generate_kpts_grid`). Nothing asserts it; the FFT convolution silently depends on it. Worth an explicit check at load. |
| solvers ↔ bse | matvec-closure + shape contract (`solvers.lanczos`, `solvers.davidson` are physics-blind). The solver program (§6) keeps this seam and makes it dim-agnostic for fine grids. |
| A3 → A5 | eigvecs replicated then streamed per-vector to h5; valence flip + Ry→eV happen ONLY at file boundaries (internal order: v=0 deepest). |

---

## 4. Ranked refactor targets (misfits; both sorters agree)

1. **`bse_io.py` (932 L)** — A1 ingest + A5 writer + D private config parsing + F
   dead, and the site of B3/B4/B5/B6. Fix = repair + split: one flat-q-normalising
   loader (both device paths share head-injection and layout code), writers out,
   config through `gw_config`. The **first** file to touch (§7 step 1).
2. **`bse_ring_comm.py` (996 L)** — five tiers in one file. Extract
   `create_mesh_2d`/`make_bse_shardings` to a C-tier home (highest fan-out symbol
   in the package); move ring smoke/check CLIs to diagnostics; delete the 2 dead
   functions; dedup the verbatim T-encode/FFT/decode blocks shared with `_full`.
3. **`bse_jax.py` (626 L)** — delete the dead matvec trio (67-160, sits where a
   reader mistakes it for the live path); move demo/ring-test dispatch out; fix
   the default-routing traps (C1: bare invocation = FEAST + **RPA** kernel) and the
   1-device knob-drop (C2); forward `--eqp`/`--n-occ` to FEAST/KPM (C3).
4. **Redundancy pass** (feedback_no_redundancy): `compute_pair_amplitude` ×3,
   `apply_D` ×2+inline ×3, eqp-override block ×3 (one correct copy —
   davidson_absorption's), head-injection ×2, T-encode ×2, restart-loader ×2
   (test_bse's is stale), eigvec writer ×2.
5. **Solver consolidation** — `archive/designs/solver_program.md` P1-P3: fix KPM
   phantom kwarg + lanczos transposed-β + final-slot overwrite; 6 Lanczos variants
   → 1 block + 1 converged; FEAST → `solvers/feast.py` with DOS-sized subspace +
   tr(P) count + locking (the clustered-eigenvalue fix); archive the 3 sweep
   harnesses. Net **−800…−1000 L**.
6. **`bse_preconditioner.py`** — keep `energy_diff_cv_k` (3 lines, 3 importers),
   delete the other 7 exports; **`davidson_absorption.py`** — collapse into
   bse_jax dispatch + one output path; **`write_eigenvectors.py`** — delete after
   repointing test_bse/kpts helper.
7. **Config** — one parsed bundle (procedural, no class hierarchy) reused across
   the CLIs; cohsex.in keys via `gw_config.read_lorrax_input` only.

---

## 5. Bug + gate reality

**Bugs** — 88 confirmed of 207 claims (`archive/DEAD_CODE.md §1` — read §1.1
first; per-claim verifier reasoning in `_raw_verdicts.json`). The structural ones:

- **B1 — exchange kernel k-block-diagonal in ALL four matvecs** (serial, simple,
  ring, dead-jax-copy). Physical K^x (Rohlfing-Louie; BGW bsex loops every
  (ik,ikp)) is dense in (k,k′); the code batches k end-to-end
  (`'kcvN,bcvk->bNk'`). Triple-confirmed: Henneke Eq. 4-5 derivation, a 2-k toy
  numerically diffed against a hand-built dense sum, and BGW `kernel_main.f90`
  cross-check. Survived validation because Si's lowest manifold is
  exchange-insensitive and `ring_matvec_correctness_check` compares ring vs serial
  — the same wrong formula on both sides. **No independent dense reference exists
  in the tree.** Fix is its own PR, gated (below), before any fine-grid/finite-Q work.
  **ADJUDICATED 2026-07-15 — DENSE, unanimous** (owner challenged the finding; a
  3-agent adversarial round — k-diagonal steelman FAILED, dense steelman, and an
  empirical dump of BGW's own bsemat.h5 showing off-diagonal k-block exchange the
  same order as the diagonal — settled it; the owner's momentum-shift-0 point is
  correct for the interaction line and orthogonal to δ_kk′; at finite Q the
  kernel is Q-block-diagonal and dense within each block, v(Q+G) keeps G=0).
  See `archive/adjudication/VERDICT.md`. Fix greenlit — PLAN.md Phase 2.
- **B2 — non-TDA B-block einsum malformed** (`'kvsM,bvksN->bMNtsk'`, output `t`
  unbound; bse_ring_comm:183,551) → full-BSE-with-W crashes at first apply; only
  ever ran as RPA. (Fix: `kvtM`.)
- **B3/B4/B5 — every load path broken vs current writers**: `_read_vq0_sharded`
  indexes 8-D legacy layout only; `_read_wq_sharded` flat-q branch demands a
  `kgrid` attr no writer sets; 1-device head injection indexes 8-D before the
  layout shim. Net: sharded path dies at B3→B4, 1-device at B5, on ANY fresh
  `do_screened` restart. The package currently runs only against legacy-era files.
- **B6 — `n_val_pad`/`n_cond_pad` carry UNPADDED counts** → trace-time crash for
  band counts not divisible by the mesh (every validated run used divisible
  counts); once fixed, zero-padded ε needs masking (KPM masks; Lanczos doesn't).
- **B7 — eqp band-window duplication**: 2 of 3 copies use raw CLI counts, not
  loader-clamped; `--n-val` too large silently wraps to TOP bands.
- **Solver layer**: bse_kpm broken at HEAD (phantom `v_couples_k` kwarg —
  TypeError on every entry); `block_lanczos_eig` transposed-β; all 3 jit Lanczos
  variants overwrite the final Krylov slot; FEAST `n_ritz=4` fixed with no
  count-in-window estimate (the clustered-eigenvalue failure, ~100 states/0.5 eV);
  bse_pseudopoles un-importable (**lost wiring** from the package consolidation —
  a roadmap feature, NOT dead; cf. feedback_parsed_unread_not_dead).
- **W0 silent fallback**: `W0_ready=False` ⇒ direct kernel silently uses bare V
  (unscreened) — should loud-fail or warn.

**Gate reality** (`archive/files/tests_gates.md`): **zero pytest-collected BSE
tests** — `testpaths=["tests"]` excludes src/bse; the only collected import is
`read_bgw_eqp` (tests/test_eqp_bgw.py). test_bse/test_davidson_bse are `-m` CLI
smoke scripts; test_bse's loader reads datasets the writer no longer produces.
The one real validation is the manual Si 4×4×4 SOC vs BGW run (STATUS.md, ~3 meV /
1.5% Haydock) — on legacy restarts, exchange-insensitive.

**Gate wall to build before moving code** (all 1-GPU, per feedback_no_16gpu_gating):
1. **Dense-reference kernel gate**: tiny fixture (2-k toy and/or the N=144
   4v4c 3×3×1 case with exact eigenvalues in `context/feast_accuracy_notes.md`) —
   dense H built by direct quadrature vs every matvec kind; catches B1-class bugs
   structurally. 2. **Fresh-restart e2e gate**: pytest-collected, runs gw_jax on
   the existing MoS2 regression fixture with `do_screened`, then BSE lowest-n_eig
   on the restart it just wrote (catches B3/B4/B5-class). 3. **BGW anchor**: Si
   eigenvalues ≤3 meV + Σ|d|² machine-match + Haydock ε₂ overlay. 4. **Pad-mask
   gate**: non-divisible band counts on a 2×2 mesh (B6). 5. **KPM/solver unit
   gates** (solver_program.md §Gates).

---

## 6. The five missing-feature designs (`archive/designs/`)

| Design | Headline | LOC | Depends on |
|---|---|---|---|
| `fine_grid_interpolation.md` | **No dcc/dvv for ψ**: centroids are k-independent real-space points, so fine pair amplitudes are EXACT by sampling a fine-NSCF ψ at the same r_μ (cheaper + more accurate than BGW's overlap scheme; dcc/dvv survives only as optional eqp interpolator). W interpolated via its real-space image W_μν(R) (q-independent μν basis ⇒ one FFT), exact at coarse q; divergent head/wing stripped and re-added analytically per fine q. TDA first. Never materialise fine-q W. | ~510-570 | B1 fix (P0); sr_lr P2-3; head_wings P2 |
| `coulomb_sr_lr.md` | Gaussian/erfc range separation v = v_SR + v_LR at the ONE per-G seam (`get_kernel(sys_dim).v_qG`), so V^SR/W^SR interpolate cleanly and v_LR re-adds analytically. W split by physical singularity: W^LR = ε₀₀⁻¹·V^LR. Head scalars unchanged, only re-partitioned. Phase 1 (α=None) is bit-identical inert scaffolding that also kills the current per-dim if/elif split-brain. BGW's head-only exclusion = the α→∞ limit. | ~430 net | none (P1 ships alone) |
| `w_head_wings_interp.md` | Promote `gw/experimental/head_wing_schur.py` (currently zero-caller) to production `gw/head_wing.py` as THE single head/wing/body path; delete `apply_q0_head_rank1{,_sharded}` (scalar rank-1 = the A_wing=0 special case). Reuse the anisotropic S_cart(3,3) from `chi_from_dipole` verbatim. Wings only become load-bearing on refined q-grids (coarse q=0 wing averages to zero — why head-only matched Si to 3 meV). Phase 1 is behaviour-preserving. NB: BGW's `fixwings` is Σ-only; the BSE wing path is mtxel_kernel.calc_wings + intkernel re-add. | ~700-750 | Phase 1: none; Phase 2: fine-grid |
| `solver_program.md` | Keep FEAST (interior) + one block-Lanczos + Davidson + KPM (window sizing) + Haydock (fine-grid primary) + Chebyshev-Jackson filter; archive 3 sweep harnesses. Clustered-eigenvalue fix = DOS-sized subspace + stochastic tr(P) count-in-window + residual locking/deflation (n_ritz=4 fixed is the failure). Procedural matvec-closure contract, dim-agnostic for fine grids. | **−800…−1000** | loader fixes (B3-B5) for e2e gates |
| `finite_q_bse.md` | On-grid Q reduces to a k-axis remap of conduction tensors + umklapp phase e^{−2πi G_umk·s_μ} at centroids + swapping the exchange tile to V_qmunu[Q] **including G=0** (finite, no head injection — BGW energy_loss convention). ψ(k+Q) already on disk (k-independent centroids). Matvec/solvers/sharding byte-identical. Spectra = structure factor S(Q,ω) seeded by ρ_cvk(Q) via the existing g0 vector + Haydock. Arbitrary off-grid Q explicitly out of scope (→ fine-grid design). | ~450-600 | B1 fix; loader fixes; arb-Q: fine-grid |

Shared seams the designs agreed on: the **B1 dense-exchange fix is Phase 0 for
everything**; head/wing injection generalises the existing
`apply_q0_head_rank1_sharded` call sites rather than adding parallel helpers; all
new sym/k-map needs route through SymMaps; fine-grid ψ/dcc caches live host-side
via io_callback.

Consolidated open questions for Jack: see `report.md` §"Decisions needed".

---

## 7. Recommended attack order

> **Superseded by `PLAN.md` (2026-07-15 revision, after owner review): fix-broken
> first, B1 adjudicated dense and greenlit, recover-or-generate V/W added,
> GW-infra alignment mandate added. The ordering below is the original synthesis.**

0. **Gate first** (the GW program's core lesson): dense-reference kernel gate +
   fresh-restart e2e pytest gate + BGW anchor + pad-mask gate (§5). Nothing moves
   before these exist — B1 survived precisely because ring-vs-serial is circular.
1. **Loader repair** (B3/B4/B5 + kgrid pass-down + W0 loud-fail): one flat-q
   normalisation, shared by both device paths; head injection single-sourced.
   Unblocks everything downstream (all e2e gates need fresh restarts).
2. **Physics fixes as their own commits**: B1 dense exchange (validated against
   the new dense gate + BGW bsemat on Si), B2 einsum, B6 padding, B7 eqp windows.
3. **Delete pass** (Tier F, ~20 confirmed-dead + archive the sweeps; re-verify
   zero callers at current HEAD before each — documentation claims of deadness
   were wrong 4× in the GW program). Restore bse_pseudopoles' lost wiring or
   explicitly park it — do NOT delete it as dead.
4. **Single-source pass** (§4 item 4 list) + config through gw_config + extract
   the sharding vocabulary from bse_ring_comm.
5. **Solver program** P1-P3 (consolidate → FEAST maturation → clustered fix).
6. **Feature designs in dependency order**: sr_lr P1 + head_wings P1 (both
   behaviour-preserving, land early) → fine-grid P1-P3 → finite-Q on-grid →
   arbitrary-Q / non-TDA / 2D-CSI later, per Jack's priorities.
