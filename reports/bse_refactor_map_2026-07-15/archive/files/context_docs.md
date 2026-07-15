# src/bse/{STATUS.md, BGW_COMPARE.md, context/*} — deep-read notes (prior-knowledge docs ledger)

Audit date: 2026-07-15, lorrax_D checkout.
Audit base note: the task named branch `agent/slate-linalg-ffi` @ e18d0e5, but the
working tree is on `agent/ppm-fit-conditioning` @ adc2197. Verified irrelevant for
this file set: `git diff --stat e18d0e5 HEAD -- src/bse/` is **empty** — the entire
`src/bse/` tree (code + docs) is byte-identical between the two commits.

Files covered (all documentation; no executable code in this note's scope):

| file | size | mtime | nature |
|---|---|---|---|
| `src/bse/STATUS.md` | 9.2 KB | 2026-05-07 | project status, validated results, index-ordering conventions |
| `src/bse/BGW_COMPARE.md` | 5.7 KB | 2026-05-07 | comparison cookbook: six conventions + recipe + sanity checks |
| `src/bse/context/README.md` | 8.7 KB | 2026-04-28 | algorithm overview (TDA/ISDF/matvec/Lanczos), sharding table, CLI table |
| `src/bse/context/bse_isdf_instructions.md` | 3.6 KB | 2026-04-28 | early (superseded) 6-axis-mesh matvec pseudocode |
| `src/bse/context/bse_feast_instructions.md` | 22 KB | 2026-04-28 | FEAST-pseudopoles theory: rational vs Chebyshev filters, GMRES, windows |
| `src/bse/context/feast_accuracy_notes.md` | 4.2 KB | 2026-04-28 | empirical FEAST accuracy findings on N=144 dense reference |
| `src/bse/context/parallel_bse_algos.md` | 25 KB | 2026-04-28 | v2 distributed-X ring/reduce-scatter matvec design, finite-Q algebra |
| `src/bse/context/tda_and_pseudopoles.md` | 12 KB | 2026-04-28 | non-TDA block matvec + windowed pseudopole construction (proposal) |
| `src/bse/context/gpt5.2suggestion.md` | 10 KB | 2026-04-28 | alternative sharding proposal (logical mesh Y→(Yν,Yb)) |
| `src/bse/context/Henneke-2020-…md` | 88 KB | 2026-04-28 | full Mathpix conversion of Henneke et al., CAMCoS 15(1) 2020 — the source paper |

Category: **context docs / prior-knowledge ledger** (validated results, conventions,
design proposals). Everything here is ≥2 months old (2026-04-28 … 05-07 vs audit date
2026-07-15), so per program rules each load-bearing claim below was re-checked against
HEAD code; rot findings are collected in "Dated / rotted claims" at the end.

## Purpose

These documents are the accumulated "what was already known/decided" for the BSE
module: (1) the validated 8×8 Si Haydock LORRAX-vs-BGW match and its exact
reproduction recipe, (2) the six comparison conventions that every BGW comparison
must satisfy, (3) FEAST/pseudopole accuracy findings, (4) three generations of
parallel-matvec design docs (of which the ring/distributed-X one was implemented),
and (5) the Henneke 2020 paper the whole ISDF-BSE formulation is built on.

### Physics as written in the docs (and verified against code)

BSE Hamiltonian, **spinor convention** (context/README.md:12-20):

```
H_BSE = D + V − W          # NOT D + 2V − W; spin factor absorbed by spin-traced M
D_{cvk} = ε_c(k) − ε_v(k)
```

ISDF pair density (README:27-33): `ρ_cv(r,k) = ψ*_c(r,k)ψ_v(r,k) ≈ Σ_μ ζ_μ(r) M_cv(μ,k)`
with **spin-traced** `M_cv(μ,k) = Σ_s ψ*_{c,s}(μ,k) ψ_{v,s}(μ,k)`.

Per-element formulas as implemented at HEAD (`bse_simple.py`, einsums verbatim,
X layout `[b, c, v, k]`, μ→`M`, ν→`N`, spinor legs `t`,`s`):

```
# V (exchange, q=0), bse_simple.py:89-131 — matches Henneke Eq 4-5 topology in the
# R[t,ν] = conj(ψ_c(ν))·ψ_v(ν) convention of tda_and_pseudopoles.md §2.2:
M_Y[k,c,v,N] = Σ_s conj(psi_c_Y[k,c,s,N]) · psi_v_Y[k,v,s,N]      # 'kcsN,kvsN->kcvN'
S[b,N,k]     = Σ_{c,v} M_Y[k,c,v,N] · X[b,c,v,k] / √Nk            # 'kcvN,bcvk->bNk'
U_mu[b,M,k]  = Σ_N V_q0[M,N] · S[b,N,k]                           # 'MN,bNk->bMk'
HX_V[b,c,v,k]= Σ_M conj(M_X[k,c,v,M]) · U_mu[b,M,k] / √Nk         # 'kcvM,bMk->bcvk'
# net prefactor (1/√Nk)·(1/√Nk) = 1/Nk  ✓ Henneke Eq 4-5

# W (screened direct), bse_simple.py:141-173 — Henneke Eq 4-6 spinor generalization:
T[b,M,N,t,s,k] = Σ_{c,v} psi_c_X[k,c,t,M] · conj(psi_v_Y[k,v,s,N]) · X[b,c,v,k]
                                                       # 'kctM,kvsN,bcvk->bMNtsk'
T_R = ifft_ortho_k[T];  U_R = W_R ⊙ T_R;  U = fft_ortho_R[U_R]    # k-convolution
A[b,c,N,s,k]   = Σ_{M,t} conj(psi_c_X[k,c,t,M]) · U[b,M,N,t,s,k]  # 'kctM,bMNtsk->bcNsk'
HX_W[b,c,v,k]  = Σ_{N,s} psi_v_Y[k,v,s,N] · A[b,c,N,s,k] / √Nk    # 'kvsN,bcNsk->bcvk'
return D_term + HX_V − HX_W                                        # bse_simple.py:175
```

Normalization check written out (README:65-68 claim, verified): with all three
transforms `norm='ortho'` (W_R built once at bse_jax.py:149,
`W_R = jnp.fft.ifftn(W_q, axes=(2,3,4), norm="ortho")`),
`fft_ortho[ifft_ortho(W)·ifft_ortho(T)](k) = (1/√Nk) Σ_{k'} W_{k−k'} T_{k'}`;
the single explicit `/√Nk` at bse_simple.py:173 then yields the physical
`(1/Nk) Σ_{k'} W_{k−k'} T_{k'}` — exactly the README's "one additional 1/√Nk" rule. ✓

Spinor structure: conduction spinor index `t` lives on the μ leg (contracted between
encode `psi_c_X[k,c,t,M]` and decode `conj(psi_c_X[k,c,t,M])`), valence spinor `s`
on the ν leg — this is the 2×2 spin-matrix `T_ts(μ,ν,k)` of README:51-63
(Henneke eq 4-6 spinor generalization). Scalar W multiplies each spin component. ✓

Conjugation-convention caveat for verifiers (NOT a bug): Henneke Eq 4-5 as
transcribed in parallel_bse_algos.md:32 has the encode as `u_{jc k'}(ν)·ū_{jv k'}(ν)·X`
(conjugate on the *valence* leg); the code conjugates the *conduction* leg
(`M_Y = conj(ψ_c)·ψ_v`, bse_simple.py:89-92) and compensates with `conj(M_X)` on
decode. Encode/decode form a mutually adjoint pair (R, R†) per
tda_and_pseudopoles.md §2.2, and V_q0 is Hermitian, so the operator is identical.

## Entry points (who consumes these docs — grep evidence)

| doc | consumers (grep evidence) |
|---|---|
| `BGW_COMPARE.md` | cited from code: `src/bse/eigvals_to_eps2.py:24` ("Used by ``BGW_COMPARE.md`` as the canonical …"); cross-linked from `STATUS.md:4-9` |
| `STATUS.md` | cross-linked from `BGW_COMPARE.md:121`; no code/skill/run references found (grep `bse/STATUS` over skills/, CLAUDE.md, CHANGELOG.md → empty) |
| `context/*.md` | referenced **only from other checkouts of the same repo** (grep for doc basenames across the sandbox hit only `tmp_clone_slate_check/`, `blitz_workspaces/blitz{4,5,6}/`, `lorrax_D_old/` copies of `src/bse/context/README.md`); no tool, test, or script reads them |
| `Henneke-2020-…md` | cited by name in `parallel_bse_algos.md` (Eq 4-5/4-6, Eq 2-32) and `context/README.md:42` ("Henneke (2020) eq (4-6)") |

i.e. these are pure prior-knowledge documents; deleting them breaks nothing
mechanically, but STATUS/BGW_COMPARE are the only record of the validated comparison.

## Per-document digest

### STATUS.md (agent C, 2026-04-28; last touched 2026-05-07)

**Module table** (STATUS:13-23): 9 entries — bse_jax (CLI/driver/_preview_lanczos),
bse_simple, bse_ring_comm, bse_lanczos (`solve_bse_sharded`), bse_io
(`write_eigenvectors_stream`), absorption_common, absorption_eigvecs,
absorption_haydock, eigenvectors.h5.spec. Verified: `_preview_lanczos` at
bse_jax.py:203, `solve_bse_sharded` at bse_lanczos.py:100,
`write_eigenvectors_stream` at bse_io.py:23. **Rot**: table says bse_simple is
"default `--matvec-kind=simple`"; HEAD default is `"ring"` (bse_jax.py:449
`choices=("ring","gather","simple"), default="ring"`; absorption_haydock.py:326
same). Table also covers only 9 of the 24 .py files now in src/bse/ (missing:
bse_feast, bse_kpm, bse_davidson_helpers, davidson_absorption, bse_preconditioner,
bse_pseudopoles, bse_serial, bse_w_exact, eigvals_to_eps2, feast_*_sweep,
pseudopoles_*, write_eigenvectors, bse_feast_dense_debug, test_*).

**Index ordering — BGW-compat conventions** (STATUS:25-34). These are the
authoritative statements distinguishing LORRAX-internal from BGW conventions;
all verified still implemented:

1. Valence axis reversed: BGW `iv=1` = highest valence; LORRAX `v=0` = deepest.
   Source cited: `BGW/Common/evecs.f90:1982`, `BSE/input_fi.f90:407`
   (`ib = kp%ifmax - ib_kp + 1`). Writer flips on write: bse_io.py:98-100
   (`vec = vec[:, :, ::-1]` with comment "v=0 at the deepest valence — flip on
   write"). Conduction axis identical in both (c=0 lowest).
2. Units: BGW eigenvalues in eV, LORRAX internal Ry; converter
   `RYD2EV = 13.6056980659` bse_io.py:40 (also `ry_to_ev` default bse_io.py:702).
3. h5py axis reversal: Fortran file dims `[scalar, ns, nv, nc, nk, N, nq]` →
   numpy `(nq, N, nk, nc, nv, ns, 2)`; axis 3 = nc, axis 4 = nv.
4. vmtxel flat index `is + (iv-1 + (ic-1 + (ik-1)*nc)*nv)*nspin` — spin fastest,
   then v, then c, k slowest ⇒ reshape `(nk, nc, nv)` C-order. Self-consistent.

**Validated results** (STATUS:63-83), the ledger of what is DONE:
- BSE eigenvalues vs BGW within ~3 meV for lowest 20 (Si 4×4×4, 8v×8c),
  saturated at ISDF compression floor (n=400 → n=2400 Krylov: no change).
- Total Σ|d_cvk|² = 2314.177 both codes — machine-precision match (gauge-invariant).
- Haydock-vs-Haydock at same n_iter: peak ε₂ within 1.5 %, peak ω within 70 meV.
- Haydock 100 ≈ BGW full diag (141 / 144 / 146 at ~3.2 eV).
- Eigenvector route n_eig=100 with `--skip-vnl`: peak ε₂ 25.96 vs BGW 26.02 (0.2 %).
- Known-open: per-state |A^S|² cosine similarity ~80 % (real ISDF eigenvector
  rotation, O(δH/spacing)≈20 %); Lanczos eigvec sum-over-states converges peak
  slowly (18 % at n=100, 50 % at n=400 — use Haydock); 485 μeV triplet splitting
  (ISDF centroid symmetry); random per-element dipole phases (gauge, harmless).
- Gauge discipline section (STATUS:49-61): compare only gauge-invariant scalars
  (Σ|d|², per-state |A|² distributions, manifold sums, ε₂(ω)); never complex
  inner products across codes.

Run artifacts verified to still exist at
`runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/`: `dipole_p_only.h5`,
`eigenvectors.h5`, `eps2_8x8_haydock_compare.png`, `eps2_8x8_converged.png`,
`eps2_reconstructed_apples.png` (ls 2026-07-15 ✓), plus BGW reference dirs
`00_bgw_bse_8x8{,_haydock}`.

**Haydock prefactor claim** (STATUS:122): `16π²/(V·N_k·n_spin·n_spinor)` matching
`BSE/absh.f90:46` — verified in code at absorption_haydock.py:109
(`pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)`); the ‖d‖² factor
(`mmts%norm`) appears as `norms[a]**2` at absorption_haydock.py:117. (BGW-source
line numbers not re-verifiable from this Perlmutter checkout — BGW tree lives on
the local box; treat f90 line refs as unverified-but-plausible.)

### BGW_COMPARE.md — the six conventions (all six re-verified at HEAD)

1. **Dipole operator**: BGW `use_momentum` = bare p̂ (despite manual);
   use `psp.get_dipole_mtxels --skip-vnl` → `dipole_p_only.h5`. Verified:
   `--skip-vnl` flag at src/psp/get_dipole_mtxels.py:470, writes
   `h5.attrs['skip_vnl']` at :734 (the sanity check in BGW_COMPARE:108-109 reads
   this attr). Default dipole.h5 (with V_NL) mismatches vmtxel ~10× on Σ|d|².
2. **QP corrections**: pass BGW's `eqp.dat` via `--eqp`, not LORRAX eqp0.dat
   (50-200 meV). `--eqp` flag: bse_jax.py:477, absorption_haydock.py:307;
   `apply_eqp_corrections` bse_io.py:700.
3. **Head injection**: `use_bgw_vcoul = true` + `bgw_vcoul_file` in cohsex.in —
   config keys live in the GW side: src/gw/gw_config.py:339-340,644-645,1149-1150;
   consumed at src/gw/compute_vcoul.py:274-284. BSE-side manual head overrides
   `vhead` / `whead_0freq` parsed from cohsex.in at bse_io.py:668-698
   (`_parse_head_overrides`) — STATUS's "vhead = 3303.748 (BGW wcoul0)" mechanism.
4. **SOC band counting**: BGW counts SP bands, LORRAX historically Kramers pairs;
   post-`agent-C/bse-band-slicing-fix` both work. The fix is **merged**: commit
   7c2e32a "bse: fix silent band-slicing bug; require WFN.h5 or explicit n_occ"
   is an ancestor of HEAD (git merge-base --is-ancestor ✓); the branch name in the
   doc is historical.
5. **n_occ resolution**: explicit `--n-occ` OR WFN.h5 `ifmax` via
   cohsex.in[wfn_file]; loader raises otherwise. Verified verbatim:
   `resolve_n_occ` bse_io.py:603-666 — order (1) explicit, (2) WFN.h5 ifmax via
   `_parse_wfn_path`, (3) explicit `fermi_energy` hint, else `ValueError`;
   docstring records that the old "largest gap" auto-detect was silently broken
   for QE reference levels. (Stale echo elsewhere: absorption_haydock.py:172
   comment still mentions "largest-gap fallback" — comment rot, not in my docs.)
6. **Polarization/η/n_iter parity**: BGW `1.0 0.0 0.0` ↔ LORRAX `b1` output;
   `energy_resolution` ↔ `--eta-eV`; `number_iterations` ↔ `--n-iter`. Flags exist:
   absorption_haydock.py:313 (`--n-iter`), :319 (`--eta-eV`).

Also verified: the "common mistakes" table's biggest trap (Lanczos-eigvec-at-n vs
BGW-full-diag-truncated-to-n is NOT a bug — Ritz vectors ≠ exact eigvecs; use
Haydock) is restated in three places (STATUS:76,83; BGW_COMPARE:99) — clearly the
most-repeated lesson; keep prominent in any doc consolidation.
`bse.eigvals_to_eps2` CLI (BGW_COMPARE:77-83): `--files/--eta-eV/--n-max/--label/--out`
all exist (eigvals_to_eps2.py:115-134). Calibration claim: η=0.20, all 500 states →
reproduces BGW absorption_eh.dat peak (143.7) within 0.4 %.

### context/README.md — algorithm overview (2026-04-28, partially rotted)

Still-correct core: TDA spinor `H = D + V − W` with the explicit note on why not
`2V` (spin-traced pair amplitude; README:20); ISDF expansion; the matvec
encode/convolve/decode structure with the 1/Nk bookkeeping (verified above);
`ISDF_JAX_PROFILE_DIR` profiling env (test_bse.py:333-334, common/jax_profile.py:12);
test_bse CLI table (all 8 flags verified at test_bse.py:311-319: `--n-val/--n-cond`
default 4, `--n-eig` 10, `--max-iter` 50, `--n-warmup` 2, `--n-bench` 10,
`--no-jit-lanczos`, `--write-eigenvectors`); eigenvectors.h5 output schema
(matches bse_io.py:45-77: nQ=1, `exciton_Q_shifts` zeros).

Rotted (details in claims section): sharding table (X now sharded on BOTH mesh
axes), "Known Limitations" 1 (V-as-W placeholder) and 3 (TDA-only), Lanczos
implementations' location (moved to solvers.lanczos), Files table (6 of 24 files).
Limitation 2 (**no finite-Q, Q=0 only**) is still TRUE at HEAD: nQ=1 hardcoded
bse_io.py:45, write_eigenvectors.py:66-67 defaults Q=0; the finite-Q algebra in
parallel_bse_algos.md §1 remains unimplemented design.

### context/bse_isdf_instructions.md — superseded design (historical only)

Proposes a 6-axis device mesh `('i','μ','ν','kx','ky','kz')` with pjit'd
`apply_D/apply_V/build_A/apply_W` (its :26-101). Nothing of this survives: HEAD
uses the 2-axis `(x, y)` mesh everywhere (`make_bse_shardings`, bse_ring_comm.py:46).
Its scaling targets (N_μ ~ 5×10⁴, nk ~ 200³) are fantasy relative to the
implemented regime (N_μ ~ 10²-10³, nk ~ 10¹-10³). Keep only as history.

### context/bse_feast_instructions.md — FEAST theory (implemented, CLI drifted)

Theory doc for rational-filter pseudopoles: FEAST contour quadrature
`P̂ ≈ Σ_j w_j (z_j I − H)⁻¹`, ellipse parameterization
`z_j = (a+b)/2 + r_x cosθ_j + i r_y sinθ_j`, `r_y = γ r_x`, conjugate-symmetry
trick (upper half-plane only, TDA/Hermitian only — non-TDA needs both halves,
:133-139), diagonal preconditioner `M(cvk) = ε_c − ε_v + 2V_diag − W_diag`
(:90-96 — note the theory doc's 2V; code is spinor V), preconditioned GMRES,
Rayleigh-Ritz pseudopoles = Gauss-quadrature nodes (moment-matching optimality),
Tr(S) spectral-weight estimate, 3-tier window allocation, cost model
`C_w = R_w(n_quad·n_inner + 1)`. Implementation exists: bse_feast.py (ellipse
quadrature :592,968-973, GMRES `gmres_solve_sharded_jit`, preconditioner
`build_preconditioner_diagonal_sharded`), and a **Zolotarev quadrature the doc
never mentions** (bse_feast.py:476 default `quadrature="zolotarev"` in the
function signature, `feast_zolotarev_quadrature` :588-589, CLI `--quadrature
{zolotarev,ellipse}` :1123-1124 with CLI default "ellipse"; sweep tool
feast_zolo_sweep.py). Doc's practical values "γ=0.4, n_quad=4" (:143) do not
match HEAD's fixed `ELLIPSE_GAMMA_FIXED = 0.2` (bse_feast.py:30).

### context/feast_accuracy_notes.md — empirical findings (findings stand, CLI rotted)

Reference: dense diag of N=144 (4v×4c, 3×3×1), exact first 8 eigenvalues listed
(1.851722 … 2.067945 eV, three degenerate pairs). Findings:
1. **GMRES tolerance doesn't matter** (1e-2 → 1e-6 changes Ritz by ≤0.02 meV;
   diagonal preconditioner converges solves in 2-7 iterations).
2. **Error comes from the FEAST filter, not GMRES**: Ritz error grows from
   +3.6 meV (filter response f=0.990) to +51 meV (f=0.240) as eigenvalues approach
   the window boundary. This is the operative accuracy model for window placement.
3. **Lanczos ghost eigenvalues** at 60 steps on N=144 without full reorth —
   reliable for E_max bounds only. (Consistent with the bse_jax `--n-reorth`
   default -1 = full reorth and its help text, bse_jax.py:436-445.)
Sweep template rotted: `--n-quad {N}` and `--gamma {G}` no longer exist
(now `--n-quad1`/`--n-quad2` bse_feast.py:1093-1095; γ fixed at 0.2, no CLI);
`--window1/--window2 A auto` still exist (bse_feast.py:1140,1147, parsed :1224-1228);
`--n-lanczos` (:1082), `--feast-ritz`/`--feast-ritz-count` (:1097-1098),
`--gmres-max-iter/--gmres-tol` (:1126-1127) all still exist. The recommended sweep
was subsequently EXECUTED — feast_sweep.py, feast_zolo_sweep.py (has `--n-quad`
nargs='+'), feast_ellipse_mixed_sweep.py are the sweep harnesses; treat the
"Recommended parameter sweep" section as done work, not an open TODO.

### context/parallel_bse_algos.md — the design that was (mostly) implemented

Defines finite-Q TDA kernels (its §1.1, with `H = D + 2V_A − W_A` — spin-restricted
convention), the ISDF matvec sums (§1.2, transcribing Henneke Eq 4-5/4-6), and a
distributed-X scheme: X-axis owns (μ, c), Y-axis owns (ν, v), X(v_Y, c_X, k)
sharded on both axes; forward contractions = rings (ppermute), reverse =
reduce-scatters; FFT convolution local; V fused into the W ring. Claimed ~20×
bandwidth win over replicated-X.

Status vs HEAD: **the layout is adopted** — `make_bse_shardings` (bse_ring_comm.py:46-63)
shards `X = P(None, "x", "y", None)` on (b, c, v, k), ψ in two copies
`psi_x = P(None,None,None,"x")` / `psi_y = P(None,None,None,"y")` (all bands, μ or
ν sharded — the doc's §2.2 table), `W = P("x","y",None,None,None)`,
`V = P("x","y")`; ring-over-Y valence contraction implemented at
bse_ring_comm.py:_ring_sum_valence (:70+, `lax.ppermute` perm from `_ring_perm`
:66-67). The doc's §9 "ring as allgather + compute" alternative and the
XLA-auto-partitioned variant both exist as `--matvec-kind gather|simple`.
NOT adopted: finite-Q (§1.1: Q=0 only at HEAD); the §12 beyond-TDA W_B block in
ring form is superseded by `build_bse_ring_matvec_full` (bse_ring_comm.py:487)
whose structure should be read from code, not this doc. Spin factor: doc writes
`result = DX + 2·VX − WX` (:290); code returns `D + V − W` (spinor).

### context/tda_and_pseudopoles.md — non-TDA + pseudopole proposal (implemented in bse_pseudopoles.py)

Proposes: fold J into the standard eigenproblem
`S = [[A, B], [−B†, −A†]]`, `A = D + V − W`, `B = V − W`; R/v/R† three-step V
contraction (R^(q)[t,r_μ] = ψ*_{c,k−q}(r_μ)ψ_{v,k} — the conjugation convention the
code actually uses, see Purpose); per-window recipe: density-biased seeds
(Φ₀ = (f_j, f_j) with f = R†vη), FEAST filter, orthonormalize, reduced H_w,
residue snapshots C_w[r_μ, j] = Σ_ν v_μν Σ_t R[t,ν](X^(j)+Y^(j))[t]
("every residue is C_w @ g"), brightness Gram G_w = C_w†C_w split bright/dim,
bright Ritz poles + stochastic tail poles rescaled by α = √(B_disc/B_tail).
Implementation exists at HEAD: bse_pseudopoles.py — non-TDA J-metric
`J[i,j] = <v_i_X|v_j_X> − <v_i_Y|v_j_Y>` (:79), C_w density snapshots with
"non-TDA w = V(dX + d*Y)" (:138-142, computed :341-354), transpose-coupled Y-sector
vertex (:285), bright extraction `d_bright = C_w @ g_b` (:389), J-normalization
(:425-427), conjugate-pair augmentation (:436), tail poles (:440-451 with
Rayleigh-quotient Ω and C_w@g residues). CLI `--n-quad` default 8, `--n-tail` etc.
(bse_pseudopoles.py:560+). Treat the doc as design input; code has post-doc
refinements (J-norms, leak filtering :373) that the doc lacks.

### context/gpt5.2suggestion.md — alternative sharding proposal (NOT adopted)

Proposes logical mesh reshape Y → (Y_ν, Y_b): μ on X, ν on Y_ν, the Lanczos
**block index b sharded on Y_b**, k replicated, c sharded on X only for
reduce-scatter output assembly; `HX = DX + 2VX − WX`. HEAD does not shard the
block axis: `X = P(None, "x", "y", None)` (bse_ring_comm.py:48) has b replicated
and v sharded on y instead — i.e., the parallel_bse_algos.md scheme won. Elements
that did land regardless of provenance: reduce-scatter decode (bse_simple.py:27-30
cites the `psum_scatter` trick), two ψ copies, k-replicated local FFT. Keep as a
record of the road not taken; nothing references it.

### context/Henneke-2020-….md — source paper (reference, verbatim)

Full Mathpix conversion of Henneke, Lin, Vorwerk, Draxl, Klein, Yang, CAMCoS 15(1)
2020. Load-bearing equations for LORRAX: Eq 2-32 (:308) — q=0 bare Coulomb with
G=G'=0 element zeroed (`V̂_0(G,G') = 4π/|G|² δ_GG', G≠0; 0, G=0` — the reason V_q0
carries no head and the vhead/wcoul0 injection exists at all); Eq 4-1/4-2 kernel
definitions; Eq 4-5 (:454) V matvec; Eq 4-6 (:464) W matvec (the k−k' convolution);
Eq 4-7 (:482) ε₂(ω) resolvent form; structure-preserving Lanczos refs [34]
(Shao/da Jornada BSEPACK). The TDA-vs-full-BSE treatment in the paper (V_B/W_B
blocks, Eq 4-3) is what §12 of parallel_bse_algos.md and tda_and_pseudopoles.md
build on. No rot possible (it's a paper); keep verbatim.

## Flags / CLI args consumed

None — these are docs. But they DOCUMENT flag surfaces; the verified-current map is:

| doc claims | HEAD reality (evidence) |
|---|---|
| `--matvec-kind=simple` default (STATUS:16) | default `"ring"` — bse_jax.py:449, absorption_haydock.py:326 |
| `--n-reorth -1` full-reorth default (STATUS:15) | ✓ default=-1, bse_jax.py:436-438 |
| `--n-quad`, `--gamma` on bse_feast (feast_accuracy_notes:74-85) | gone; `--n-quad1/--n-quad2` bse_feast.py:1093-1095, γ fixed `ELLIPSE_GAMMA_FIXED=0.2` :30, plus new `--quadrature {zolotarev,ellipse}` :1123 |
| `--window1/--window2 … auto` (feast_accuracy_notes:69) | ✓ bse_feast.py:1140,1147 |
| STATUS/BGW_COMPARE recipe flags (`--bse --lanczos --tda --n-val --n-cond --n-occ --eqp --write-eigs --max-lanczos-iter --px --py --n-iter --eta-eV --V-cell --no-eps1 --out-prefix --dipole`) | ✓ all present: bse_jax.py:353-483, absorption_haydock.py:302-326, absorption_eigvecs.py:94-115 |
| test_bse CLI table (context/README:164-174) | ✓ test_bse.py:311-319; imports still resolve (test_bse.py:34-38 ← bse_jax `__all__` bse_jax.py:44-64) |
| `ISDF_JAX_PROFILE_DIR` (README:180) | ✓ test_bse.py:333, common/jax_profile.py:12 |
| cohsex.in keys `vhead`, `whead_0freq` (STATUS:42) | ✓ bse_io.py:668-698 |
| cohsex.in keys `use_bgw_vcoul`, `bgw_vcoul_file` (BGW_COMPARE §3) | ✓ gw_config.py:339-340 (GW side, feeds the restart bundle BSE consumes) |

## Sharding / residency / TDA / spin — what the docs establish

- **Sharding (current truth, README table superseded)**: 2-axis mesh (x, y);
  `X[b,c,v,k] = P(None,'x','y',None)`; non-TDA `X_full = P(None,None,'x','y',None)`
  (extra leading X/Y-sector axis); ψ two host copies μ-sharded/ν-sharded;
  `W_q[μ,ν,kx,ky,kz] = P('x','y',None,None,None)`; `V_q0 = P('x','y')`
  (bse_ring_comm.py:46-63).
- **TDA vs full BSE**: TDA is the production path (`--tda`); non-TDA machinery
  exists (build_bse_ring_matvec_full bse_ring_comm.py:487; bse_w_exact.py
  non-TDA Casida shifted solves; bse_pseudopoles X/Y sectors) — README's
  "TDA only" is stale.
- **Spin/nspinor**: spinor-native. V uses spin-traced M (Σ_s at each vertex);
  W keeps the 2×2 (t,s) spin matrix through the convolution. `H = D + V − W`
  (no singlet factor 2). absorption tools take `--n-spin` (default 1) and
  `--n-spinor` (haydock default 2, eigvecs default None→file attr).
  SOC band-counting convention is BGW_COMPARE §4.
- **BGW-compat vs internal**: valence-axis flip and Ry→eV happen ONLY at the
  eigenvectors.h5/eigenvalues.dat file boundary (bse_io.py:40,98-100); everything
  in-memory is LORRAX-internal (v=0 deepest, Ry). Any verifier comparing arrays
  must apply STATUS:25-34 before calling a mismatch a bug.

## Coupling to gw/ and isdf/ modules (as established by these docs)

- BSE consumes the GW pass's restart bundle: `cohsex.in`, `centroids_frac_*.txt`,
  `WFN.h5`, `tmp/isdf_tensors_*.h5` (BGW_COMPARE:38-41); W_q/V_q0 come from that
  bundle (bse_jax.py:309 `payload["W_q"]` via bse_io.load_bse_data_from_restart_sharded)
  — NOT from eps0mat.h5 and NOT a V-placeholder.
- Head injection chain: BGW `vcoul` → gw_config `use_bgw_vcoul`/`bgw_vcoul_file`
  (gw side) or direct `vhead`/`whead_0freq` overrides (bse_io.py:668).
- Dipoles from `psp.get_dipole_mtxels` (momentum operator; `--skip-vnl` for
  BGW `use_momentum` parity — ties to the sandbox-wide velocity-sign convention:
  p−vNL is a BGW convention, p+vNL is physical).
- eqp corrections from BGW `eqp.dat` via bse_io.apply_eqp_corrections.

## Dated / rotted claims (all >2 months old — for verifier re-check)

1. STATUS:16 "`--matvec-kind=simple` … default" — default is now "ring"
   (bse_jax.py:449). Also re-benchmark "fastest" before restating.
2. context/README:229 "V as W placeholder … Real W loading is TODO" — stale;
   real W_q loaded from restart bundle (bse_jax.py:309), and STATUS's own 3-meV
   eigenvalue match could not exist with placeholder W.
3. context/README:233 "TDA only … not implemented" — stale; non-TDA machinery
   exists (bse_ring_comm.py:487, bse_w_exact.py:1, bse_pseudopoles.py:79).
4. context/README:105-111 sharding table (`X: P(None,'x',None,None)`, W_q with
   flat nk axis) — stale vs bse_ring_comm.py:46-63.
5. context/README:82-89,251-261 Lanczos-in-bse_jax + 6-file Files table — moved
   to solvers.lanczos (bse_lanczos.py:19-25); package is 24 files.
6. feast_accuracy_notes:74-85 sweep template flags `--n-quad`/`--gamma` — gone;
   γ now fixed 0.2 (notes' reference data used γ=0.4); zolotarev option postdates
   the notes; the sweep itself was executed (feast_*_sweep.py).
7. bse_feast_instructions:143 "practical values γ=0.4, n_quad=4" — HEAD pins
   γ=0.2 (bse_feast.py:30) and defaults n_quad1=4/n_quad2=8.
8. BGW_COMPARE:23 branch ref `agent-C/bse-band-slicing-fix` — merged (7c2e32a in
   HEAD ancestry); phrasing "post-…-fix" is now unconditional.
9. Docs' `H = D + 2V − W` (bse_feast_instructions:90, parallel_bse_algos:5,290,
   gpt5.2suggestion:1,289) vs code `D + V − W` — spinor convention, intentional
   (README:20). Convention flag so nobody "fixes" the 2.
10. STATUS:13-23 module table covers 9 of 24 files — needs regeneration, not
    trust.
11. bse_isdf_instructions.md 6-axis mesh — fully superseded; candidate for
    archive/deletion (no references anywhere).
12. gpt5.2suggestion.md block-sharded-b mesh — not adopted; candidate for
    archive with a one-line "road not taken" note.
13. STATUS:29,32,122 BGW f90 line-number citations (evecs.f90:1982,
    input_fi.f90:407, haydock.f90:536, absh.f90:46) — unverifiable from this
    checkout (BGW source not mounted here); re-pin if BGW is updated.

Still-true (re-verified, do not flag): six conventions' mechanics (§1-§6 all have
live code), resolve_n_occ contract, valence flip + Ry→eV on write, Haydock
prefactor, Q=0-only limitation, run artifacts on disk, test_bse CLI + imports,
`--n-reorth -1` default, ISDF_JAX_PROFILE_DIR.
