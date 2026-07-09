# Group: src/common — chi head/wing + bispinor foundations

Files: `src/common/chi_sos.py`, `src/common/chi_from_dipole.py`,
`src/common/gamma_matrices.py`, `src/common/bispinor_init.py`.
Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.
Caller evidence = grep across `src/`, `tests/`, `tools/`, `scripts/` for module names
and each public function name (commands recorded per file below).

---

## 1. src/common/chi_sos.py (363 LOC)

**Purpose.** Sum-over-states (SOS) evaluation of χ⁰ head/wing values at finite q,
plus the q=0 S-tensor and linear wing coefficient w_{α,μ}, from the finite-q matrix
elements written by `psp.get_dipole_mtxels --with-finite-q` (`dipole.h5/finite_q`).
Declared "sister to `common.chi_from_dipole`" and designed to be interchangeable with
the Sternheimer head/wing backend. Pure math module: imports only jax/numpy, no repo deps.

**Category guess.** physics: chi0 head/wing (SOS backend) — currently *unwired*.

**Physics (from module docstring, verbatim conventions).**
- `F_t(q, ω) = (f_v − f_c) · 2 Δ_t(q) / (ω² − Δ_t(q)²)`
- `Δ_t(q) = ε_{c,k−q} − ε_{v,k}`
- `h_t(q) = ⟨u_{c,k−q}|u_{v,k}⟩ = rho_cvkq[c,v,k,q]`
- `χ_head(q,ω) = (1/N_kΩ) Σ_t F_t |h_t|²`
- `χ_wing,μ(q,ω) = (1/N_kΩ) Σ_t F_t b*_{t,μ}(q) h_t(q)`
- `S_{αβ}(ω) = (1/N_kΩ) Σ_t F_t(0,ω) d_t^{α*} d_t^β`, `d_t^α = v^α_t/Δ_t`
- `w_{α,μ}(ω) = (1/N_kΩ) Σ_t F_t(0,ω) b^{0*}_{t,μ} d_t^α`

The centroid vertex `b_{t,μ}(q)` (ISDF ψψ* at centroids) is explicitly NOT plumbed
through — callers must build it from the zeta pipeline (docstring lines 37–42).

### Function table

| Function | Lines | Role |
|---|---|---|
| `deltaE_cv_at_q(en_full, kminq_idx_kq, v_lo, c_lo, c_hi)` | 57–76 | numpy: builds Δ_cvkq `(nc,nv,nk,nq)` from full-BZ energies `(nk_full,nb)` and `kminq_idx` `(nk_full,nq)` (from `--with-finite-q`). `delta = en[kminq_idx, c_lo:c_hi][:,:,:,None] − en[:,v_lo:c_lo][:,None,None,:]`, transposed `(2,3,0,1)`. |
| `_compute_chi_head_jit` | 83–116 | jit (static `omegas_is_scalar`). `weight = (f_v−f_c)·|rho|²`; per-ω `F = 2Δ/(ω²−Δ²)` with `jnp.where(|denom|>1e-16, ..., 0.0)` guard; einsum verbatim: `'cvkq,cvkq->q'` on (weight, F). vmap over ω. Returns `(n_w, nq)`. |
| `compute_chi_head_at_q` | 119–147 | public driver; prefactor `pref = 2.0/(Ω·N_k·max(nspin,1)·max(nspinor,1))`; casts all to c128/f64 device arrays. |
| `_compute_chi_wing_jit` | 150–178 | jit. Inputs `b_cvkq_mu (nc,nv,nk,nq,n_mu)` c128, `rho_cvkq`. einsum verbatim: `'cvkqm,cvkq,cvkq->qm'` on `(jnp.conj(b_cvkq_mu), weight_h, F)`. Returns `(n_w, nq, n_mu)`. |
| `compute_chi_wing_at_q` | 181–208 | public driver, same pref = 2.0/(...). |
| `_compute_S_tensor_jit` | 215–245 | jit. `weight = fvfc[None,...]·d_alpha_cvk (3,nc,nv,nk)`; einsum verbatim: `'acvk,bcvk,cvk->ab'` on `(jnp.conj(d_alpha_cvk), weight, F)`. Returns `(n_w,3,3)`. Docstring: "Convention matches existing `chi_from_dipole.compute_S_omega`: full Hessian of χ_head w.r.t. q_i,q_j (NOT quadratic-coef)." |
| `compute_S_tensor_sos` | 248–285 | public; builds `d^α = v^α/Δ` with |Δ|>1e-12 guard (term zeroed below), pref = 2.0/(...). Input `v_alpha_cvk_q0` = velocity mtxel at q=0 (the `dipole_cart` slice in dipole.h5). |
| `_compute_w_tensor_jit` | 288–312 | jit. einsum verbatim: `'cvkm,acvk,cvk,cvk->am'` on `(jnp.conj(b0_cvk_mu), d_alpha_cvk, fvfc, F)`. Returns `(n_w,3,n_mu)`. |
| `compute_w_tensor_sos` | 315–354 | public; `b0_cvk_mu` = q=0 centroid vertex `b^0_{t,μ} = Σ_s ψ*_{c,k,s}(r_μ) ψ_{v,k,s}(r_μ)`. |

`__all__` = deltaE_cv_at_q, compute_chi_head_at_q, compute_chi_wing_at_q,
compute_S_tensor_sos, compute_w_tensor_sos.

### Entry points / callers
Grepped repo-wide (`grep -rn --include='*.py'` on `chi_sos`, `deltaE_cv_at_q`,
`compute_chi_head_at_q`, `compute_chi_wing_at_q`, `compute_S_tensor_sos`,
`compute_w_tensor_sos` across src, tests, tools, scripts and whole repo):
**ZERO functional callers.** Only mentions are docstrings:
- `src/common/kq_mapping.py:9` — "consumed by `common.chi_sos`" (aspirational; chi_sos does not import kq_mapping either).
- `src/psp/run_sternheimer.py:956` — comment describing kq_mapping as shared with "the SOS finite-q matrix element pipeline (`common.chi_sos` + `get_dipole_mtxels`)".

The *producer* side exists and is live: `src/psp/get_dipole_mtxels.py:275
compute_finite_q_mtxels` + `--with-finite-q` CLI flag (line 489) writes
`dipole.h5/finite_q` (`rho_cvkq`, `v_cvkq`, `kminq_idx`, group created at line 737).
The *consumer* (this module) is never invoked.

### I/O
None directly. Consumes in-memory arrays whose canonical source is
`dipole.h5` group `finite_q` (datasets `rho_cvkq`, `v_cvkq`, `kminq_idx`) written by
`python -m psp.get_dipole_mtxels --with-finite-q`. Caller does the h5 read.

### Flags consumed
None (no LorraxConfig / cohsex.in access).

### Suspects
- **dead_suspects:** entire module. Evidence: grep for `chi_sos` and all five
  `__all__` names across the whole repo returns only the two docstring mentions above;
  no import statement anywhere.
- **redundancy_suspects:**
  - `compute_S_tensor_sos` duplicates the physics of
    `chi_from_dipole.compute_S_omega` (same S(ω) Hessian, same dipole.h5 inputs) —
    a parallel new/old path; the module admits sisterhood in its header.
  - The `F = 2Δ/(ω²−Δ²)` + `jnp.where` guard + prefactor + `omegas_is_scalar`
    vmap scaffold is copy-pasted 4× within the file (head/wing/S/w jit bodies).
- **weird_code:**
  - lines 96–100 + 134: docstring says prefactor should "match whatever
    `chi_from_dipole.compute_S_omega` uses to keep convention" (2/(N_kΩ) vs
    4/(N_kΩ) spin doubling) — but this module hardcodes **2.0** while
    `compute_S_omega` hardcodes **4.0** (chi_from_dipole.py:153). Factor-2
    convention mismatch left unresolved; hypothesis: never reconciled because the
    module was never wired in.
  - magic thresholds `1e-16` (ω-denominator) vs `1e-12` (Δ regularization in the
    d^α build) — two different cutoffs for the same kind of guard.

---

## 2. src/common/chi_from_dipole.py (169 LOC)

**Purpose.** Compute the q→0 head Hessian tensor S_{αβ}(ω) from `dipole.h5`
(velocity mtxels `p + i[r,V_NL]`) in the Adler–Wiser dipole form; used to build the
q=0 Coulomb head average `wcoul0` for the GW head correction. Also the h5 reader for
`dipole_cart`/`deltaE`.

**Category guess.** physics: chi0 q=0 head (dipole/SOS) — live, feeds head correction.

### Function table

| Function | Lines | Role |
|---|---|---|
| `read_dipole_h5(path)` | 99–103 | reads `dipole.h5` datasets `dipole_cart` `(3,nk,nb,nb)` c128 and `deltaE` `(nk,nb,nb)` f64 → jnp arrays (full device residency). |
| `_compute_S_omega_jit` | 106–135 | jit (static `nelec, nb, omegas_is_scalar`). Slices `v_cvk = dipole_cart[:, :, c_idx[:,None], v_idx[None,:]]` `(3,nk,nc,nv)` with `c_idx=arange(nelec,nb)`, `v_idx=arange(0,nelec)`; `denom = dE_cv·((ω+iη)²−dE_cv²)`; `W = where(|denom|>1e-16, (f_v−f_c)/denom, 0)·pref`; einsum verbatim: `'ancv,ncv,bncv->ab'` on `(jnp.conj(v_cvk), W, v_cvk)`; vmap over ω. Comment: replaces ~30 eager-pjit cache misses of the original wrapper with one compile. |
| `compute_S_omega(dipole_cart, deltaE, f_nk, cell_volume, nk_tot, nspin, nspinor, omegas, eta=0.0)` | 138–162 | public driver. `nelec = clip(sum(f_nk[0] > 0.5))` — occupations from k=0 row only. `pref = 4.0/(Ω·N_k·max(nspin,1)·max(nspinor,1))`. Returns `(n_w,3,3)` c128. |

Physics: `S_{αβ}(ω) = pref · Σ_{k,c,v} (f_v−f_c) · v^{α*}_{cvk} v^β_{cvk} / [ΔE·((ω+iη)²−ΔE²)]`.

Lines 13–88: a large WARNING comment block — "this S(ω) is in the DFT basis only —
must be re-derived for SC GW". Documents 4 unimplemented items (QP-basis rotation of
v via `U† V U`, einsum `'kpm,akpq,kqn->akmn'` given in the comment; QP ΔE; QP f_nk;
Σ-derivative velocity term) and their consequence: HeadResolver caches DFT-basis
S(ω) across all SC iterations → O(QP-shift/gap) systematic on the head. Explicitly a
roadmap, "intentionally NOT modified".

### Entry points / callers (grep evidence)
- `compute_S_omega, read_dipole_h5 <- src/gw/head_correction.py:155–166`
  (`from_s_tensor()` head-sample builder; feeds `gw.vcoul.compute_q0_averages(S_cart=...)`).
- `compute_S_omega, read_dipole_h5 <- scripts/checks/sigma_direct_check.py:67,184,190`.
- `common.chi_from_dipole` referenced in docstring of `src/psp/run_sternheimer.py:539`
  (Adler–Wiser formula pointer only).

### I/O
Reads `dipole.h5` (path resolved by caller: `head_correction.py:120` fixes it to
`<input_dir>/dipole.h5`): datasets `dipole_cart` `(3,nk,nb,nb)`, `deltaE` `(nk,nb,nb)`.
Writes nothing. Producer: `python -m psp.get_dipole_mtxels`.

### Flags consumed (indirectly, by its caller head_correction.py)
- `wcoul0_source` ("s_tensor" | "epshead", default "s_tensor" — gw_config.py:285,533)
  selects whether this module runs at all.
- `wcoul0_eta` → `eta` argument (gw_config.py:286,534).
Module itself reads no config.

### Suspects
- **dead_suspects:** none — both public functions have two live call sites.
- **redundancy_suspects:** `compute_S_tensor_sos` in chi_sos.py reimplements the same
  S(ω) (see above). Also `psp/run_sternheimer.py` computes an S-tensor via a wholly
  different (Sternheimer) backend — intentional backend pair, but three S-tensor
  producers total.
- **weird_code:**
  - line 151: `nelec` inferred from `f_nk[0]` (k-index 0 only) with a 0.5 threshold —
    assumes k-independent integer occupations; metals/smearing would silently
    mis-slice v/c blocks. (head_correction.py builds f_nk from `wfn.nelec` as 0/1 so
    it's consistent today.)
  - line 153: prefactor **4.0**/(ΩN_k·nspin·nspinor) vs chi_sos's 2.0 — the factor-2
    spin-doubling convention lives here implicitly.
  - lines 13–88: 75-line TODO/roadmap comment block (SC-GW invalidity of the whole
    module) — half the file is commentary.
  - `from __future__ import annotations` on line 1 *above* the module docstring
    (line 3) — the docstring is therefore not `__doc__` (a statement precedes it).
    Cosmetic bug.

---

## 3. src/common/gamma_matrices.py (321 LOC)

**Purpose.** Dirac/Pauli matrix constants (in a modified convention: γ̃^μ = γ⁰γ^μ so
`ψ† γ̃ ψ` replaces `ψ̄ γ ψ`) plus optimized spin-axis contraction kernels for the
bispinor pipeline: monomial (perm, phase) decomposition of each γ̃, single-vertex
`gamma_apply`, and the double contraction `gamma_double_contract` with three
XLA-memory-layout strategies (take/einsum/scan) selected by a module-global mode.

**Category guess.** physics kernel library: bispinor spin-algebra (γ̃ vertices) +
memory-layout tuning knob.

### Constants
- `sigma_x, sigma_y, sigma_z` (lines 12–14): 2×2 Pauli, c128.
- `gamma0..gamma3` (19–37): **not** standard Dirac γ — line 17 comment:
  "JM: actually I replace gamma0-3 with gamma0*gamma0-3, so that I can use
  psidag = conj(psi) rather than psibar = conj(psi) gamma0". So gamma0 = I₄,
  gamma1..3 = α^i = γ⁰γ^i (Hermitian, monomial).
- `gamma5` (40–43): labeled "γ^5 = iγ⁰γ¹γ²γ³" — value is off-diagonal-identity
  blocks (standard Dirac-rep γ5). No consumer (see dead suspects).
- `gammas = [gamma0..3]` (78); `gammas_sparse = [_to_sparse(g) for g in gammas]` (79).
- `gammas_perm (4,4) int32`, `gammas_phase (4,4) c128` (86–88): stacked
  (perm, phase) decompositions.
- `_GAMMA_CONTRACT_MODE: str = "take"` (299): module-global, set once at driver init.

### Function table

| Function | Lines | Role |
|---|---|---|
| `_to_sparse(mat)` | 45–48 | (rows, cols, values) of nonzeros via `jnp.nonzero`. Only consumer of output is... nothing (see suspects); computing it at import time is itself load-bearing (isdf_fitting.py:2026–2032 does a "warm import" so `jnp.nonzero` runs outside jit and avoids ConcretizationTypeError). |
| `_to_perm_phase(mat)` | 51–75 | numpy: decompose monomial 4×4 into `mat[α,β] = phase[α]·δ_{β,perm[α]}`; asserts one nonzero per row. |
| `gamma_perm_phase(mu_lorentz)` | 91–100 | public accessor: `(gammas_perm[μ], gammas_phase[μ])`. |
| `gamma_apply(X, perm, phase, axis, is_identity=False)` | 103–126 | `Y[...,β,...] = phase[β]·X[...,perm[β],...]` — `jnp.take` + broadcast multiply replacing the 4×4 spin matmul; `is_identity=True` (trace-time Python bool) returns X unchanged. |
| `_gamma_matrix_from_perm_phase(perm, phase, ns, dtype)` | 129–144 | rematerialize dense γ̃ = `phase[:,None]·eye[perm]`; None ⇒ I_ns. |
| `_gamma_double_contract_take` | 147–160 | historical impl: two `gamma_apply` on P_r then `jnp.sum(P_l_conj·P_r_p, axis=spin_axes)`. Comment: "Verified to drive 5 concurrent rank-5 slots in XLA's BufferAssignment on MoS2 3×3. See `gw/gflat_memory_model.py`." |
| `_gamma_double_contract_einsum` | 163–189 | einsum verbatim: `'kabmr,aA,bB,kABmr->kmr'` on `(P_l_conj, γL, γR, P_r)` with `optimize='optimal'`; only for `spin_axes == (1,2)` (standard rank-5 `(k,a,b,μ,ν)` layout), else falls back to take. |
| `_gamma_double_contract_scan` | 192–239 | `lax.scan` over ns² spin pairs; rank-3 slices via `lax.dynamic_index_in_dim` (a then b on P_l_conj; perm_L[a], perm_R[b] on P_r); `acc + phase_L[a]·phase_R[b]·Pl_3·Pr_3`. Predicted 2 concurrent rank-5 slots vs take's 5. Falls back to take for spin_axes ≠ (1,2). |
| `gamma_double_contract(P_l_conj, P_r, perm_L=None, phase_L=None, perm_R=None, phase_R=None, spin_axes=(1,2))` | 242–293 | public dispatcher on `_GAMMA_CONTRACT_MODE`. Math (docstring): `Σ_{αβ} phase_L[α]·phase_R[β] · P_l_conj[...,α,β,...] · P_r[...,perm_L[α],perm_R[β],...]`; perm=None ⇒ identity (charge channel = pure Frobenius). All three strategies mathematically identical, differ only in XLA buffer layout. |
| `set_gamma_contract_mode(mode)` | 302–313 | validates ∈ {take, einsum, scan}, sets the global. Docstring warns: mutating after kernels traced does not re-trace. |

### Entry points / callers (grep evidence)
- `sigma_x, sigma_y, sigma_z <- src/centroid/current_density.py:26,65`.
- `set_gamma_contract_mode <- src/gw/gw_init.py:542,547`
  (`set_gamma_contract_mode(cfg.backend.gamma_contract_mode)`).
- `gamma_perm_phase, gamma_apply <- src/gw/sigma_x_bispinor.py:84–98`
  (`_wfns_with_lorentz_vertices`, folds γ̃^μ into psi_xn axis 1 / psi_yr axis 2);
  `<- tests/test_sigma_x_bispinor.py:23,40,57` (identity + dense-matmul equivalence
  on both axes, also imports gamma0..3 as dense reference).
- `gamma_double_contract <- src/common/isdf_fitting.py:16–18,284,430,818`
  (pair-density C_R/Z_R kernels of the bispinor ζ-fit);
  `gamma_perm_phase` (aliased `_gamma_perm_phase_mu`) `<- isdf_fitting.py:17,1735,2098`;
  `<- tests/test_zq_from_psi_sm_bit_identity.py:36,191,411`.
- Warm-import side effect `<- src/common/isdf_fitting.py:2031–2032`.
- Comment references: `gw/gw_config.py:250`, `gw/aot_memory_model/kernels/fit_one_rchunk.py:389`.

### I/O
None.

### Flags consumed
- cohsex.in `gamma_contract_mode` ("take" | "einsum" | "scan", default "take" —
  gw_config.py:255,630,1039) via `set_gamma_contract_mode` called from gw_init.

### Key arrays crossing the boundary
- `P_l_conj, P_r`: rank-5 `(k, a, b, μ, ν/r)` c128, spin axes size 4, device-resident
  inside the per-r-chunk ζ-fit jit; the whole take/einsum/scan machinery exists to
  control their XLA slot count (memory HWM).
- `perm (4,) int32`, `phase (4,) c128` per Lorentz index.

### Suspects
- **dead_suspects:**
  - `gamma5` — grep for `gamma5` repo-wide: only its definition and `__all__`. Zero consumers.
  - `gammas_sparse` / `_to_sparse` — grep for `gammas_sparse`: only the definition and
    the isdf_fitting.py:2027 *comment*; nothing reads the sparse triples. It is kept
    alive purely as an import-time side effect the warm-import comment describes.
    Refactor note: if `gammas_sparse` were deleted, the warm import in
    isdf_fitting.py:2026–2032 loses its stated reason.
- **redundancy_suspects:** three parallel implementations of the same contraction
  (`_gamma_double_contract_take` / `_einsum` / `_scan`) kept simultaneously behind a
  runtime mode — deliberate (memory-layout experiment, documented slot counts), but a
  textbook parallel-paths candidate for consolidation once a winner is picked.
  `_gamma_matrix_from_perm_phase` inverts what `_to_perm_phase` did (round-trip
  dense→(perm,phase)→dense) just for the einsum path.
- **weird_code:**
  - line 17: "JM: actually I replace gamma0-3 with gamma0*gamma0-3..." — the exported
    names `gamma0..3` are NOT the Dirac γ^μ but γ̃^μ = γ⁰γ^μ; any future reader
    treating them as standard γ will get sign errors. gamma5's docstring formula
    ("i γ^0 γ^1 γ^2 γ^3") is written in terms of the *true* γ's, not the stored ones.
  - lines 2–9: `import jax` precedes the module docstring → docstring is not `__doc__`
    (same cosmetic bug as chi_from_dipole).
  - `_GAMMA_CONTRACT_MODE` global mutated post-import (line 299 defined *after* the
    functions that read it); trace-time capture means changing it after first trace
    silently does nothing (docstring admits this).
  - spin_axes ≠ (1,2) silently falls back to the take path in einsum/scan modes
    (lines 172–178, 201–203) — mode flag is not honored on the "one rare path".

---

## 4. src/common/bispinor_init.py (41 LOC)

**Purpose.** Legacy bispinor small-component lift: given large-component 2-spinor
ψ_nk(G), compute the lower Dirac components `ψ_small = (α/2)(σ·(k+G)) ψ_large`.
Single function, per-k (not k-vectorised).

**Category guess.** physics: bispinor preprocessing (legacy; superseded by wfn_loader's
vectorised lift, retained as test oracle).

### Function table

| Function | Lines | Role |
|---|---|---|
| `get_small_psi_component(gvecs, kvec, bvec, psi_G)` | 12–41 | `gvecsk_cart = (gvecs + kvec) @ bvec`; builds σ·p explicitly as 2×2 of G-arrays `[[p_z, p_x−ip_y],[p_x+ip_y, −p_z]]`; einsum verbatim: `"ijG,bjG->biG"` on `(sigmadotp, psi_G[:, 0:2, :])`, scaled by `halfalpha = jnp.complex128(0.00364867628215)` (α/2, fine-structure). Inputs: gvecs `(ngk,3)` int crystal, kvec `(3,)` crystal, bvec `(3,3)` recip rows, psi_G `(nb, nspinor, ngk)`. Output `(nb, 2, ngk)`. Deliberately not @jax.jit ("ngk varies per k-point → recompilation"). Docstring TODO: "σ·v with v = p + [r, V_NL + Σ], DKH4 contribution." |

### Entry points / callers (grep evidence)
- `get_small_psi_component <- tests/test_wfn_loader_eager.py:170,194` ONLY —
  used as the byte-for-byte reference oracle for the production lift.
- Production path does NOT call it: `src/file_io/wfn_loader.py` has
  `_apply_bispinor_lift` (line ~911) + `_bispinor_lift_kernel`/`_get_bispinor_lift_jit`
  (lines ~1024–1040) which "Matches the legacy
  `common.bispinor_init.get_small_psi_component` byte-for-byte but is vectorised
  across k" (wfn_loader.py:926, 1030), with its own copy of the constant
  `_HALFALPHA = 0.00364867628215` (wfn_loader.py, comment "matches common.bispinor_init").

### I/O
None.

### Flags consumed
None directly (whether the lift runs is a wfn_loader/bispinor-mode concern).

### Suspects
- **dead_suspects:** none strictly (test-only consumer), but production-dead: grep for
  `get_small_psi_component` across src/tests/tools/scripts finds only
  tests/test_wfn_loader_eager.py and docstring cross-references in wfn_loader.py.
- **redundancy_suspects:** the entire module duplicates the math of
  `file_io/wfn_loader.py`'s `_apply_bispinor_lift`/`_bispinor_lift_kernel` — a
  classic legacy/new parallel pair, including the duplicated magic constant
  `0.00364867628215` (`halfalpha` here, `_HALFALPHA` there). Candidate: keep one as
  the single source (e.g. move the scalar into a shared constant) or fold the oracle
  into the test file.
- **weird_code:**
  - line 30: hardcoded `0.00364867628215` = α/2 with no citation of α or units
    (Hartree atomic units implied); duplicated verbatim in wfn_loader.
  - Non-relativistic kinetic-only σ·p (no V_NL commutator, no Σ) — acknowledged in
    the docstring as a possible improvement, i.e. the "bispinor" states are a
    leading-order kinetic-balance lift only.
