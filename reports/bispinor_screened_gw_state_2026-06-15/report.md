# Bispinor GW in LORRAX: current state and the gap to screened bispinor W

**Date:** 2026-06-15 · **Checkout:** `sources/lorrax_C` @ `main` `e85be60` (rebased this session) · **Owner:** session-C

**Provenance:** Produced by a 7-agent read+synthesis workflow (`bispinor-gw-state-map`,
6 parallel readers over gw source + physics docs + status reports → synthesis),
cross-checked by direct reads of the load-bearing files, and confirmed against the
canonical theory report `reports/bispinor_theory_2026-05-09/report.md` (§7, §12) and
`sources/lorrax_B/docs/BISPINOR_DHFB_DESIGN.md`. All file:line anchors are in the
`lorrax_C` tree on `main`.

This report answers: **what does the bispinor pipeline compute today, and what is the
precise gap to full screened bispinor GW** (invert `(δ_{μν} − V_{μν'} χ_{ν'ν})` to get
the screened `W_{μν}`)?

---

## 0. Housekeeping done this session

- `sources/lorrax_C` was on `agent/si-band-sensitivity` (no unique commits vs old `main`
  `0f355b7`). `origin/main` had advanced 9 commits. Checked out `main`, fast-forwarded to
  `origin/main` `e85be60` (*JAX-0.9 + CPU MPI + backend-aware planner*). Clean and current.

---

## 1. Current bispinor pipeline — what runs today

Today the bispinor self-energy is a **bare, unscreened, static, Hartree-Fock-like
(DHF + bare-Breit) exchange and nothing else.** There is **no screened term, no
Coulomb-hole term, no χ₀, no W** on any bispinor/transverse channel. The theory report
§7.1 states this verbatim: *"no transverse screening, no retardation."*

The total bispinor exchange is the sum of two **bare** pieces:

1. **Charge-channel bare exchange (Σ^C-bare).** Literally the existing scalar bare Fock
   exchange: `sig_x = sigma_sx_k(wfns, Gij, V_q)` (`cohsex_sigma.py:228`), fed the **bare**
   Coulomb `V_q` (not `W`). Under `bispinor=True` it runs on 4-spinor wavefunctions but is
   otherwise the scalar kernel unchanged. The (0,0)/charge–charge Lorentz tile lives here.

2. **Transverse bare exchange (Σ^B).** Computed by `compute_sigma_x_bispinor`
   (`sigma_x_bispinor.py:103`), summed over the 9 transverse Lorentz pairs (i,j)∈{1,2,3}²:
   > Σ^B_{αβ} = −Σ_{i,j} γ̃^i_{αγ} G⁰_{γδ} γ̃^j_{δβ} D^{ij}_bare,  γ̃^μ ≡ γ⁰γ^μ
   > (γ̃⁰=I₄, γ̃^i=α^i), **D^{ij}_bare = V_qmunu_TT_{ij}** — the transverse-projector-weighted
   > **bare** Coulomb.

   Added onto `sig_x` at `cohsex_sigma.py:240-251`.

**Precision 1 — bare exchange only; no bispinor Coulomb-hole / screened-exchange term.**
Σ_SX and Σ_COH (`cohsex_sigma.py:90-106`) consume only the scalar charge `W_q`. There is no
Σ_SX^B or Σ_COH^B. The design self-describes as *"Phase-1 DHF + bare-Breit"*
(`sigma_x_bispinor.py:3`).

**Precision 2 — "maybe COHSEX, maybe PPM?": neither, in a subtle way.** Σ^B is **not
routed through the mode dispatcher.** It is added once in the single pre-dispatch call
`compute_cohsex_sigma(...)` at `gw_jax.py:359-365` — the only site threading
`wfns_transverse`/`bispinor_v_q_path`. Because Σ^B is folded into `sig_x` *before* the
`compute_mode` pivot, any mode (X_ONLY/COHSEX/GN_PPM/HL_PPM) that consumes that `sig_x`
inherits Σ^B's bare exchange — **but only via the one-shot path.**

> ⚠ **Latent correctness trap.** The dispatcher `compute_sigma_xc` (`sigma_dispatch.py:88`)
> has **no** bispinor parameters, and the self-consistent loop calls it without them
> (`sc_iteration.py:294`). **The QSGW/SC path silently drops Σ^B.** Not currently exercised
> (validated bispinor runs use `x_only=true, do_screened=false`), but a trap for any
> bispinor SC or dynamic run. The bispinor *restart* path is also explicitly unsupported
> (`gw_init.py:1318-1322`, `wfns_transverse` forced None).

---

## 2. The pair-density (μ/ISDF) basis and the bispinor channel structure

**Basis.** Everything — V, χ₀, W, Σ — lives in the ISDF centroid (μ) pair-density basis:
charge-density-weighted k-means centroids {r_μ}, n_μ ≈ 10·n_bands, Bloch-spinor pair
products interpolated onto ζ_{q,μ}(r). The bare Coulomb metric is the Hermitian outer
product V_{q,μ,ν} = Σ_G ζ̃*_{q,μ}(G) v_q(G) ζ̃_{q,ν}(G). This μ-basis matrix is what gets
inverted (§3), not a G-space ε.

**Channel structure (implemented).** The pair density carries a 4-vector **Lorentz index**
μ_L ∈ {0,1,2,3} from a Pauli decomposition: τ⁰=I (charge) and τ^{1,2,3}=(σ_x,σ_y,σ_z) (three
magnetization/current channels). **4 spin-density channels**, not 4 independent spinor
outer-products.

- **Charge channel (μ_L=0)** uses γ̃⁰=I — scalar machinery, short-circuited (no vertex).
- **3 transverse channels (μ_L=1,2,3)** use γ̃^i=α^i as **vertex insertions**, folded into ψ
  by left-multiplying `psi_xn` (left vertex) and `psi_yr` (right vertex) on the spin axis
  (`sigma_x_bispinor.py:62`); γ̃ are monomials (perm+phase), a gather+phase not a 4×4 matmul
  (`gamma_matrices.py:103`).

**Bispinor bare Coulomb — 7-tile sectorization (implemented).** `v_q_bispinor.py:57`
(`UNIQUE_TILES`): CC(0,0) + 3 TT-diag (i,i) + 3 TT-upper (i<j), each with a transverse weight
folded into v(q+G) per-G: CC t=1; TT-diag t = 1 − K̂_i²; TT-off t = −K̂_i K̂_j;
charge–current (0,i)/(i,0) t=0 (Coulomb gauge → `ZERO_TILES`). The (i,j) weights are exactly
the transverse projector t^{ij}(K)=δ^{ij}−K̂_iK̂_j.

**Docs intent vs. code.** The theory docs describe the same rank-5 open-spin pair density and
7-tile V_q — but describe **screening (the Dyson solve) only for the scalar charge channel.**
There is **no doc section** describing a bispinor χ with current-current channels or a screened
bispinor W. So the 4-channel structure is real and implemented **through bare V_q tiles**, and
stops there.

---

## 3. The charge-case screened-W machinery (the template)

The chain χ₀ → ε=(δ − Vχ₀) → invert → W **already exists and is wired — charge channel only.**

| Stage | Function / site | Notes |
|---|---|---|
| χ₀ build | `compute_chi0` (`w_isdf.py:568`) → `minimax_tau_integrate_chi` (`w_isdf.py:144`) | CTSP/minimax imaginary-time τ-scan; flat-q χ₀(nq,μ,μ). |
| **spin trace** | einsum `'Rambn,Rambn->Rmn'` (`w_isdf.py:181`) | G carries explicit spin axes (a,b); this **sums both**, collapsing to scalar density-density χ₀. **This is where channel structure is destroyed today.** |
| Bare V_q | `compute_all_V_q` (`compute_vcoul.py:846`) | Charge V^{0,0} tile only. |
| **Inversion** | `solve_w` (`w_isdf.py:354`) → `solve_one` (`w_isdf.py:250-253`) | `A = I − V[iq]@(pref·χ₀)[iq]; lu_factor(A); W[iq] = lu_solve((lu,piv), V[iq])` — per-q **LU**, one (n_μ×n_μ) matrix per q, in the μ basis (NOT G-space), via `shard_map`+`fori_loop`. |
| Alt backend | `_get_w_solve_fn_low_mem` (`w_isdf.py:269`) | Fused cuBLASMp/cuSOLVERMp FFI, W = X H⁻¹ X† via **Cholesky** on H=I−X†(pref·χ)X; needs HPD. |
| Prefactor | `_w_solve_pref_scalar` (`w_isdf.py:304-311`) | nspinor enters **only as a scalar rescale**, not as channels. |
| Freq planner | `screening_requests_for` (`screening.py:71`) + `compute_screening` (`screening.py:103`) | Single source of truth for which (ω,role) W's are built. |

**Static vs dynamic.** The inversion is **static per call** (single frequency). COHSEX calls it
once at ω=0. GN/HL-PPM call it **twice** (ω=0 + probe), then fit a **two-point plasmon-pole**
W^c=W−V elementwise on (q,μ,ν). Dynamic W(ω) is a *post-inversion parameterization*, not a
continuous ε⁻¹(ω). **Confirmed bispinor-blind:** grep for `bispinor|transverse|mu_L|gamma` in
`screening.py`/`ppm_sigma.py` returns nothing.

---

## 4. The gap to screened bispinor GW

Four components must be built or generalized.

**(a) Channel-resolved bispinor χ in the μ basis.** *Does not exist; partial analog.*
`compute_chi0` builds a single scalar χ₀ because the einsum `'Rambn,Rambn->Rmn'`
(`w_isdf.py:181`) sums both spin axes. The bispinor χ must **keep the spin axes uncontracted**
and project onto the Lorentz channels (μ_L,ν_L) → a (4n_μ)×(4n_μ) tensor (or 4×4-block of
(μ,μ)). `build_G_tau` (`greens_function_kernel.py:34`) already carries the spin axis; the
*contraction* changes. **Open:** which uncontraction pattern yields the desired channel χ, and
whether the FFT Hermitian-swap trick survives per-channel.

**(b) The (δ − V χ) assembly.** *Does not exist; charge analog is one line* (`w_isdf.py:251`).
Becomes channel-blocked: V is the 4×4 Lorentz tensor of (μ,μ) tiles (7 unique + gauge-zeros +
Hermitian fills, already on disk via `BispinorVqReader`, `v_q_bispinor.py:755`), χ from (a), and
the product mixes Lorentz indices. **Bare V tiles already exist** (`compute_V_q_bispinor_g_flat_to_h5`,
`v_q_bispinor.py:482`); missing piece is contracting them against a channel χ.

**(c) The inversion.** *Charge analog directly reusable, with a known conditioning caveat.*
Per-q solve at `w_isdf.py:250-253` generalizes to (4n_μ)×(4n_μ) (or block) LU. **Critical:** the
transverse CCT is **Hermitian but indefinite** — `isdf_fitting.py:1021` states *"the CCT is
Hermitian but indefinite — Cholesky NaNs"* (matches MEMORY: μ_L=i CCT indefinite → LU not
Cholesky). The zeta-fit already dispatches per-channel (Cholesky for charge, pivoted-LU+ridge for
transverse, `isdf_fitting.py:914-1080`). **Implication:** the bispinor W solve must use the **LU
backend** (`w_isdf.py:250-253`), **NOT the cuBLASMp Cholesky FFI** (`w_isdf.py:269`), unless the
channel-blocked ε is provably HPD — which it likely is not.

**(d) Σ_c contraction with the screened bispinor W.** *Does not exist; charge analog exists.*
Once a screened W^B exists, Σ_COH^B / Σ_SX^B (static) and Σ_c^B(ω) (dynamic) contract it through
the same γ̃ vertex insertions Σ^B already uses (`sigma_x_bispinor.py:62-100`). Templates: static
COHSEX (`cohsex_sigma.py:90-106`), PPM Σ_c (`ppm_sigma.py:530-555`). New part = feeding a
**screened** W instead of bare V, plus the COH/dynamic pieces.

**Summary of the delta:** today = bare V_q tiles + bare Σ^B (exchange only). Target = (new
bispinor χ) → (new channel-blocked δ−Vχ) → (LU inversion, reusing `solve_w`'s LU but NOT Cholesky)
→ (Σ_c^B contraction reusing the γ̃-fold). Two of four (inversion mechanics, vertex contraction)
have directly reusable analogs; the other two (channel χ, blocked assembly) are genuinely new,
gated on un-tracing the spin axes at `w_isdf.py:181`.

---

## 5. The physics design fork — decide before building

The phrase "invert (δ_{μν} − v_{μν'} χ_{ν'ν}) for the full screened W_{μν}" admits **two
physically distinct targets**, and the choice sets the entire scope. The in-tree theory does not
specify either, so this is the maintainer's decision.

**Option A — Charge-channel RPA screening + bare Breit (the conventional Dirac-GW choice).**
Screen only the charge–charge channel: scalar χ⁰ (already exists), W^CC = (1−V^CC χ^CC)⁻¹ V^CC,
used in charge Σ_SX/Σ_COH — while transverse Σ^B stays **bare** (first-order Breit, as in most
relativistic quantum chemistry, where the transverse photon is not RPA-screened). **Scope:** small.
Mostly = wire Σ^B through the dispatcher (§4 trap fix) so screened-charge + bare-Breit coexist, plus
a bispinor-charge χ⁰ (≈ the scalar one). The (a)–(c) channel-χ work is **not** needed.

**Option B — Full Lorentz-channel screened W (the literal δ−vχ over μ,ν = Lorentz⊗centroid).**
Build the channel-resolved χ (keep spin axes, 4 channels), assemble the 4×4-block (δ − Vχ)
including current-current channels, invert the full block matrix, contract the **fully-screened
transverse W** into Σ_c^B. Screens the Breit/transverse interaction itself — more complete and more
novel, and the bigger build (the entire §4 plan, with the indefiniteness risk of §4c).

A is a strict subset/prerequisite of B and is the physically standard first milestone.
**Recommended sequencing: A first (mostly-built, low-risk, immediately validatable against
scalar GW in the charge block), then B.**

---

## 6. Sequenced next steps (target-agnostic prefix, then per-option)

1. **Locate/confirm the governing design intent.** `BISPINOR_DHFB_DESIGN.md` lives in
   `sources/lorrax_B/docs/` (not in C); `reports/bispinor_theory_2026-05-09/report.md` is the
   canonical math ref. Neither specifies a screened bispinor W — confirm with the maintainer whether
   bare-Breit is the intended endpoint or a Phase-1 milestone (this is §5's fork).
2. **Wire Σ^B through the dispatcher (correctness fix, needed by both A and B).** Add
   `wfns_transverse`/`bispinor_v_q_path` to `compute_sigma_xc` (`sigma_dispatch.py:88`) and pass from
   `sc_iteration.py:294`, so the SC/QSGW path stops silently dropping Σ^B.
3. *(Option A)* Enable COHSEX/PPM for the bispinor **charge** channel: confirm the bispinor-charge
   χ⁰ reduces to the scalar χ₀, run `solve_w` on V^CC, feed screened W^CC into the existing charge
   Σ_SX/Σ_COH, keep Σ^B bare. Validate the charge block against scalar GW.
4. *(Option B)* **Un-trace the χ₀ kernel.** Generalize `minimax_tau_integrate_chi` (`w_isdf.py:144`)
   so the einsum `'Rambn,Rambn->Rmn'` (`w_isdf.py:181`) optionally retains spin axes and projects onto
   the 4 Lorentz channels. Validate the μ_L=0 block reduces exactly to scalar χ₀.
5. *(Option B)* **Assemble channel-blocked ε and invert with LU.** `A = I − V_tensor @ χ_tensor`
   over the Lorentz block structure, feeding the 7-tile bare V from `BispinorVqReader`. Generalize
   `solve_w`'s LU path. **Force LU; do not use the cuBLASMp Cholesky FFI** given indefinite transverse
   blocks (`isdf_fitting.py:1021`).
6. *(Option B)* **Contract Σ_c^B with the screened W**, reusing the γ̃-fold
   (`sigma_x_bispinor.py:62-100`) and static/dynamic kernels (`cohsex_sigma.py:90-106`,
   `ppm_sigma.py:530-555`).
7. **Replace the scalar 1/nspinor prefactor with per-channel weights** (`w_isdf.py:304-311`) once
   spinor structure is explicit.
8. **Test on a non-centrosymmetric SOC system before trusting IBZ.** The bispinor TRS spinor fix
   (i·σ_y·conj) is **not exercised** by CrI3 (spatial inversion → TRS never fires). Validate on e.g.
   1H-MoSe₂ or BiI₃ before relying on the IBZ cascade.

---

## 7. Open questions / risks

- **Is screened bispinor W even the intended target?** Code self-describes as "Phase-1 DHF +
  bare-Breit"; theory report §7.1 = *"no transverse screening, no retardation"*; §12's open questions
  are all bare-Breit-path bugs. The screened formalism is unspecified in-tree. **Resolve §5 fork
  first.**
- **Conditioning of the channel-blocked ε.** Transverse CCT confirmed indefinite
  (`isdf_fitting.py:1021`); the Cholesky FFI backend (`w_isdf.py:269`) is probably unusable, LU path
  mandatory. Verify empirically on a real bispinor χ, not by assumption.
- **Charge–current gauge tiles.** (0,i)/(i,0) tiles are zeroed by Coulomb gauge in the *bare* V
  (`v_q_bispinor.py:64`). Whether gauge mixing reappears in a *screened/dynamic* W is open; the
  bare-tile zeroing may not survive inversion against a coupled χ.
- **Periodic dynamic screening is itself not yet trustworthy (even for charge).**
  `SIGMA_FREQ_AUDIT_STATUS.md` (2026-03-31): scalar GN-PPM Σ_c agrees with BGW for CO (~6.5e-3 eV MAE)
  but ~3.7 eV MAE on periodic MoS₂. A screened bispinor Σ_c would inherit this. Fix the charge dynamic
  path first (status on a newer branch uncertain).
- **The freqint rewrite is not in-tree.** The clean-room PoleBlock CTSP engine (FREQ_INTEGRATION docs)
  is a blueprint whose code is absent from all A/B/C/D checkouts; its sigma mode is a
  `NotImplementedError` placeholder. Do not assume it as a foundation.
- **CrI3 transverse blowup (bare path).** Even bare Σ^B is ~10⁵× too large on CrI3 (theory §10.4),
  narrowed to current-centroid conditioning. A screened path inherits whatever centroid pathology this
  is — worth resolving on the bare path before screening.
