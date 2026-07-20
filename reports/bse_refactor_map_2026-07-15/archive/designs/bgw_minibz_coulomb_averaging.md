# Reference: BerkeleyGW mini-BZ Coulomb cell-averaging (wcoul0), and the per-Q generalization LORRAX needs

BSE refactor-map program, 2026-07-20. Read-only source study of
`sources/BerkeleyGW` (debug-read only, no edits). Units Rydberg,
`v(q+G) = 8π/|q+G|²` (BGW convention). Cross-referenced against the in-repo
digests `w_head_wings_interp.md`, `coulomb_sr_lr.md`, `arbitrary_q_bse.md §9.5`
and the live LORRAX code in `sources/lorrax_D`.

Purpose: document EXACTLY how BGW mini-BZ-averages the Coulomb head so LORRAX's
`eval_vq` can match it for arbitrary-Q exciton `V_Q`. Throughout, **what BGW
DOES** is separated from **what the owner's arbitrary-Q generalization REQUIRES**.

The five load-bearing citations:

1. `Common/minibzaverage.f90:35-90` — `minibzaverage_3d_oneoverq2`, the 3D
   analytic-sphere + MC cell average and `wcoul0 = <v>·ε⁻¹₀₀`.
2. `Common/vcoul_generator.f90:214-352` — mini-BZ Voronoi point generation
   (`qran`), `q0sph2` inscribed-sphere radius, and the per-q averaging trigger.
3. `Common/nrtype.f90:109-114` — `ncell=3`, `nmc=nmc_fine=2 500 000`,
   `nmc_coarse=250 000`.
4. `BSE/inread.f90:901-906` — `avgcut` auto = `1/TOL_ZERO` (average at EVERY q)
   for 3D-semiconductor no-truncation; else `TOL_ZERO` (average q+G=0 only).
5. `BSE/intkernel.f90:429-442, 882-896, 1103-1146` — the fine-q re-add:
   `w_eff = <v(q_fi)>_mBZ · ε⁻¹₀₀(q_fi)` per fine q, and the head/wing/body
   prefactor assembly.

---

## 0. TL;DR scheme summary

- BGW cell-averages the **bare** Coulomb `v(q+G) = 8π/|q+G|²` over the **mini-BZ**
  = the Wigner-Seitz (Voronoi) cell of the q-grid, `V_mBZ = (2π)³/(V_cell·N_k)`.
- The average is a **Monte-Carlo** integral over `nmc = 2 500 000` fixed-seed
  pseudo-random points folded into the Voronoi cell (`minibzaverage.f90:57-77`,
  `vcoul_generator.f90:265-289`).
- **Only for the true head** (`|q+G|² < 1e-12`) is the leading `1/q²` singularity
  handled **analytically** (Baldereschi-Tosatti) over the largest sphere inscribed
  in the mini-BZ, MC handling only the region outside that sphere
  (`minibzaverage.f90:55-81`). At **finite** `|q+G|` the average is **pure MC**
  with an adaptively reduced sample count — no analytic term, no screened head.
- The **screened** head `wcoul0 = <v>_mBZ · ε⁻¹₀₀(q→0)` (`minibzaverage.f90:83-85`)
  is a **single scalar**, computed only at exactly q+G=0.
- **The routine already accepts an arbitrary q-shift** (its `qk` argument is a
  general `q+G`). For 3D semiconductors BGW *already* mini-BZ-averages `v` at
  **every fine q** in `intkernel` (`avgcut = 1e12`, `intkernel.f90:429`). What is
  **not** generalized per-Q: the analytic Baldereschi sphere term and the screened
  `wcoul0` (both hard-gated to q+G=0). The finite-q screened head is instead
  rebuilt as `<v(q_fi)>_mBZ · ε⁻¹₀₀(q_fi)` with `ε⁻¹₀₀` from an interpolated
  `epsdiag` point cloud, **not** a fresh average of `v·ε⁻¹`.
- **Exchange head is NOT averaged — it is ZEROED** (`vbar`, `mtxel_kernel.f90:1107-1114`,
  `gx_sum.f90:49`). Cell-averaging (`wcoul0`) is a **direct/screened-W** construct only.

---

## 1. THE AVERAGING SCHEME

### 1.1 Which q-points get averaged? (the trigger)

The averaging condition, in `vcoul_generator` (`vcoul_generator.f90:390`):

```fortran
if (ekinx < avgcut .and. peinf%jobtypeeval==1) then   ! ekinx = |q+G|²
```

Two gates:

- **`jobtypeeval`** (`vcoul_generator.f90:48-61`): `0` = "dumb generator"
  (Epsilon — **never** averages, always uses finite non-averaged `v(q)`);
  `1` = "smart consumer" (Sigma / Absorption / Kernel — **averages**).
- **`avgcut`** (Ry): "if `|q+G|² < avgcut`, compute `<1/(q+G)²>`, else `1/(q+G)²`"
  (`vcoul_generator.f90:99-101`). Its value decides *how many* q+G get averaged:

| context | `avgcut` | file:line | effect |
|---|---|---|---|
| default (all trunc/screen except 3D-SC-notrunc) | `TOL_ZERO = 1e-12` | `inread.f90:902`, `inread_sig.f90:905` | average **only q+G ≈ 0** (the head) |
| **3D semiconductor, no truncation** | `1/TOL_ZERO = 1e12 Ry` | `inread.f90:903-904`, `inread_sig.f90:906-907` | average at **EVERY q+G** |
| coarse `kernel.x` (hard-coded) | `TOL_ZERO` | `kernel_main.f90:271` | coarse kernel averages **only q+G=0** |
| fine `intkernel.x` (uses `xct%avgcut`) | `1e12` (3D-SC) | `intkernel.f90:431` | averages **every fine q = k−k′** |
| Sigma per-q | `sig%avgcut`; `0` for sub-q | `sigma_main.f90:1640-1643` | head only (comment `1756-1757`: "should really be done for all |q+G|<avgcut, but for now only |q|<avgcut and G=0") |

**Net:** on the **coarse** grid BGW averages only the single q=0 head. On the
**fine** interpolation grid, for a 3D semiconductor with no truncation, BGW
mini-BZ-averages `v` at **every** fine q. This is the machinery the owner wants
to reuse for arbitrary Q. `TOL_ZERO = 1e-12` (`nrtype.f90:162`); the comment in
`vcoul_generator.f90:100` saying "default is TOL_SMALL" is stale — the code sets
`TOL_ZERO`.

### 1.2 The mini-BZ cell and its Monte-Carlo sampling

The mini-BZ is the **Wigner-Seitz/Voronoi cell of the q-grid**: the full BZ scaled
down by the q-grid, `V_mBZ = V_BZ / N_k = (2π)³/(V_cell·N_k)`. The sample points
`qran(3,nn)` are built **once** (guarded by `ifirst==1`) in
`vcoul_generator.f90:214-352`:

1. `nn = nmc = 2 500 000` (`vcoul_generator.f90:214`, `nrtype.f90:114`).
2. Fixed seed `5000` for reproducibility; discard the first `3·nn` draws (warm-up);
   draw `ran(3,nn)` uniform in `[0,1)`; MPI-broadcast so all ranks share them
   (`vcoul_generator.f90:233-247`).
3. `dd = 1/qgrid` — the mini-BZ is a `1/qgrid` fraction of the BZ
   (`vcoul_generator.f90:250`).
4. Cholesky the metric `bdot = Uᵀ U` (`:252-254`) for fast lengths.
5. **Voronoi fold:** for each raw point `qpp = ran(:,jj)`, search replicas
   `i1,i2,i3 ∈ [−ncell+1, ncell]` with `ncell=3` (`nrtype.f90:109`), form
   `fq = qpp − (i1,i2,i3)`, keep the `fq` minimizing `|U·fq|²`; then
   `qran(:,jj) = dd · fq` (`:265-289`). Result: `qran` are points **uniformly
   distributed inside the q-grid Voronoi cell** (the mini-BZ), in crystal coords.

The **inscribed-sphere radius** `q0sph2` (squared) = smallest squared length among
the mini-BZ face centres `fq = (i1,i2,i3)·dd·0.5` over nonzero replicas
(`vcoul_generator.f90:296-312`, and the 2D-only `i1,i2` variant `:318-331` for slab).
`q0sph2` is the Baldereschi cut radius: the sphere in which `1/q²` is integrated
analytically.

### 1.3 Analytic vs Monte-Carlo split

`minibzaverage_3d_oneoverq2` (`minibzaverage.f90:35-90`) branches on
`length_qk = qk·bdot·qk = |q+G|²`:

- **True head, `|q+G|² < TOL_Zero`** (`:55-62, 79-81`):
  MC sums `1/|qk+δq|²` over all `nn=2.5M` points **but skips points inside the
  inscribed sphere** (`if length > q0sph2`); the sphere's contribution is then
  **added analytically** (§2.1). This is the classic Baldereschi-Tosatti scheme:
  analytic where `1/q²` is singular, MC where it is smooth.
- **Finite `|q+G|² > TOL_Zero`** (`:63-75`): **pure MC**, no sphere exclusion, no
  analytic term. The sample count is **adaptively reduced**:
  `nn2 = nint(nmc_coarse · 4·q0sph2 / |q+G|²)`, clamped to `[1, nn]`
  (`:68-69`; `nmc_coarse = 250 000`, `nrtype.f90:112`). Rationale (comment
  `:64-68`): 3D MC error `≈ σ/N^{3/2}`, `σ ∼ 1/ekinx^{3/2}`, so fixing
  `N ∼ 1/ekinx` gives ≈constant error per q; the anchor is
  `N(|q+G|² = 4·q0sph2) = nmc_coarse`.

So: **analytic term only at q+G=0; the sample count shrinks as `1/|q+G|²` away
from the zone centre.** MC uses fixed-seed pseudo-random points (not Sobol, not a
lattice rule).

---

## 2. THE EXACT FORMULAS

### 2.1 3D bulk, no truncation (`minibzaverage_3d_oneoverq2`, `minibzaverage.f90:35-90`)

Let `δq` range over the `qran` mini-BZ points, `V_mBZ = 8π³/(V_cell·N_k) =
(2π)³/(V_cell·N_k)`.

**Head (q+G = 0):**
```
<v>_mBZ  =  (8π / nn) · Σ_{δq : |δq|²>q0sph2}  1/|δq|²          ! MC, outside sphere
              +  32 π² · √q0sph2 / V_mBZ                        ! analytic, inside sphere
         =  (8π / nn) · Σ_{outside}  1/|δq|²
              +  4 · √q0sph2 · V_cell · N_k / π
```
(`minibzaverage.f90:77` for the MC prefactor `8π/nn2`, `:80` for the analytic term
with denominator `8π³/(celvol·nfk) = V_mBZ`, `celvol=V_cell`, `nfk=N_k`.)

Derivation of the analytic term (it *is* `<v>` over the inscribed sphere):
`∫_sphere (8π/q²) d³q = 8π·∫₀^{q0} (1/q²)·4πq² dq = 8π·4π·√q0sph2 = 32π²·√q0sph2`,
divided by `V_mBZ` to make it an average → line 80 verbatim.

**Screened head (only if `averagew` and q+G=0):**
```
wcoul0  =  <v>_mBZ · epshead ,     epshead = ε⁻¹(q→0; G=G'=0)
```
(`minibzaverage.f90:83-85`). A **single scalar**. For 3D metals/graphene the
`wcoul0` assignment differs (`vcoul_generator.f90:412-463`): metal uses
`wcoul0 = epshead·8π/q0²`, graphene `wcoul0 = epshead·<1/q>/q0len`.

**Finite q+G (`|q+G|² > TOL_Zero`):**
```
<v>_mBZ(q+G)  =  (8π / nn2) · Σ_{ii=1..nn2}  1/|q+G+δq|² ,
                 nn2 = clamp( nint(nmc_coarse·4·q0sph2/|q+G|²), 1, nn )
```
(`minibzaverage.f90:68-77`). **No** analytic term, **no** `epshead` multiply.

### 2.2 2D slab (`minibzaverage_2d_oneoverq2`, `minibzaverage.f90:97-186`)

Average is **in-plane only** — `δq` perturbs `qk(1:2)` (`:138`); the z-direction is
carried analytically by the slab-truncated Coulomb. `zc = π/√bdot₃₃` (half the
supercell height; `vcoul_generator.f90:759`). With `kxy = |in-plane(q+G+δq)|`,
`kz = |z(q+G+δq)|`, `length = |q+G+δq|²`:

**Bare (all q+G):**
```
<v_slab>_mBZ = (8π/nn) · Σ_δq [ 1 + e^{−kxy·zc}( (kz/kxy)·sin(kz·zc) − cos(kz·zc) ) ] / length
```
(`minibzaverage.f90:152-154, 178`). No inscribed-sphere split; pure MC over the
in-plane Voronoi cell (the slab `q0sph2` uses only `i1,i2`).

**Screened head (q+G=0, averagew)** uses the Ismail-Beigi model dielectric
(Sohrab, PRB 2006), NOT a plain `epshead` scalar (`:122-166, 180`):
```
vc_qtozero = (1 − e^{−q0len·zc}) / q0len²
gamma      = (1/epshead − 1) / (q0len² · vc_qtozero)
alpha      = 0
  per δq:  vc       = (1 − e^{−kxy·zc}) / kxy²
           epsmodel = 1 + vc·kxy²·gamma·e^{−alpha·kxy}
           integralW += vc / epsmodel
wcoul0 = (8π/nn) · integralW
```

### 2.3 1D wire and the `oneoverq` wing average

`minibzaverage_1d` (`minibzaverage.f90:223-376`) does a real-space
`K₀`-Bessel z-line integral (`nline=1000`) with the same Ismail-Beigi
`gamma` model for `wcoul0`. `minibzaverage_3d_oneoverq` (`:192-219`) is the
**wing** average `<1/q> = (8π/nn)·Σ 1/|q+G+δq|`, pure MC, used to re-attach the
`1/q` wing factor (`vcoul_generator.f90:364`). There is also
`minibzaverage_3d_oneoverq2_mod` (`:379-411`) for the hybrid-functional / TDDFT
range-separated Coulomb (a per-point `long/short_range_frac_fock` blend).

---

## 3. THE KERNEL APPLICATION (head/wing/body composition)

BGW's fine-grid BSE stores the coarse **direct** kernel as **head / wing / body**
with the divergent q-factors **stripped**, then re-attaches the analytic factors at
each fine q. (`mtxel_kernel.f90` coarse; `intkernel.f90` fine.)

### 3.1 Coarse kernel storage (`mtxel_kernel.f90:528-559`)

Per epsilon column `igp`, building `W(:,igp)` into `wptcol`:

- **Head** (`indinv(igp)==1`, `ig=1`), 3D **semiconductor**: `wptcol(1) = 1.0`
  (`:533`). The **whole** `v·ε⁻¹₀₀` is stripped to a bare placeholder `1.0`.
  Graphene/metal store `v·ε⁻¹·q` / `v·ε⁻¹` instead (`:534-547`).
- **Wings** (`calc_wings`, `:1016-1059`): 3D-SC no-trunc
  `wptcol = ph·epscol·vcoul·qq` — multiplied by `qq = |q|` so the stored quantity
  is `O(1)` (`:1038`). Truncated: no `·qq` (`:1041`). **q=0 wings are zeroed**
  (`:1043-1045`) — Baldereschi-Tosatti: the wing averages to zero at the zone
  centre.
- **Body** (`ig,igp ≠ 0`, `:555-558`): `wptcol(ig) = ph·conj(phinv)·epscol·vcoul(indinv(igp))`
  — the full `ε⁻¹_{GG'}·v(q+G')`, untouched (smooth through q=0).

`epshead = xct%epsdiag(ind(1),irq)` is the head of `ε⁻¹` at this coarse q
(`mtxel_kernel.f90:308`). These three go into `bsedhead`, `bsedwing`, `bsedbody`.

### 3.2 Fine-grid re-add (`intkernel.f90`)

At each fine `q_fi = k − k'`, `intkernel` (a) calls `vcoul_generator` with
`xct%avgcut` (`=1e12` for 3D-SC → **mini-BZ-averages at this fine q**) and (b)
interpolates `ε⁻¹₀₀`:

```fortran
! intkernel.f90:429-442  (once per distinct fine |q_fi|)
call vcoul_generator(..., dq_bz, ..., xct%avgcut, oneoverq, ..., xct%wcoul0)
vcoul_array(inew)    = vcoul0(1)/(8π)     ! = <v(q_fi)>_mBZ / 8π
oneoverq_array(inew) = oneoverq/(8π)      ! = <1/q_fi> / 8π   (wing factor)
```
```fortran
! intkernel.f90:855, 882-896   (per pair)
eps   = eps_intp(ik_cells)                ! ε⁻¹₀₀(q_fi), interpolated from epsdiag
vcoul = vcoul_array(iold)                 ! <v(q_fi)>_mBZ / 8π
w_eff = vcoul * eps                       ! = <v(q_fi)>_mBZ · ε⁻¹₀₀(q_fi) / 8π
if (abs_q2 < TOL_ZERO .and. SEMICOND)  w_eff = xct%wcoul0/(8π)   ! q_fi=0 → screened head
```
```fortran
! intkernel.f90:1103-1146   (prefactor by matrix type)
imatrix=1 (HEAD): bsemat_fac = fac_d · w_eff        ! head=1.0 → w_eff = <v>·ε⁻¹₀₀
imatrix=2 (WING): bsemat_fac = fac_d · oneoverq     ! restores stripped 1/q
imatrix=3 (BODY): bsemat_fac = fac_d                ! already complete
        fac_d = intp_coefs(ivert,ik) · intp_coefs(ivertp,ikp)   ! interp weights (:1146)
```

`eps_intp` is built by interpolating the `epsdiag` `ε⁻¹₀₀(q)` **point cloud**
(`intkernel.f90:337-347`, `interp_eval`), tabulated over `q+G` beyond the 1st BZ
(`epsdiag.f90`). So the fine-q screened head is `<v(q_fi)>_mBZ · ε⁻¹₀₀(q_fi)`, an
**interpolated** ε⁻¹ times a **mini-BZ-averaged** bare `v`.

**Answers to the owner's point 3:**
- The head is a **single scalar** `w_eff` per fine q multiplying the (interpolated)
  head matrix element (an `M`-density product). It is **not per-G** — it is the G=0
  head only.
- The body is per-G (`ε⁻¹_{GG'}·v(q+G')`); it **composes additively** with head and
  wing: `K^d = bsedhead·w_eff + bsedwing·oneoverq + bsedbody·1`, summed with the
  interpolation weights.
- At exactly q_fi=0 the head factor collapses to the single scalar `wcoul0/(8π)`.

### 3.3 Exchange kernel — head is ZEROED, not averaged

The exchange (`bsex`) uses the **bare** Coulomb `vcoularray` with the head
**zeroed** (`vbar`), not averaged:

```fortran
! mtxel_kernel.f90:689-691
call get_vcoul(.not.energy_loss, .true., xct%finiteq, xct%qpg0_ind)  ! finite-Q: zero smallest |−Q+G|
call get_vcoul(.true., .false.)                                       ! q=0: zero G=0
! get_vcoul (:1107-1114):  if (vbar)  vcoul(iqpg0 or 1) = 0
! gx_sum.f90:49 comment: "we are using the modified Coulomb potential here, where vbar(G=0)=0"
```

So **BGW does not mini-BZ-average the exchange head; it drops it** (the smallest
`|Q+G|` bare-Coulomb entry is set to 0). Even for **finite Q** the head is zeroed
(unless `energy_loss`). Cell-averaging (`wcoul0`) is exclusively a direct/screened-W
construct. *This is the single most important convention difference for the owner's
arbitrary-Q exchange `V_Q`* (see §5).

---

## 4. 2D SPECIFICS FOR MoS2-CLASS SLABS

- **What BGW's 2D average captures:** the slab-truncated integrand
  `8π·[1 + e^{−kxy·zc}(…)]/|q+G|²` (`minibzaverage.f90:152-154`). At `kz=0`
  (in-plane q) and small `kxy` it → `8π·zc/kxy`, i.e. the head goes like `1/|q_xy|`
  (a **cusp**, not a `1/q²` pole) — the correct 2D leading behaviour.
- **What it does NOT capture — the direction (winding):** the integrand depends only
  on the **magnitudes** `kxy = |q_xy|` and `kz`, never on the in-plane **angle**.
  BGW's 2D cell average is therefore **isotropic in the in-plane direction**. The
  averaging domain (the 2D Voronoi cell — a hexagon for MoS2) is anisotropic, but the
  integrand is not, so the directional structure is averaged away. The Qiu-Cao-Louie
  2D head — `A|Q|` intravalley, `A|Q|·e^{−i2θ}` intervalley (winding number 2),
  PRL 115 176801 / arXiv:1507.03336, transcribed in `arbitrary_q_bse.md §9.5` — is
  **not** reproduced by BGW's isotropic `minibzaverage_2d`.
- **Consequence:** at the single coarse `Q=0` point this is fine (Baldereschi-Tosatti
  makes the direction average out — this is *why* LORRAX's isotropic scalar passed the
  coarse MoS2 gates, `w_head_wings_interp.md:69`). Once Q is refined toward 0 —
  exactly the arbitrary-Q regime — the isotropic 2D cell-average is **wrong**, and the
  directional `e^{−i2θ}` head must come from the **rank-1** `g0(Q̂)⊗g0*(Q̂)` factor
  with a Q̂-rotating `g0` (the transition-dipole orientation), per
  `arbitrary_q_bse.md §9.5` and the discarded `S_cart` anisotropic generator in
  `w_head_wings_interp.md`.

---

## 5. PER-Q GENERALIZATION: WHAT BGW DOES vs WHAT THE OWNER NEEDS

### 5.1 Does `minibzaverage` already accept an arbitrary q-shift? — YES

`minibzaverage_3d_oneoverq2(nn, bdot, integral, qran, qk, …)` takes `qk` as a
**general** vector; in `vcoul_generator` it is `qk = gk + qvec_mod` = a full `q+G`
(`vcoul_generator.f90:382`). The routine internally branches on `|qk|²` (§1.3), so:

- **On the machinery level, per-Q averaging is already implemented.** For 3D
  semiconductors `intkernel` *already* calls it at **every** fine q
  (`intkernel.f90:429`, `avgcut=1e12`) and gets `<v(q_fi)>_mBZ`. The bare-`v`
  per-Q average is DONE in BGW; nothing is hard-coded to q=0 for the bare average.

What **is** hard-gated to `|q+G|² < TOL_Zero` (i.e. NOT generalized per-Q):

1. The **analytic Baldereschi sphere term** (`minibzaverage.f90:79-81`). At finite q
   the average is pure MC (correct — no singularity to subtract — but with the
   `nn2 ∼ 1/|q|²` adaptive count).
2. The **screened head `wcoul0 = <v>·epshead`** (`:83-85`). At finite fine q the
   screened head is instead rebuilt downstream as
   `w_eff = <v(q_fi)>_mBZ · ε⁻¹₀₀(q_fi)` (`intkernel.f90:887`), where `ε⁻¹₀₀(q_fi)`
   is **interpolated** from the `epsdiag` cloud — not a fresh `<v·ε⁻¹>` average.

### 5.2 What the owner's arbitrary-Q `V_Q` requires

For an exciton at centre-of-mass momentum **Q**:

- **Direct term** `W(q=k−k′)`: q still ranges over the (fine) grid; the head is at
  q→0 regardless of Q. BGW's existing per-fine-q averaging (`avgcut=1e12`) covers
  this directly — no new averaging logic, just the `intkernel` re-add pattern.
- **Exchange term** `V_Q`: the head is the **smallest-`|Q+G|`** bare-Coulomb entry.
  Here the owner **departs from BGW**: BGW **zeros** this entry (`vbar`, §3.3); the
  owner wants it **cell-averaged** so the exciton dispersion `E(Q)` is smooth through
  the near-singular small-`|Q+G|` region. The required call is exactly
  `minibzaverage_3d_oneoverq2` with `qk = Q + G*`, `G* = argmin_G |Q+G|_bdot`:
  - if `|Q+G*|² < TOL`: Branch A (2.5M MC outside the inscribed sphere + analytic
    sphere term). Only happens when `Q` sits on the grid and `G*` cancels it.
  - if `|Q+G*|² > TOL` (generic off-grid Q): Branch B (pure adaptive MC around
    `Q+G*`, `nn2 = nmc_coarse·4·q0sph2/|Q+G*|²`).

**So the per-Q generalization the owner needs is: (i) pick `G*` per Q, (ii) call the
same two-branch average around `Q+G*`, and (iii) inject the result as the head
instead of zeroing it.** Item (iii) is the real change from BGW; (i)–(ii) already
exist in the BGW routine.

---

## 6. RECOMMENDATION for LORRAX's arbitrary-Q `eval_vq`

### 6.1 Current LORRAX state (grounding)

- `Bulk3D.q0_average` (`sources/lorrax_D/src/gw/coulomb/bulk_3d.py:32-73`):
  `vc0_mean = <v>` and `wcoul0 = <v/(1−v·q̂ᵀSq̂)>` (anisotropic) or the
  Ismail-Beigi `gamma` fallback — over Sobol samples. **Pure quasi-MC; NO analytic
  Baldereschi sphere term.**
- `sample_minibz_qpoints` (`.../coulomb/base.py:107-162`): Sobol (default
  `nsamples=2¹⁸`, `qmc_reps=10` → ~2.6M points) folded to the q-grid Voronoi cell
  via `wrap_points_to_voronoi(..., nmax=1)`; uniform fallback bumps to 2.5M.
- Head injected as rank-1 `v_scalar·conj(g0)⊗g0` at **q=0 only**
  (`head_correction.apply_q0_head_rank1*`), with `g0_μ = ζ(q=0,μ,G=0)`; `V_qmunu`
  carries the body with G=G'=0 **zeroed** (`coulomb_sr_lr.md §Current state`) —
  BGW's strip-and-re-add pattern, already in place.

Three deltas vs BGW: LORRAX (a) uses Sobol not pseudo-MC, (b) omits the analytic
sphere term, (c) wraps with `nmax=1` vs BGW `ncell=3`, and (d) averages **only q=0**.

### 6.2 Concrete per-Q head scheme

**(1) Which G\* to average around.** Per exciton Q, `G* = argmin_G |Q+G|_bdot`
over the WFN G-sphere (the umklapp bringing Q into the 1st BZ). Average around
`qk = Q+G*`. For the **direct** W head keep the existing q→0 machinery unchanged;
generalize the **exchange** `V_Q` head. Reuse the *one* canonical SymMaps
`Q`-fold, do not add a parallel "shift Q" helper (`feedback_unified_sym_action`).

**(2) Analytic-vs-MC split.** Mirror BGW's two branches on `|Q+G*|²`:

```
if |Q+G*|² < TOL:                                   # Q+G* ≈ 0  (on-grid Q)
    <v>_mBZ = (8π/N)·Σ_{δq:|δq|²>q0sph2} 1/|δq|²    +   4·√q0sph2·V_cell·N_k/π
else:                                               # generic off-grid Q
    N_Q     = clamp(round(N_coarse·4·q0sph2/|Q+G*|²), 1, N)
    <v>_mBZ = (8π/N_Q)·Σ_{s=1..N_Q} 1/|Q+G*+δq_s|²
```

with `δq` the mini-BZ Voronoi points, `q0sph2` the inscribed-sphere radius, and
`V_mBZ = (2π)³/(V_cell·N_k)`. **Add the analytic sphere term to LORRAX** — it is
the piece LORRAX omits, and it is what makes the q→0 result seed-independent and
BGW-matchable. Compute `q0sph2` once from the q-grid + `bdot` (a `√`-free min over
face centres, `vcoul_generator.f90:296-312`).

**(3) MC/QMC sample count and distribution.** Keep Sobol (converges ≈`1/N` vs MC
`1/√N`, so LORRAX can use fewer points than BGW's 2.5M for the *outside-sphere*
MC), but:
- match BGW's Voronoi fold with `ncell=3`, not `nmax=1` — for skewed cells
  (hexagonal MoS2, layered VI3/CrI3) `nmax=1` can miss the true nearest replica.
  Change `wrap_points_to_voronoi` `nmax` to 3 (or auto from cell skew).
- for finite `Q+G*`, adopt BGW's adaptive count `N_Q ∼ q0sph2/|Q+G*|²` so far-from-Γ
  heads don't burn 2.6M points (they don't need them).
- because the analytic term now carries the singular sphere, the Sobol part only
  integrates the **smooth** `1/q²` tail outside the sphere — so 2¹⁸ points is ample;
  the current `qmc_reps=10` gives a free error bar.

**(4) 2D slab.** Use the `minibzaverage_2d` integrand
`8π·[1+e^{−kxy·zc}((kz/kxy)sin(kz·zc)−cos(kz·zc))]/|q+G|²`, `zc=π/√bdot₃₃`,
averaging **in-plane only** (`slab_2d.py`'s `f_2D` envelope already encodes the
truncation). **This is isotropic and will NOT give the winding-2 head** (§4); for
MoS2 finite-Q the directional `e^{−i2θ}` must come from the **rank-1** head
`v(Q)·conj(g0(Q̂))⊗g0(Q̂)` with a Q̂-rotating `g0` (the G=0 ζ-projection = transition
dipole), per `arbitrary_q_bse.md §9.5` and `w_head_wings_interp.md`. The cell-average
supplies the `|Q|` magnitude; the rank-1 `g0(Q̂)` supplies the angle. Do **not** try
to bake the winding into the scalar average — it cannot carry it.

**(5) Composition with the b26p SR/LR channel + SR body (no double-counting).**
Under the Gaussian range separation (`coulomb_sr_lr.md`):
`v(q+G) = v_SR + v_LR`, `v_SR(0)=2π/α²` finite (smooth over the mini-BZ),
`v_LR = 8π·e^{−|q+G|²/4α²}/|q+G|²` carries the whole singularity. Therefore:

```
<v(Q+G*)>_mBZ  =  <v_SR>_mBZ + <v_LR>_mBZ  ≈  v_SR(Q+G*) + <v_LR(Q+G*)>_mBZ
```

because `v_SR` is smooth ⇒ `<v_SR>_mBZ ≈ v_SR(Q+G*)` to MC accuracy. Rule to avoid
double-counting, given LORRAX zeros the head-G in the tile and injects it rank-1:

- Build the SR body tile from `v_SR` at **every** G (including `G*`) — it is finite
  and needs no averaging; it rides in `V^SR_q[μ,ν]` with no special-casing.
- **Cell-average ONLY the LR channel's head:** inject the rank-1 head
  `<v_LR(Q+G*)>_mBZ · conj(g0(Q))⊗g0(Q)` (direct term additionally times
  `ε⁻¹₀₀(Q)`, macroscopic-screening rule). Run the §6.2(2) two-branch average on
  `v_LR` (Gaussian-weighted `1/q²`), **not** on the full `v`.
- **Zero `v_LR(Q+G*)` in the tile's `G*` slot** (the singular entry) before the
  rank-1 add, exactly as the current q=0 code zeros `G=0`. Then
  `head_total = v_SR(Q+G*)  [in tile]  +  <v_LR(Q+G*)>_mBZ  [rank-1]`, with `v_LR`
  counted once. Equivalently, if you prefer to keep the full `v` tile: inject
  `(<v_LR(Q+G*)>_mBZ − v_LR(Q+G*))·g0⊗g0` (the mini-BZ *correction* to the singular
  entry). Pick one convention and gate it (G1 below).
- The `bare_coulomb_cutoff` (default ecutwfc) is applied to the full `v` after the
  split; it is harmless to `v_LR` (already `~e^{−G²/4α²}→0` past the Gaussian).

**(6) Direct vs exchange.** For the **direct** W head reuse the `w_eff =
<v(q_fi)>_mBZ · ε⁻¹₀₀(q_fi)` structure (`intkernel.f90:887`): mini-BZ-average the
bare `v` (or `v_LR`) at fine q and multiply by the scalar (or `S_cart`-anisotropic)
dielectric head. For the **exchange** `V_Q` head, inject
`<v(Q+G*)>_mBZ` (no ε⁻¹) — this is where LORRAX *diverges from BGW on purpose*: BGW
zeros it, the owner averages it. Keep `energy_loss`-style "don't zero" as the analog
of BGW's `.not.xct%energy_loss` guard.

### 6.3 Validation gates (1-GPU, BGW anchors)

- **G1 — coarse non-regression.** With per-Q averaging enabled but Q=0, MoS2 3×3 /
  Si 4×4×4 eigenvalues bit-reproduce today's isotropic-scalar result (proves the
  re-partition is behaviour-preserving before per-Q turns on).
- **G2 — analytic-sphere match.** LORRAX q=0 `<v>_mBZ` (MC-outside + analytic-sphere)
  matches BGW `minibzaverage_3d_oneoverq2` (`vcoul0(1)`) to ~1e-4 relative. This is
  the piece LORRAX currently omits; a direct numeric anchor from a BGW `vcoul.dat`.
- **G3 — finite-Q MC vs BGW.** For a handful of off-grid Q, LORRAX Branch-B
  `<v(Q+G*)>_mBZ` matches BGW's finite-q `vcoul_generator` (call it with
  `avgcut` large) to MC tolerance, including the adaptive `N_Q`.
- **G4 — 2D magnitude vs winding.** MoS2: cell-averaged `|Q|`-head magnitude matches
  `minibzaverage_2d`; the `e^{−i2θ}` angular structure appears only through the
  rank-1 `g0(Q̂)` (regression against Qiu-Cao-Louie `A|Q|e^{−i2θ}`,
  `arbitrary_q_bse.md §9.5`), NOT through the scalar.

---

## 7. Exhaustive file:line index

| Fact | file:line |
|---|---|
| 3D cell-average routine (analytic sphere + MC) | `Common/minibzaverage.f90:35-90` |
| — MC outside inscribed sphere (head) | `minibzaverage.f90:55-62` |
| — adaptive-count pure MC (finite q) | `minibzaverage.f90:63-75` |
| — analytic Baldereschi sphere term | `minibzaverage.f90:79-81` |
| — `wcoul0 = <v>·epshead` (screened head, scalar) | `minibzaverage.f90:83-85` |
| 2D slab cell-average (in-plane only) | `minibzaverage.f90:97-186` |
| — slab-truncated integrand | `minibzaverage.f90:152-154` |
| — Ismail-Beigi model-ε `wcoul0` | `minibzaverage.f90:122-166, 180` |
| wing average `<1/q>` | `minibzaverage.f90:192-219` |
| 1D wire (K₀ line integral) | `minibzaverage.f90:223-376` |
| range-separated `v` average (hybrid) | `minibzaverage.f90:379-411` |
| mini-BZ Voronoi point generation `qran` | `vcoul_generator.f90:214-289` |
| — fixed seed 5000 + warm-up | `vcoul_generator.f90:229-247` |
| — Voronoi fold over `ncell` replicas | `vcoul_generator.f90:265-289` |
| — inscribed-sphere `q0sph2` (3D / 2D) | `vcoul_generator.f90:296-312 / 318-331` |
| averaging trigger `|q+G|²<avgcut & jobtypeeval==1` | `vcoul_generator.f90:390` |
| 3D-SC per-q dispatch to `minibzaverage_3d` | `vcoul_generator.f90:394-410` |
| `jobtypeeval` 0=generator / 1=consumer | `vcoul_generator.f90:48-61` |
| `ncell=3`, `nmc=2.5e6`, `nmc_coarse=2.5e5` | `Common/nrtype.f90:109-114` |
| `TOL_Zero = 1e-12` | `Common/nrtype.f90:162` |
| `avgcut` auto = 1e12 for 3D-SC no-trunc | `BSE/inread.f90:901-906`; `Sigma/inread_sig.f90:904-907` |
| coarse kernel avgcut = TOL_ZERO (head only) | `BSE/kernel_main.f90:271` |
| coarse head stored `1.0` (SEMICOND) | `BSE/mtxel_kernel.f90:533` |
| coarse wings `·qq`, q=0 zeroed | `BSE/mtxel_kernel.f90:1038, 1043-1045` |
| coarse body `ε⁻¹·v(q+G')` | `BSE/mtxel_kernel.f90:555-558` |
| exchange head ZEROED (`vbar`) | `BSE/mtxel_kernel.f90:689-691, 1107-1114`; `gx_sum.f90:49` |
| fine-q `vcoul_generator` call (per q) | `BSE/intkernel.f90:429-442` |
| fine-q `w_eff = <v>·ε⁻¹₀₀`; q=0→wcoul0 | `BSE/intkernel.f90:882-896` |
| epsdiag ε⁻¹₀₀ interpolation | `BSE/intkernel.f90:337-347` |
| head/wing/body prefactor assembly | `BSE/intkernel.f90:1103-1146` |
| `fixwings` rescales ε for `W=ε⁻¹v` (Sigma) | `Common/fixwings.f90:61-168` |
| — head `epstemp = wcoul0/vcoul` | `Common/fixwings.f90:102-103` |
| LORRAX `q0_average` (Sobol, no analytic sphere) | `lorrax_D/src/gw/coulomb/bulk_3d.py:32-73` |
| LORRAX mini-BZ Voronoi sampler (`nmax=1`) | `lorrax_D/src/gw/coulomb/base.py:107-162` |

---

## 8. Cross-reference to in-repo digests

- `w_head_wings_interp.md` — the coarse→fine head/wing/body split and the
  (unplumbed) `head_wing_schur` kernel; confirms LORRAX collapses the anisotropic
  `S_cart` head to one scalar `wcoul0` and adds no wing.
- `coulomb_sr_lr.md` — the Gaussian SR/LR split that §6.2(5) composes with; confirms
  `v_SR(0)=2π/α²` finite and `v_LR` carries the whole singularity.
- `arbitrary_q_bse.md §9.5` — the 2D winding-2 head; confirms the isotropic
  cell-average cannot carry the `e^{−i2θ}` structure and the rank-1 `g0(Q̂)` must.
- `docs/docs_bgw/{epsilon,sigma}-overview.md` — the SX = (SX−X)+X partition that
  `vcoul_generator.f90:57-61` keeps `<v>` and `<v·ε⁻¹>` consistent for.
