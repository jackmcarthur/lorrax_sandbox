# BerkeleyGW fine-grid BSE machinery — reference deep-read (bgw_fine_grid_reference)

Audit date: 2026-07-15, for the BSE refactor-map program. Sources read at
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/BerkeleyGW` (read-only reference) and
`/pscratch/sd/j/jackm/lorrax_sandbox/docs/docs_bgw/`. This is a REFERENCE document
for design agents: it documents how BGW implements the coarse→fine BSE
machinery that LORRAX (`src/bse/`) currently lacks. All formulas are written
**as they appear in the code**, with explicit indices. BGW conventions
throughout: Rydberg units, `v_bare(q+G) = 8π/|q+G|²`, Fortran 1-based
column-major arrays.

## Purpose

BGW splits the BSE into two executables:

1. **`kernel.x`** (`BSE/kernel.f90:15` → `kernel_main.f90`) computes the
   electron–hole kernel on the **coarse** k-grid of `WFN_co` and writes
   `bsedmat`/`bsexmat` (Fortran binary) or `bsemat.h5`, with the direct term
   stored in **three separately-normalized pieces — head, wing, body** — so the
   diverging q-dependence can later be re-applied analytically at fine q.
2. **`absorption.x`** (`BSE/absorption.f90:17` → `absorption_main.f90:23`,
   dispatching to `diag.f90` or `haydock.f90` at `absorption_main.f90:106-110`)
   expands the **fine**-grid wavefunctions (`WFN_fi`, `WFNq_fi`) in the coarse
   ones (`intwfn.f90` → dcc/dvv coefficients), interpolates the coarse kernel
   onto every fine (k,k′) pair (`intkernel.f90`), re-attaches the analytic
   `v(q)·ε⁻¹(q)` singular factors at the fine q = k−k′, builds
   `hbse_a` (+`hbse_b` for non-TDA), and diagonalizes.

The physics being interpolated (Rohlfing & Louie PRB 62, 4927 (2000), Eqs.
34-35, 39, 42-46; restated in `docs_bgw/kernel-overview.md:99-111`):

```
K^x(vck,v'c'k'|Q) =  Σ_G  M*_vc(k,Q,G)  v(Q+G)   M_v'c'(k',Q,G)
K^d(vck,v'c'k'|Q) = −Σ_GG' M*_cc'(k,Q,G) W_GG'(q) M_v'v(k+Q,q,G'),  q = k'−k
M_nn'(k,Q,G) = <n k+Q| e^{i(Q+G)·r} |n' k>
```

Category: **external reference — coarse→fine BSE pipeline (interpolation, singular-term re-add, finite-Q, CSI subsampling)**.

## Entry points (grep evidence)

| symbol | file | callers (grep evidence) |
|---|---|---|
| `program absorption` | `BSE/absorption.f90:17` | executable; calls `absorption_main` |
| `program kernel` | `BSE/kernel.f90:15` | executable; calls `kernel_main` |
| `program inteqp` | `BSE/inteqp.f90:19` | executable; eqp-only interpolation, reuses `intwfn` (`inteqp.f90:133`) |
| `intwfn` | `BSE/intwfn.f90:74` | `diag.f90:261`, `haydock.f90:171`, `inteqp.f90:133` |
| `intwfn_sub` | `BSE/intwfn.f90:1078` | `diag.f90:283` (only under `xct%subsample_line`) |
| `mtxel_t` | `BSE/mtxel_t.f90:36` | `intwfn.f90:529,533`, `intwfn.f90:1544,1548` |
| `intkernel` | `BSE/intkernel.f90:69` | `diag.f90:583,588,593,598`, `haydock.f90:518` |
| `interpolate` (private) | `BSE/intkernel.f90:1266` | `intkernel.f90:991,996,1003,1008,1017,1022,1029,1034` |
| `epsdiag` / `read_epsdiag` / `write_epsdiag` | `BSE/epsdiag.f90:46` | `intkernel.f90:195-197` |
| `epscopy` (BSE flavor) | `BSE/bse_epscopy.f90:37` | kernel side only — loads full ε⁻¹ columns into `xct%epscol`/`xct%epsdiag` for `mtxel_kernel` |
| `mtxel_kernel` | `BSE/mtxel_kernel.f90:74` | `kernel_main.f90` main k-pair loop |
| `vcoul_generator` | `Common/vcoul_generator.f90:62` | `intkernel.f90:429,434`, `kernel_main.f90:279,309,338` |
| `minibzaverage_3d_oneoverq2` | `Common/minibzaverage.f90:35` | `vcoul_generator.f90:399,405,414,447` |
| `fixwings`/`fixwings_dyn` | `Common/fixwings.f90:61,176` | **Sigma only** (`Sigma/sigma_main.f90:1797-1826`); grep over `BSE/*.f90` finds zero call sites — the BSE codes do their own wing scaling in `mtxel_kernel.f90` (coarse side) + `intkernel.f90` (fine side) |
| `program setup_subsampling_csi` | `BSE/setup_subsampling_csi.f90:32` | standalone tool that emits `subsample.inp` scaffolding |

---

## 1. Coarse→fine wavefunction interpolation (`intwfn.f90` + `mtxel_t.f90`)

### 1.1 The dcc/dvv expansion (R&L Eq. 39)

Definition (docstring `intwfn.f90:5-27`): the fine state is expanded over the
**closest** coarse k-point(s):

```
|ic, ik_fi>  ≈  Σ_jc  dcc(ic,jc,is,ik_fi,ivert) |jc, jk_co>,  jk_co = fi2co_wfn(ivert,ik_fi)
|iv, ik_fi>  ≈  Σ_jv  dvv(iv,jv,is,ik_fi,ivert) |jv, jk_co>
```

Array shapes (`intwfn.f90:85-88`): `dcc(ncb_fi, n2b_co, nspin, nkpt_fi,
npts_intp_kernel)`, `dvv(nvb_fi, n1b_co, nspin, nkpt_fi, npts_intp_kernel)`.
`npts_intp_kernel` is 1 by default and `idimensions+1` under
`kernel_k_interpolation` (`inread.f90:149,482`, `bse_init.f90:168`).

Per-element coefficient formula, exactly as computed (`mtxel_t.f90:179-188`
inside `mtxel_dmm`, driven by the G-loops at `mtxel_t.f90:76-118`):

```
dmm(im_fi, im_co, is) = Σ_{ig=1..wfn_fi%ng} Σ_{ispinor=1..nspinor}
    wfn_fi%cg(ig, im_fi, is·ispinor) · conj( wfn_co%cg(ig_co, im_co, is·ispinor) )
```

where `ig_co` maps the fine G-vector **plus the umklapp vector** into the
coarse sorting (`map_ig`, `mtxel_t.f90:132-163`): `G_co = G_fi + G_umk` with

```
G_umk = nint( k_fi − k_co )              (intwfn.f90:476-478, findvector)
```

resolved through the FFT-grid address `kaddn = ((kn1−1)·N2 + kn2−1)·N3 + kn3`
(`mtxel_t.f90:151-158`); out-of-sphere targets are skipped (`ig_co=0`,
`mtxel_t.f90:153-155,161`). This equals
`<m_co,k_co| e^{i(k_co−k_fi)·r} |m_fi,k_fi>` (comment `mtxel_t.f90:72`), i.e.
`<u_co|u_fi>` of the periodic parts — the residual smooth offset
`k_fi − k_co − G_umk` is **neglected** (the "restriction to close k-points"
approximation).

### 1.2 Which blocks are computed — restricted vs unrestricted

`mtxel_t` fills four blocks (`mtxel_t.f90:74-118`):

- `v_fi ← v_co` : always (`:76-81`).
- `v_fi ← c_co` : only if `qflag==1 .and. .not.restricted` (`:82-90`).
- `c_fi ← c_co` : always (`:93-100` restricted, `:102-108` unrestricted).
- `c_fi ← v_co` : only if unrestricted and `qflag==1` (`:109-117`).

`restricted = .not. xct%unrestricted_transf` (call sites `intwfn.f90:529-535`).
With `unrestricted_transformation`, `n1b_co = n2b_co = nvb_co + ncb_co`
temporarily (`intwfn.f90:138-144`); if the kernel itself is restricted
(`truncate_coefs = unrestricted .and. .not.extended_kernel`,
`intwfn.f90:145`), the coefficients are **truncated back** after the norm
check:

```
dvv = dvn(:, 1:nvb_co, ...)   ;   dcc = dcn(:, nvb_co+1:, ...)     (intwfn.f90:991-994)
```

or zeroed under `zero_unrestricted_contribution` (`intwfn.f90:995-998`).
Afterwards `n1b_co/n2b_co` are reset to depend only on `extended_kernel`
(`intwfn.f90:1002-1008`). Note: with `extended_kernel` the coarse band axis of
dvv/dcc is the **stacked** ordering `(vbm, …, vbm−nvb_co+1, cbm, …,
cbm+ncb_co−1)` — valence counted downward first (`intkernel.f90:1060-1068`).

### 1.3 Vertex selection, weights, and renormalization

- Closest coarse points: Delaunay tessellation (`interp_init`/`interp_eval`,
  `intwfn.f90:316-318, 383`) or legacy greedy `intpts_local`
  (`intwfn.f90:385-386`; flag `greedy_interpolation` → `delaunay_interp=.false.`,
  `inread.f90:311`). Returns up to 4 `closepts` + barycentric `closeweights`.
- Only vertex 1 feeds the kernel unless `kernel_k_interpolation`; then
  `intp_coefs(:,ik_fi) = closeweights(1:npts)` (`intwfn.f90:408-418`).
- QP-energy interpolation transform (`kpt_interpolation_exp_transform`,
  default): `closeweights = exp(1 − 1/|closeweights|)` where nonzero, then
  normalized (`intwfn.f90:443-450`) — preserves parabolic band extrema.
- eqp shifts are interpolated with **|d|² weights** (`intwfn.f90:555-641`):
  `evshift(iv_fi,ik_fi,is) += dweight · Σ_iv_co |dvn(iv_fi,iv_co)|²·evshift_co(iv_co,·)/tempsum`,
  including coarse-conduction contributions for `iv_co > nvb_co`
  (`intwfn.f90:567-571`); `tempsum=1` when `no_renormalization`
  (`renorm_transf=.false.`, `inread.f90:351`).
- Norm diagnostics: `normv(ik,iv,is,ivert) = Σ_co |dvn|²` written to
  `dvmat_norm.dat` / `dcmat_norm.dat` (`intwfn.f90:908-972`); warning if any
  2-norm < 0.5 (`warn_norm`, `intwfn.f90:1794-1826`).
- Renormalization (default on): `d → d/√(Σ_co |d|²)` per fine state
  (`intwfn.f90:1013-1038`).
- `dtmat` caching: written incrementally (`intwfn.f90:336-344, 774-820`),
  reread wholesale under `read_dtmat` (`intwfn.f90:165-252`).
- `skip_interpolation`: `dvn/dcn = identity`, `fi2co_wfn` by exact k-match
  (`intwfn.f90:279-311`).

### 1.4 Spin / spinor handling in the expansion

Spinor components are **summed inside** each coefficient (loop `ispinor=1..nspinor`
with the composite index `is·ispinor`, `mtxel_t.f90:182-185`) — dcc/dvv carry a
spin index `is=1..nspin` but no spinor index. Same composite `is*isp` indexing
appears in the subsampled reader (`intwfn.f90:1383-1411`). For `nspinor=2`,
`nspin=1` so `is·ispinor ∈ {1,2}` walks the two spinor components.

### 1.5 Fine-grid inputs: which file supplies which bands

`genwf` calls at `intwfn.f90:504-507`: conduction bands from `kg_fi` (`WFN_fi`),
valence bands from `kgq_fi` (`WFNq_fi`, index `ikq_fi = indexq_fi(ik_fi)`). With
the momentum operator `WFNq_fi` is not needed and `kgq == kg`
(`docs_bgw/absorption.inp:124-152`). The shifted-grid map is built in
`input_fi_q.f90:320-361`:
`qq = kg%f(:,ik) − (kgq%f(:,ikq) − xct%shift − xct%finiteq)` must be integer
(umklapp) for a match (`input_fi_q.f90:332`); `xct%qshift =
|shift + finiteq|_bdot` (`input_fi_q.f90:257-262`).

---

## 2. What `kernel.x` stores: the head/wing/body/exchange split (`mtxel_kernel.f90`)

Strategy comment (`mtxel_kernel.f90:377-386`): the direct term is
`bsed(iv,ivp,ic,icp) = Σ_{ig,igp} [Mvvp(igp)]* · W(ig,igp) · Mccp(ig)`, built by
looping over distributed ε⁻¹ columns `igp`, forming one column `wptcol(:)` of W
at a time, then `w_sum` (partial sum over igp) and `g_sum` (final sum over ig).

Two G-orderings coexist (`mtxel_kernel.f90:241-253`): gvec-space (`M12`,
`tempw/b`, `wptcol`, `vcoul`) vs eps-space (`epscol`); maps `ind`/`indinv` with
phases `ph`/`phinv` built by `gmap` (`mtxel_kernel.f90:265-294`). The ε⁻¹ head
value for the current q comes from `xct%epsdiag(ind(1),irq)`
(`mtxel_kernel.f90:307-308`); full columns from `xct%epscol` loaded by
`bse_epscopy.f90` (raw copy of `eps0mat`/`epsmat`, **no** fixwings-style
rescale — `bse_epscopy.f90:389-446`).

**What goes into `wptcol` (the W column) — this defines what is stored and
therefore what is smooth enough to interpolate** (comment
`mtxel_kernel.f90:517-525`; code `:528-559` and `calc_wings` `:1016-1059`):

| piece | condition | stored value (as written) |
|---|---|---|
| head (`ig=1, igp=head`) | SEMICOND (any trunc) | `wptcol(1) = 1.0d0` (`:532-533`) — the whole `v(q)ε⁻¹₀₀(q)` factor is stripped; re-applied on the fine grid |
| head | GRAPHENE, no trunc | `wptcol(1) = vcoul(1)·epscol(ind(1))·qq` (q=0: `·q0len`) (`:536-541`) — smooth `W·q` stored, `1/q` re-applied |
| head | GRAPHENE, trunc | `1.0d0` (`:543`) |
| head | METAL | `vcoul(1)·epscol(ind(1))` (`:545-547`) — W is finite, stored whole |
| wings (`g≠0,gp=0` and `g=0,gp≠0`) | SEMICOND, no trunc | `wptcol(ig) = ph(ig)·conj(phinv(igp)) · epscol(ig_eps) · vcoul(igp_gvec) · qq` (`:1038`) — since ε⁻¹wing ∝ q and v(q)=8π/q², the product `ε⁻¹·v·q` is O(1); the missing `1/q` is re-applied on the fine grid |
| wings | SEMICOND, trunc | `ph·conj(phinv)·epscol·vcoul` — **no** `·qq` (`:1039-1042`); the truncated wing is already smooth (wing DEFINED as smooth·q·Vtrunc(0,q)·ε⁻¹₀₀, comment `:519-525`), nothing re-applied at fine q |
| wings | SEMICOND, q_co=0 | `ZERO` (`:1043-1046`) — Baldereschi-Tosatti zeroing; sign-odd wings average to 0 over the mini-BZ (`docs_bgw/kernel-overview.md:50-79`) |
| wings | METAL/GRAPHENE | `ph·conj(phinv)·epscol·vcoul` (wire+graphene q=0 → 0) (`:1049-1053`) |
| body (`g≠0,gp≠0`) | all | `wptcol(ig) = ph(ig)·conj(phinv(igp)) · epscol(ind(ig)) · vcoul(indinv(igp))` — full `W_GG' = ε⁻¹_GG'(q)·v(q+G')` (`:555-558`) |

`sinv==−1` (ik>ikp) conjugates the whole column (`:561-562`).

`w_sum_cpu` routing (`w_sum.f90:62-100`): for the head column
(`indinvigp==1`): `temph(i1,i1p,isv) = wptcol(1)·conj(m11p(1,i1,i1p,isv))` and
`tempw(ig≥2,...) = wptcol(ig)·conj(m11p(head))` (right wing); for other columns:
`tempw(1,...) += wptcol(1)·conj(m11p(igp))` (left wing) and
`tempb(ig≥2,...) += wptcol(ig)·conj(m11p(igp))` (body). `g_sum_TDA_cpu` then
contracts with `Mccp` (`g_sum.f90:124-162`):

```
bsedbody/wing(iv,ivp,ic,icp,isc,isv) = Σ_ig temp{b,w}(ig,iv,ivp,isv)·mccp(ig,ic,icp,isc)
bsedhead(...)                        = mccp(1,ic,icp,isc)·temph(iv,ivp,isv)
```

**Unscreened tail** (`calc_direct_unscreened_contrib`,
`mtxel_kernel.f90:844-887`): for every G with `ph(ig)==ZERO` — i.e. G-vectors
of the wavefunction cutoff that lie **outside the ε⁻¹ cutoff** — the bare
`vcoul(ig)·conj(m11p(ig,...))` is added to `tempb`. So the stored "body"
already contains W = v beyond `screened_coulomb_cutoff`; a LORRAX
implementation that truncates W at the eps cutoff without this bare tail will
disagree.

**Exchange** (`mtxel_kernel.f90:673-699`, `gx_sum.f90:53-75`):

```
bsex(iv,ic,ivp,icp) = Σ_G conj(mvc(G,iv,ic)) · vbar(G) · mvpcp(G,ivp,icp),  vbar(G=0) = 0
```

(`gx_sum.f90:53` multiplies `mvpcp` by `vcoul`; head zeroing via
`get_vcoul(vbar=.true., ...)` `mtxel_kernel.f90:690-691, 1107-1115`).

**Units/normalization stored on disk**: `vcoularray = vcoularray/(8π)`
(`kernel_main.f90:363`), so with `v_BGW(q+G) = 8π/|q+G|²` the stored matrix
elements carry `1/|q+G|²` — matching `bsemat.h5.spec:184` "Kernel matrix
elements times V/8·Pi·Ry". `intkernel` restores the physical prefactor
(`factor`, §3). The q=0 element of `vcoularray` uses the mini-BZ average at
`q0vec` with `ncoul=1` (`kernel_main.f90:336-341`).

**bsemat.h5 layout** (`bsemat.h5.spec:173-215`): datasets `mats/head`,
`mats/wing`, `mats/body`, `mats/exchange` (TDDFT: head/body/exchange/fxc,
`intkernel.f90:1606-1626`), rank 7 Fortran dims
`(flavor, n1b, n1b, n2b, n2b, nk·ns, nk·ns)`. The absorption reader hyperslabs
`offset(6) = (is−1)·nkpt_co`, `offset(7) = (ikprime−1) + (isp−1)·nkpt_co`
(`intkernel.f90:1649-1655`) and permutes `(iv,ivp,ic,icp)` →
`bsedmatrix(iv,ic,ivp,icp,is,isp,ik)` (`intkernel.f90:1677-1691`). Restricted
kernels: n1b=nvb (valence counted **downward** from Fermi), n2b=ncb (upward)
(`bsemat.h5.spec:186-199`) — the same valence-axis flip LORRAX already handles
(`src/bse/STATUS.md` "Index ordering").

---

## 3. Kernel interpolation onto the fine grid (`intkernel.f90`)

### 3.1 Prefactors

```
factor = −8π/(crys%celvol · xct%nktotal) · xct%scaling          (intkernel.f90:224)
singlet: fac_d = factor, fac_x = factor                          (:232-236)
triplet: fac_d = factor, fac_x = 0                               (:227-231)
local-fields: fac_d = 0, fac_x = factor                          (:237-241)
spinor:  fac_d = factor, fac_x = factor  (requires nspinor==2)   (:242-247)
```

Kernel combinations (docstring `intkernel.f90:22-25`): singlet `K_d + 2K_x`,
triplet `K_d`, local-fields `2K_x`, **spinor `K_d + K_x`** (the factor 2 is a
spin sum that spinor calculations perform explicitly).

### 3.2 ε⁻¹-head interpolation (epsdiag) and v(q) pre-tabulation

All fine transitions `dq = k − k′` are precomputed once
(`dqs(:,ik) = UNIT_RANGE(kg%f(:,ik) − kg%f(:,1))`, `intkernel.f90:306-309`;
patched-sampling builds the full regular grid instead, `:292-304`), hashed into
a cells structure (`cells_init`, `:319`), folded to the Wigner-Seitz 1st BZ
(`point_to_bz`, `:1712-1740` — brute-force over ±ncell images minimizing
`qq·bdot·qq`).

- **epsdiag** (`epsdiag.f90`): reads `eps0mat`/`epsmat` (or `epsdiag.dat`) and
  keeps only diagonal elements, stored as a point cloud over vectors
  `q + G` with `|q+G|² < emax = 1.5·max_i(bdot(i,i))` (`epsdiag.f90:100-113,
  447-476`). Rationale (comment `:455-460`): `ε⁻¹_q(G,G) = ε⁻¹(q+G,q+G)`, so
  the diagonal at neighboring `G ≠ 0` provides the head values beyond the first
  BZ needed to interpolate near zone boundaries. Only periodic directions keep
  extra G's (`keep_gvec`, `:462`). Output cached to `epsdiag.dat`
  (`write_epsdiag`, `:78-95`; reread with `read_epsdiag` flag,
  `intkernel.f90:194-198`).
- Per dq: `eps_intp(ik) = Σ_{ijk=1..idim+1} closeweights(ijk)·epsi%eps(closepts(ijk))`
  — Delaunay interpolation over the epsdiag cloud with `periodic=.false.`
  (`intkernel.f90:336-347`). Special case: truncation + SEMICOND + `|q|²<TOL` →
  `eps = 1.0` (`:331-334`).
- **v(q)**: `vcoul_generator` is called once per **unique |dq_bz|²**
  (deduplication via `dist_array`, `:397-449`) with `nkpt=xct%nktotal`,
  `ncoul=1`, `isrtrq=[1]` (G=0 only), yielding `vcoul_array(i) = vcoul0/(8π)`
  and `oneoverq_array(i) = oneoverq/(8π)` (`:429-442`).

### 3.3 The mini-BZ average and `wcoul0` (`Common/vcoul_generator.f90`, `Common/minibzaverage.f90`)

`vcoul_generator` MC-averages `v` over the mini-BZ (the BZ scaled by 1/N_k)
whenever `|q+G|² < xct%avgcut` (`vcoul_generator.f90:99-101, 390`). The
absorption-side default is **avgcut = ∞ for untruncated semiconductor
screening** and `TOL_ZERO` otherwise (`inread.f90:901-906`) — i.e. for a 3D
semiconductor **every** fine `v(dq)` is a mini-BZ average
`⟨8π/q²⟩_{mBZ(dq)}`, not the point value. (`docs_bgw/absorption.inp:279-283`
still claims "default is 1.0d-12" — stale vs code.)

`minibzaverage_3d_oneoverq2` (`minibzaverage.f90:35-90`), as written:

```
integral = (8π/nn2) · Σ_{ii=1..nn2} 1/|qk + qran_ii|²_bdot        (:56-75)
q=0 case: points with |q|² ≤ q0sph2 are skipped and replaced by the
analytic sphere integral:  += 32π²·√q0sph2 / ( 8π³/(celvol·nfk) )  (:79-81)
adaptive sampling: nn2 = nint(nmc_coarse·4·q0sph2/|qk|²), clamped   (:68-69)
wcoul0 = integral · epshead        (only at q=0 and averagew)       (:83-85)
```

So `wcoul0 = ⟨v⟩_mBZ · ε⁻¹₀₀(q0)` for 3D semiconductors; metals instead get
`wcoul0 = epshead·8π/q0len²` (`vcoul_generator.f90:456`), graphene
`epshead·⟨8π/q⟩/q0len` (`:417-422`). Doc comment `vcoul_generator.f90:113-120`.
`average_w` is **always on** in absorption (`inread.f90:127-128`;
`docs_bgw/absorption.inp:519-524` confirms "always done regardless").
`oneoverq` = mini-BZ average of `8π/|q|` at q=0 (`vcoul_generator.f90:362-374`),
else `vcoul(1)·qlen` (`:480-483`).

The random mini-BZ points `qran` are generated once and reused
(`vcoul_generator.f90:258-345`); `destroy_qran` at `intkernel.f90:466`.

Related but **not** in the BSE path: `Common/fixwings.f90` rescales ε⁻¹
head/wings so Sigma's `W = ε⁻¹v` averaging works
(`fixwings.f90:38-58`; note `:54` "oneoverq ... is actually 8*PI/q!"); called
only from `Sigma/sigma_main.f90:1797-1826`. Its untruncated-semiconductor wing
scalings (`epstemp(irow_G0)·oneoverq·q0len/(8π)` at `:86`,
`epstemp(i)·fact·oneoverq/(vcoul·q0len)` at `:94`, slab `zc` factor
`:139-162`) are the Sigma-side mirror of the BSE `calc_wings` convention.

### 3.4 Fine-q re-attachment of the singular factors

Inside the big (ikp, ivertp, ik, ivert) loop: `dq = kg%f(:,ik) − kg%f(:,ikp)`
(`intkernel.f90:841`), O(1)-mapped through the cells structure (`:846`), then

```
vcoul   = vcoul_array(vq_map(ik_cells))       ! ⟨v(dq)⟩/(8π)      (:882)
oneoverq= oneoverq_array(...)                 ! (1/|dq|)-like      (:883)
eps     = eps_intp(ik_cells)                  ! interpolated ε⁻¹₀₀ (:854-859)
w_eff   = vcoul · eps ;  at dq=0 & SEMICOND:  w_eff = xct%wcoul0/(8π)   (:886-890)
```

and the per-matrix prefactor (`intkernel.f90:1103-1146`):

| imatrix | screening/trunc | `bsemat_fac` |
|---|---|---|
| 1 head | SEMICOND | `fac_d · w_eff` — analytic `v(q_fi)·ε⁻¹(q_fi)` times the stored smooth `M*ccp(0)Mvvp(0)` |
| 1 head | GRAPHENE, no trunc | `fac_d · oneoverq` |
| 1 head | otherwise (incl. METAL) | `fac_d` (head stored fully screened) |
| 2 wing | SEMICOND, no trunc, theory=0 | `fac_d · oneoverq` — restores the 1/q stripped at coarse level |
| 2 wing | otherwise | `fac_d` |
| 3 body | BSE | `fac_d` |
| 4 exchange | BSE | `−fac_x`, `·2` if `nspin==1 .and. krnl≠SPINOR` (`:1137-1138`) |

then `bsemat_fac *= intp_coefs(ivert,ik)·intp_coefs(ivertp,ikp)` (`:1146`) and
the internal debug factors `exchange_fact`/`direct_fact` (`:1148-1155`).
`zero_q0_element`: `==1` skips the dq=0 contribution after interpolation
(`:915-920`), `==2` zeroes the q_co=0 kernel before interpolation (`:793-797`)
(cf. PRB 93, 235435 (2016) Fig. 7).

### 3.5 The `interpolate()` contraction — per-element formula

Coefficients for the primed point are pre-transposed
(`intkernel.f90:819-830`): `dcckp(jc,ic,js) = dcc(ic,jc,js,ikp,ivertp)`; for
finite-Q the valence coefficients are looked up at the **shifted** fine index:
`dvvkp(:,:,js) = transpose(dvv(:,:,js, xct%indexq_fi(ikp), ivertp))` (`:824-827`),
and the unprimed dvv passed is `dvv(:,:,:,xct%indexq_fi(ik),ivert)` (`:1008-1011`).

`interpolate(xct, bse_co, bse_fi, dcck, dcckp, dvvk, dvvkp, ivin, icin,
imatrix, krnl, resonant)` (`intkernel.f90:1266-1574`). The `ipar==1` chain
(`:1453-1565`), composed step by step (resonant branch):

```
mat_vfvc(iv,jvp,jc,jcp) = Σ_jv  dvvk(iv,jv)         · bse_co(jv,jc,jvp,jcp)   (:1479-1485)
mat_vfvf(iv,ivp,jc,jcp) = Σ_jvp mat_vfvc(iv,jvp,·)  · conj(dvvkp(jvp,ivp))    (:1490-1497)
mat_cfcc(ic,jcp,iv,ivp) = Σ_jc  conj(dcck(ic,jc))   · mat_cccc(jc,jcp,·)      (:1525-1531)
mat_cfcf(ic,icp,iv,ivp) = Σ_jcp mat_cfcc(ic,jcp,·)  · dcckp(jcp,icp)          (:1536-1543)
bse_fi(iv,ic,ivp,icp)   = mat_cfcf(ic,icp,iv,ivp)                             (:1558-1563)
```

Net per-element (resonant / TDA block), with `dcckp(jcp,icp) = dcc_at_k'(icp,jcp)`
and `dvvkp(jvp,ivp) = dvv_at_k'(ivp,jvp)`:

```
K_fi(iv,ic; ivp,icp | k,k') = Σ_{jv,jc,jvp,jcp}
      dvv_k(iv,jv) · conj(dcc_k(ic,jc))
    · K_co(jv,jc,jvp,jcp | jk,jkp)
    · conj(dvv_k'(ivp,jvp)) · dcc_k'(icp,jcp)
```

Conjugation pattern: conj on the **conduction** coefficient at k, conj on the
**valence** coefficient at k′ — consistent with
`K^d = <ck|·|c'k'> W <v'k'|·|vk>`.

**Coupling (anti-resonant) block** (`resonant=.false.`; comment
`intkernel.f90:1278-1281` "apply complex conjugation to dvvkp and dcckp and
switch the jvp and jcp indices in bse_co"):

```
K^B_fi(iv,ic; ivp,icp) = Σ  dvv_k(iv,jv)·conj(dcc_k(ic,jc))
    · K_co(jv,jc, jcp, jvp)                    ! primed indices SWAPPED (:1467-1472)
    · dvv_k'(ivp,jvp) · conj(dcc_k'(icp,jcp))  ! conj moved to the other pair (:1499-1505, 1544-1551)
```

only meaningful with `extended_kernel` (the `(vc)→(c'v')` blocks exist,
`bsemat.h5.spec:201-215`). The `ipar≥2` path (`:1367-1451`) is the same
contraction via `X(gemm)` with `dummy(jv,ic,jvp,icp) = Σ_jcp
dcckp(jcp,icb)·Σ_jc bse_co(jv,jc,jvp,jcp)·conj(dcck(ic,jc))` etc.

Spin-index resolution (`:1296-1365`): `nspin==1` → all indices 1; singlet:
direct term only for `js==jsp`, exchange for all `(js,jsp)`; triplet
(`nspin==2`): direct only, with the **valence coefficients taken from the
opposite spin channel** (`js_dvvk = js±1`, `:1307-1319`); spinor kernels are
`nspin=1,nspinor=2` and take the `nspin==1` path.

`skip_interpolation` bypass (`:1043-1091`): direct copy
`bsedmt(jv,jc,jvp,jcp,js,jsp,1) = bsedmatrix_loc(jv, [nvb_co+]jc, jvp+ivout−1,
[nvb_co+]jcp+icout−1, js,jsp, jkp_offset+jk)` — with the extended-kernel
stacked offsets, and the coupling block read as `(jv, nvb_co+jc,
nvb_co+jcp+icout−1, jvp+ivout−1)` (`:1079-1084`).

### 3.6 Assembly into the BSE Hamiltonian

```
ikcvs  = bse_index(ik, ic, iv, js)  = js + (iv−1 + (ic−1 + (ik−1)·ncb)·nvb)·nspin
                                                     (Common/misc.f90:627-652)
hbse_a(ikcvs, ikcvsd) += bsemat_fac · bsedmt(iv,ic,ivb,icb,js,jsp,1)   (intkernel.f90:1174-1195)
hbse_b(ikcvs, ikcvsd) += bsemat_fac · bsedmt(...,2)   (non-TDA)        (:1192-1195)
```

Diagonal part built in `diag.f90:466-530`:
`hbse_a(ikcvs,ikcvsd) = ecqp(icb,ik,is) − evqp(ivb,ik,is)` (`diag.f90:523`);
for `qflag==2` the valence energy is taken at `xct%indexq_fi(ik)`
(`diag.f90:489-495`). QP energies come from `eqp.dat`/`eqp_q.dat`
(`eqp_corrections`, `input_fi.f90:310-316`) or are themselves interpolated
from `eqp_co.dat` via the |d|² scheme (`eqp_co_corrections`, §1.3), or
scissors (`scissors_shift`, `input_fi.f90:306`).

### 3.7 Coarse-matrix distribution (BGW's "sharding")

- `hbse_a(nmat, peinf%nblocks·block_sz)`: rows global, columns distributed over
  owned blocks; block granularity by `xct%ipar` = 1 (k), 2 ((k,c)), 3 ((k,c,v))
  (`intkernel.f90:903-912`, `diag.f90:448-453,504-506`).
- Each rank keeps only the coarse `jkp` slabs it needs:
  `jkp2offset(jkp) ≠ −1` iff some owned fine `ikp` has `fi2co_bse(:,ikp)==jkp`
  (`intkernel.f90:715-728`); `bsedmatrix_loc` dims
  `(n1b,n2b,n1b,n2b,ns,ns,jkp_offset)` (`:729`).
- bsemat I/O is **rank-0 read + MPI_BCAST of a full k-slab per imatrix**
  (`intkernel.f90:744-800`, FIXME comment `:777-779`); Fortran-binary record
  layout `read(ifile) ikp,jcp,jvp, (((((bsedmatrix(jv,jc,jvp,jcp,js,jsp,jk)...`
  (`:770-774`).
- `wfn2bse`/`fi2co_bse` remap bsemat k-order to WFN_co k-order allowing lattice
  vector offsets (`:645-689`).

---

## 4. Finite-Q BSE (exciton center-of-mass momentum)

Flag: `exciton_Q_shift Qflag Qx Qy Qz` → `xct%qflag, xct%finiteq`
(`inread.f90:402-404`; same keyword in kernel.inp). Semantics
(`docs_bgw/absorption.inp:494-515`, `bsemat.h5.spec:153-167`):
`qflag=1` standard Q=0; `qflag=2` Q commensurate with the coarse grid (ψ_{vk+Q}
taken from `WFN_co` itself); `qflag=0` arbitrary Q read from `WFNq_co`
(deprecated/under development). **Sign convention**: the shift is applied to
the valence states, `|cvkQ> = |ck>|vk+Q>`, and the exciton COM momentum is
**−Q** (`docs_bgw/kernel-overview.md:93-111`,
`docs_bgw/absorption-overview.md:117-132`).

Kernel side (`kernel-overview.md:104-111`): `K^x = Σ_G M*vc(k,Q,G) v(Q+G)
M_v'c'(k',Q,G)`; `K^d = −Σ_GG' M*cc'(k,Q,G) W_GG'(q) M_v'v(k+Q,q,G)`.
Implementation:

- Exchange at Q≠0 uses `v(G−Q)`: "we end up evaluating `Σ_G [M_vc(G)]*
  v(G−Q) M_cpvp(G)`, so we need v at q = −Q = xct%finiteq" (comment
  `mtxel_kernel.f90:679-689`); `v(−Q+G)` is stored as the extra column
  `vcoularray(:,qg%nf+1)` (`kernel_main.f90:276-283`).
- The "head" of the exchange is the G closest to −Q: `xct%qpg0_ind = argmin_G
  |G+finiteq|²` (`kernel_main.f90:285-297`); it is zeroed unless `energy_loss`
  is set (`get_vcoul(.not.xct%energy_loss, .true., xct%finiteq, xct%qpg0_ind)`,
  `mtxel_kernel.f90:689`; "we should NEVER use finite Q without the
  energy_loss flag", comment `kernel_main.f90:287-289`). `energy_loss` recorded
  in `bse_header/params/energy_loss` (`bsemat.h5.spec:88-91`).
- Absorption side: valence WFNs come from the shifted file; `xct%indexq_fi`
  maps `k → k+shift+finiteq` (`input_fi_q.f90:320-361`); `intwfn` computes
  separate closepts for the shifted grid (`closepts_q`,
  `intwfn.f90:388-407`), separate umklapp `igumkq` (`:481-495`), and stores
  `dvn` at `ikq_fi` for `qflag≠1` (`:534`); `mtxel_t` disables v↔c mixing
  blocks for `qflag≠1` (`mtxel_t.f90:82,109`). `intkernel` indexes dvv at
  `indexq_fi` (`intkernel.f90:824-827,1008-1011`); the QP diagonal uses
  `evqp(iv, indexq_fi(ik))` for qflag=2 (`diag.f90:489-495`).
- Restriction: does not work with `use_symmetries_coarse_grid`
  (`docs_bgw/absorption.inp:511-513`).

---

## 5. Clustered-subsampling interpolation (CSI / `subsample_line`)

Purpose (`docs_bgw/absorption.inp:455-473`): in 1D/2D the screening varies too
fast at small q for the coarse grid to capture; for fine `|q| <
subsample_cutoff` (a.u.), the interpolated kernel is **replaced** by matrix
elements from separate subsampled `bsemat` files computed on radial clusters
of k-points around each coarse point. Assumes the kernel is **isotropic**
within each coarse Voronoi cell.

- `subsample.inp` (read at `intkernel.f90:489-548` and `intwfn.f90:1175-1190`):
  line 1 `nk_sub` (points per file), line 2 `nsub_files` (= nkpt_co enforced,
  `diag.f90:270-272`), then nsub_files coarse k-points, then nsub_files bsemat
  filenames (HDF5 only, `inread.f90:602-604`), then nsub_files WFN_sub
  filenames, then (qflag=0) WFNq_sub filenames.
- `intwfn_sub` (`intwfn.f90:1078-1792`) computes 6D
  `dcc_sub(ncb_fi,n2b_co,nspin,nkpt_fi,idim+1,nk_sub)` — expansions of each
  fine point over each **subsampled** k-point of each Delaunay vertex, reading
  WFN_sub serially per rank (peinf spoofing hack `:1257-1263`), with per-point
  `gmap` phase fixing (`:1443-1444`) and the same restriction/renorm tail
  (`:1723-1772`).
- In `intkernel` (`:924-1000`): find a Delaunay vertex **shared** by ik and ikp
  (`closepts_sub(ivert,ik) == closepts_sub(ivertp,ikp)`, `:936-949`;
  `subsample_algo` 0 = first found/reproducible, 1 = closest, `:174-180`), then
  pick the subsampled point whose radial distance from its coarse point best
  matches |q_fi|: `jk_sub = argmin_isub | kpoints_sub_len(isub) − √(abs_q2) |`
  (`:955-965`), and call `interpolate` with `bsedmt_sub(:,:,:,:,jkp_sub,jk_sub,
  imatrix)` and the `dcc_sub/dvv_sub` coefficients (`:985-1000`). The subsample
  bsemat is read as `<k_sub|K|k_co=1>` — coarse point is always the first
  k-point in each file (`bse_hdf5_read` with `subsample=.true.`,
  `:1598-1603`).
- Radial distances are validated identical across files (`:511-525`).
- Setup tool: `setup_subsampling_csi.x ASCII|BIN|HDF5 WFN_co WFN nk_fi_1..3
  nsub_factor direction [qshift use_syms]` emits `kpoints_sub_*.dat`,
  `epsilon_q0s.inp`, `kpoints_wfnq*.dat`, `subsample.inp`
  (`setup_subsampling_csi.f90:1-80`).
- Guards: HDF5 only; incompatible with `kernel_k_interpolation`
  (`inread.f90:600-608`).
- The related-but-distinct **`patched_sampling`** (PRB 108, 235117 (2023)) runs
  the fine grid on a patch of a regular grid (`intkernel.f90:292-304`,
  `inread.f90:295-296`), with `patched_sampling_co` for a patched coarse grid.

---

## 6. TDA vs full BSE

- `xct%tda` default `.true.` (`inread.f90:168`); `full_bse` sets
  `tda=.false.` and **forces** `extended_kernel=.true.` and
  `unrestricted_transf=.true.` (`inread.f90:618-623`).
- `intkernel` allocates `bsedmt(...,tda_sz)` with `tda_sz=2` for non-TDA
  (`intkernel.f90:584-586`) and dies if `hbse_b` is absent (`:182-183`); the
  coupling block is produced by the second `interpolate(...,.false.)` call
  chain (`:1014-1040`) and accumulated into `hbse_b` (`:1192-1195`).
- Solvers (`inread.f90:170-172, 392-400`; `docs_bgw/absorption.inp:435-453`):
  `BSE_NTDA_SOLVER_SSEIG` (default parallel; structure-preserving, Shao et al.
  LAA 488, 148 (2016)), `_GVX` (p/zhegvx), `_ELPA`. TDA path calls
  `diagonalize(xct,neig,nmat,hbse_a,...)`; non-TDA passes
  `hbse_b` and gets left+right eigenvectors (`diag.f90:677-681`). Iterative
  (PRIMME) diagonalization is TDA-only (`inread.f90:565`).
- Non-TDA oscillator strengths add negative-transition contributions with the
  documented sign rules `s_(c→v) = −s_(v→c)^*` (`diag.f90:720-738`).
- `zero_coupling_block` (internal) zeroes H_B for testing (`inread.f90:406-407`).
- Lanczos (`absp_lanczos.f90`) supports both TDA and full BSE
  (`docs_bgw/absorption.inp:245-249`).

## 7. Spin / nspinor summary

- `ns` (nspin) = 2 only for spin-polarized single-spinor; `nspinor=2` implies
  ns=1 (`bsemat.h5.spec:120-129`). `BSE_KERNEL_SPINOR` requires nspinor==2
  (`intkernel.f90:247`) and drops the singlet ×2 on exchange
  (`:1133,1138,1143`).
- dcc/dvv: spinor-summed, spin-indexed (§1.4).
- Kernel matrices stored per `(js,jsp)` pair; hyperslab index folds spin into
  the k axis (`nk·ns`, `intkernel.f90:1649-1655`).
- Triplet kernel = direct-only **with cross-spin valence coefficients**
  (`intkernel.f90:1307-1319`), valid only for spin-unpolarized ground states
  (`docs_bgw/absorption.inp:365-387`).
- Fine/coarse spin counts must match (`dtmat` check `intwfn.f90:208-211`).

## 8. Flags consumed (absorption.inp → this machinery)

Everything below verified against `BSE/inread.f90` (defaults at `:104-172`).

| flag | effect | evidence |
|---|---|---|
| `number_val/cond_bands_fine/coarse` | nvb_fi/ncb_fi/nvb_co/ncb_co | inp doc `:7-17` |
| `use_velocity` / `use_momentum` / `use_dos` | dipole operator; velocity pulls valence from WFNq_fi | `absorption.inp:124-152`, `intwfn.f90:506` |
| `no/use_symmetries_{fine,shifted,coarse}_grid` | BZ unfolding per file; **default: no unfold** | `absorption.inp:61-117` |
| `skip_interpolation` | identity dcc/dvv + direct copy path | `inread.f90:486`, `intwfn.f90:279`, `intkernel.f90:1041` |
| `delaunay_interpolation` / `greedy_interpolation` | interp scheme (default delaunay) | `inread.f90:124,311-313` |
| `kpt_interpolation_linear/exp_transform` | eqp weight transform (default exp) | `inread.f90:125,314-317`, `intwfn.f90:443-449` |
| `kernel_k_interpolation` (internal) | npts_intp_kernel = idim+1 vertex-weighted kernel interp | `inread.f90:481-482`, `bse_init.f90:168`, `intkernel.f90:1146` |
| `extended_kernel` | 4-block kernel; stacked coarse band axis | `inread.f90:328-329`, `intkernel.f90:1059-1091` |
| `unrestricted_transformation` / `zero_unrestricted_contribution` | dvn/dcn over all coarse bands / zero cross blocks | `inread.f90:330-333`, `intwfn.f90:138-157,991-998` |
| `no_renormalization` (internal) | renorm_transf=.false. | `inread.f90:350-351`, `intwfn.f90:1013-1038` |
| `eqp_corrections` / `eqp_co_corrections` | read eqp.dat / interpolate eqp_co.dat | `input_fi.f90:310-316`, `intwfn.f90:555-641` |
| `spin_singlet/spin_triplet/local_fields` (+spinor auto) | fac_d/fac_x table | `intkernel.f90:226-251` |
| `cell_average_cutoff` | avgcut for mini-BZ averaging; auto default ∞ (semicond, no trunc) else TOL_ZERO | `inread.f90:120,423-424,901-906` |
| `average_w` | wcoul0 = ⟨v⟩·ε⁻¹₀₀ at q=0 — always on in absorption | `inread.f90:127-128`, `minibzaverage.f90:83-85` |
| `zero_q0_element n` | zero W(q=0) after(1)/before(2) interpolation | `inread.f90:336-337`, `intkernel.f90:793-797,915-920` |
| `subsample_line cutoff` / `subsample_algo` | CSI replacement below cutoff | `inread.f90:338-343`, §5 |
| `exciton_Q_shift Qflag Qx Qy Qz` | finite-Q (qflag, finiteq) | `inread.f90:402-404`, §4 |
| `full_bse` (+`full_bse_solver_*`) | non-TDA; forces extended+unrestricted | `inread.f90:388-400,618-623` |
| `read_dtmat` / `read_epsdiag` / `read_vmtxel` / `read_eigenvalues` | stage caches | `inread.f90:415-418`, `intwfn.f90:165`, `intkernel.f90:194` |
| `kernel_scaling` | xct%scaling multiplies `factor`; 0 short-circuits intkernel | `inread.f90:487-488`, `intkernel.f90:159,224` |
| `exchange_factor`/`direct_factor` (internal) | debug multipliers | `inread.f90:410-413`, `intkernel.f90:1148-1155` |
| `energy_loss` (kernel.inp) | keep exchange head at finite Q | `mtxel_kernel.f90:689` |
| `screened_coulomb_cutoff`/`bare_coulomb_cutoff` (kernel.inp) | ecuts/ecutg; bare tail beyond ecuts added unscreened | `kernel.inp:11-21`, `mtxel_kernel.f90:844-887` |
| `dump_bse_hamiltonian`/`read_bse_hamiltonian` | hbse.h5 bypass of the whole interpolation | `diag.f90:571-573,654-656` |
| `patched_sampling`(`_co`) | non-uniform fine(coarse) grid patches | `inread.f90:295-296,352-353`, `intkernel.f90:292-304` |

## 9. Memory residency / parallel layout (MPI analogue of sharding)

- Replicated on every rank: `dcc`, `dvv`, `intp_coefs`, `fi2co_wfn` (Allreduce,
  `intwfn.f90:838-856`), `epsi%eps` point cloud (Bcast,
  `intkernel.f90:204-215`), `vcoul_array`/`eps_intp` (size nktotal doubles),
  and — when subsampling — the **entire** `bsedmt_sub`
  `(n1b,n2b,n1b,n2b,nkpt_co,nk_sub,4)` (Bcast `intkernel.f90:562-563`).
- Distributed: `hbse_a/hbse_b` columns (owned (k,c,v) blocks);
  `bsedmatrix_loc` restricted to needed coarse jkp slabs (§3.7). ε⁻¹ columns
  distributed block-cyclic in kernel.x (`NUMROC`/`INDXL2G`,
  `mtxel_kernel.f90:462-509`; `low_comm` replicates).
- GPU offload exists only in kernel.x inner loops (`mtxel_algo`, `w_sum_algo`,
  `g_sum_algo`, `kernel.inp:62-82`; `w_sum.f90:110+,268+`) and in the PRIMME
  matvec (`primme_algo`, `absorption.inp:236-243`). The interpolation itself
  (intwfn/intkernel) is CPU + BLAS.

## 10. Mapping to LORRAX (what is missing, where it would attach)

LORRAX `src/bse/` (lorrax_D @ agent/slate-linalg-ffi) is a **single-grid** BSE:
`apply_W` contracts the ISDF `W_q(μ,ν,q)` via 3D k-FFT
(`src/bse/bse_jax.py:124-149`), `apply_V` uses a single `V_q0` slab
(`bse_jax.py:98`), and the q=0 head is injected as one scalar matched to BGW's
`wcoul0` (`src/bse/BGW_COMPARE.md`, `vhead = 3303.748`; see also
`src/bse/STATUS.md` "Same head value"). Grep over `src/bse/*.py` finds no
coarse/fine interpolation of any kind. Missing vs this reference:

1. dcc/dvv wavefunction expansion + norm diagnostics (§1) — LORRAX analogue
   would be overlaps of ISDF-interpolated ψ between two k-grids.
2. Head/wing/body split of the ISDF `W_q` with smooth-part interpolation and
   analytic `v(q_fi)·ε⁻¹(q_fi)` / `1/q_fi` re-attachment (§2-3). Note the ISDF
   W_q is a (μ,ν) matrix, not G-space: the head/wing separation would have to
   happen in the G=0 projection of the ζ basis before compression.
3. epsdiag-style ε⁻¹₀₀(q) tabulation beyond the 1st BZ (§3.2).
4. Mini-BZ-averaged v(q) for **all** fine q (3D semicond) and the wcoul0
   convention already matched manually in BGW_COMPARE.md (§3.3).
5. Unscreened bare tail beyond the screened cutoff (§2, `mtxel_kernel.f90:844-887`).
6. Finite-Q (valence-shifted) kernel and the v(G−Q)/energy_loss exchange (§4).
7. CSI subsampling (§5) — least relevant until 2D systems are targeted.

## 11. Notes / discrepancies found (docs vs code)

1. `docs_bgw/absorption.inp:279-283` states `cell_average_cutoff` default is
   `1.0d-12`; the code default is `avgcut = 1/TOL_ZERO` (∞) for untruncated
   semiconductor screening and `TOL_ZERO` otherwise (`inread.f90:120,901-906`).
   Copying the documented default would silently disable the mini-BZ averaging
   of v(q) at every fine q for 3D semiconductors.
2. `fixwings` (`Common/fixwings.f90`) is **not** part of the BSE path — zero
   call sites under `BSE/` (only `Sigma/sigma_main.f90:1797-1826`). BSE wing
   handling lives in `mtxel_kernel.calc_wings` (coarse) + the
   `bsemat_fac`/`oneoverq` re-add in `intkernel` (fine). Design docs should not
   cite fixwings as the BSE wing mechanism.
3. `fact = 16π/celvol` in `mtxel_kernel.f90:398` is assigned but never used in
   that file (grep: declaration `:123`, assignment `:398`, no other hits) —
   legacy; the real prefactor is applied in `intkernel.f90:224`.
4. `intwfn` uses the **unshifted** interpolation vertices for the shifted grid
   when qflag==1 (comment `intwfn.f90:376-378`: "In principle, we should be
   calling the below function twice… insignificant difference for typical
   small q's") — a documented approximation, relevant when validating LORRAX
   velocity-operator runs against BGW at coarse fine-grids.
5. qflag=0 (arbitrary Q from WFNq_co) still falls back to the old greedy
   interpolation for the shifted grid even under `delaunay_interpolation`
   ("TODO: Fix This!!!", `intwfn.f90:398-402`).
