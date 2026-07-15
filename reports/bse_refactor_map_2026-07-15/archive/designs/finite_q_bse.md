# Finite-momentum H^BSE_Q for arbitrary Q — design

Program: BSE refactor-map, 2026-07-15. Checkout `sources/lorrax_D` (BSE tree
byte-identical between e18d0e5 and working HEAD adc2197 per
`kernel_dataflow_trace.md:3-7`). Scope: solve the TDA exciton problem at nonzero
center-of-mass momentum **Q** in the `|v k, c k+Q>` pair basis.

Convention (LORRAX / `parallel_bse_algos.md:9`): `X^Q(v,c,k)` = electron at
`(c, k+Q)`, hole at `(v, k)`; exciton COM momentum `+Q`. BGW uses the mirror
convention (shift on valence, COM `−Q`, `bgw_fine_grid_reference.md:466-475`) —
a relabeling, not a different physics.

---

## Current state in LORRAX (file:line grounded)

**Q=0 only, hardcoded.** `nQ=1` is baked at the eigenvector writer
(`bse_io.py:45`), `exciton_Q_shifts` written as zeros
(`context/README.md:210`, `write_eigenvectors.py:66-67`), and no code path
carries a Q. `context_docs.md:216-218` confirms the finite-Q algebra in
`parallel_bse_algos.md §1` "remains unimplemented design."

**The matvec looks up every wavefunction at k, not k+Q.** Reference kernel
`bse_serial.apply_bse_hamiltonian_single_device` (`bse_serial.py:38-80`):
- D term `(ε_c[k,c] − ε_v[k,v])·X` — both energies at k (`bse_serial.py:56-57`,
  `bse_simple.py:82-83`).
- V (exchange) uses the single q=0 tile `V_q0` and pair amplitude
  `M[k,c,v,μ]=Σ_s conj(ψ_c[k,c,s,μ])ψ_v[k,v,s,μ]`, all at k
  (`bse_serial.py:59-64`, `bse_simple.py:89-131`).
- W (direct) FFT-convolves `T[b,μ,ν,t,s,k]` built from `ψ_c[k]`, `ψ_v[k]`
  over q=k−k′ (`bse_serial.py:69-78`, `bse_simple.py:141-173`).

**Loader picks the q=0 tile and injects the divergent head.** The sharded
loader (`bse_io.py:358-536`) slices `val_indices = [n_occ−n_val, n_occ)`,
`cond_indices = [n_occ, n_occ+n_cond)` (`:433-434`), reads only the q=0
exchange tile via `_read_vq0_sharded` (`:460`), the full W_q(μ,ν,kx,ky,kz)
(`:461`), then rank-1-injects the q→0 Coulomb head
`V_q0 += (vhead/V_cell)·conj(g0)⊗g0` using `g0=ζ(q=0,μ,G=0)`
(`:468-513`, `gw/head_correction.py:779`). The full V_qmunu(nq,μ,ν) flat-q
tensor with **all** q tiles is on disk (`kernel_dataflow_trace.md:167`,
`bse_io.py:389-411`; `nq = nkx·nky·nkz` enforced, `:318-320`) — only the q=0
slice is currently read.

**k-order convention.** `generate_kpts_grid` (`write_eigenvectors.py:156-173`)
fixes the C-order `k = ix·nky·nkz + iy·nkz + iz`, `k_frac=(ix/nkx,…)`; the FFT
convolution and `V_qmunu`/`W0_qmunu` flat-q order all coincide with it
(`kernel_dataflow_trace.md:357-362`). eqp IBZ→full-BZ unfolds through the one
`common.symmetry_maps.SymMaps` table (`bse_io.py:713`).

**Spectra are Q=0 optical only.** Both routes seed the dipole
`d[α,k,c,v]=⟨c|v̂_α|v⟩/ΔE` from `dipole.h5` (`absorption_common.py:86-102`,
`absorption_haydock.py`, `absorption_eigvecs.py`); there is no structure-factor
path.

---

## Reference physics + BGW implementation

Rohlfing–Louie kernel at finite Q (`bgw_fine_grid_reference.md:33-36,477-478`):
```
K^x(vck,v'c'k'|Q) =  Σ_G  M*_vc(k,Q,G) v(Q+G)  M_v'c'(k',Q,G)
K^d(vck,v'c'k'|Q) = −Σ_GG' M*_cc'(k,Q,G) W_GG'(q) M_v'v(k+Q,q,G'),   q = k'−k
M_nn'(k,Q,G)      = <n k+Q| e^{i(Q+G)·r} |n' k>
```

**Direct term W: structurally unchanged, Q cancels in the phase.** In the ISDF
centroid form (`parallel_bse_algos.md:14-17`) the convolution kernel
`W̃_{q,μν}` is **Q-independent**; only the conduction wavefunctions enter
Q-shifted:
```
W^Q_A(ivic k, jvjc k') = (1/Nk) Σ_μν  ū_{ic,k+Q}(μ) u_{jc,k'+Q}(μ)  W̃_{k−k',μν}  ū_{jv,k'}(ν) u_{iv,k}(ν)
```
Valence stays at k, conduction at k+Q (`parallel_bse_algos.md:38-43`).

**Exchange term V: acquires a Q-tile, INCLUDING G=0, no divergence-exclusion.**
```
V^Q_A(ivic k, jvjc k') = (1/Nk) Σ_μν  ū_{ic,k+Q}(μ) u_{iv,k}(μ)  Ṽ^Q_μν  ū_{jv,k'}(ν) u_{jc,k'+Q}(ν)
Ṽ^Q_μν = |Ω| Σ_G  4π/|Q+G|²  conj(ζ^V_μ(G)) ζ^V_ν(G)          (parallel_bse_algos.md:22)
```
At Q≠0 the G=0 term `4π/|Q|²` is **finite** — no head rank-1 injection, no
mini-BZ average. At Q=0 this collapses to `Ṽ_μν` with G=0 excluded (Henneke
Eq 2-32, `context_docs.md:324-326`), which is exactly the current `V_q0` +
separate head machinery. BGW builds `v(−Q+G)` as one extra Coulomb column
(`kernel_main.f90:276-283`) and zeroes the head — the G closest to −Q,
`qpg0_ind = argmin_G |G+Q|²` (`kernel_main.f90:285-297`) — **unless
`energy_loss` is set**: "we should NEVER use finite Q without the energy_loss
flag" (`kernel_main.f90:287-289`, `mtxel_kernel.f90:689`,
`bgw_fine_grid_reference.md:481-490`). i.e. BGW's *default* finite-Q kernel
keeps the full `v(Q+G)`; the head-zeroing is the optical-limit special case.

**Diagonal.** `D = ε_c(k+Q) − ε_v(k)` (`parallel_bse_algos.md:11-12`); BGW
takes the valence energy at the shifted index `evqp(iv, indexq_fi(ik))`
(`diag.f90:489-495`).

**On-grid vs arbitrary Q.** BGW `exciton_Q_shift Qflag Qx Qy Qz`
(`inread.f90:402-404`): `qflag=2` = Q commensurate with the coarse grid,
`ψ_{c,k+Q}` taken from `WFN_co` itself (no shifted NSCF);
`qflag=0` = arbitrary Q from a separate `WFNq_co`, "deprecated/under
development" (`bgw_fine_grid_reference.md:466-475`,
`docs_bgw/absorption.inp:494-515`).

---

## Proposed design

**Core result: on-grid finite-Q is a k-axis remap of the conduction tensors
plus a Q-tile swap. The matvec, solvers, sharding, FFT convolution, and ISDF
ζ/centroid machinery are BYTE-IDENTICAL to Q=0.** Because `psi_full_y` stores
the cell-periodic part `u_{n,k}(r_μ)` at **every** k and at k-independent
centroids `r_μ` (`kernel_dataflow_trace.md:169,346-348`), `u_{n,k+Q}(r_μ)` is
already on disk at the wrapped grid index — no new ζ fit, no new NSCF.

### Dataflow

```
                         Q (integer grid steps, on the k-grid)
                                     │
 isdf_tensors_*.h5 ─► load_bse_data_from_restart_sharded(..., q_shift=Q)
   psi_full_y ─┐        kpQ_index[k], G_umk[k]  = kgrid_shift_map(kgrid, Q)   ← NEW helper
   enk_full  ─┤        (C-order fold; SymMaps k-table)
   V_qmunu   ─┤
   G0_mu_nu ──┘  conduction remap (host cache, io_callback per μ-shard):
                    psi_c_Q[k,c,s,μ] = e^{-i2π G_umk[k]·s_μ} · psi_full[kpQ_index[k], c, s, μ]
                    eps_c_Q[k,c]     = enk_full[kpQ_index[k], cond_indices]
                 exchange tile:
                    V_Q  = V_qmunu[Q_flat]   (NO head injection when Q≠0)
                 valence + W_q + head(q=0 only if Q==0):  unchanged
                                     │
                 ┌───────────────────┴────────────────────┐
                 ▼  (identical data dict; conduction slots hold the Q-shifted caches)
   build_bse_simple_matvec / _ring / _serial  ── UNCHANGED ──►  H^Q X
                                     │
   FEAST / Lanczos / Davidson  ── UNCHANGED ──►  E_S(Q), A^S
                                     │
                 ┌───────────────────┴────────────────────┐
                 ▼ spectra: NO dipole coupling
   structure_factor S(Q,ω): seed ρ_cvk(Q)=Σ_μ conj(psi_c_Q)ψ_v·g0[μ]  (reuse g0!)
                            → run_haydock / sum-over-states  (1 polarization)
```

### Explicit formulas for the physics-critical pieces

**k+Q fold and umklapp (the one genuinely new physics).** With
`Q=(Qx,Qy,Qz)` integer steps and grid point `k=(ix,iy,iz)`:
```
jx = ix+Qx ;  kpx = jx mod nkx ;  Gx_umk = jx // nkx           (0 or 1)
kpQ_index[k] = C_order(kpx,kpy,kpz)
G_umk[k]     = (Gx_umk, Gy_umk, Gz_umk)   (integer reciprocal-lattice vector)
```
When k+Q wraps the zone (`G_umk≠0`, ≈half the k-points), the cell-periodic part
gains the umklapp phase `u_{n,k+Q}(r) = e^{-iG_umk·r} u_{n,(k+Q)_BZ}(r)`. At
centroid fractional coords `s_μ` (from `centroids_frac_*.txt`):
```
phase[k,μ] = exp(-2πi (G_umk[k]·s_μ))          # exact at ISDF nodes
psi_c_Q[k,c,s,μ] = phase[k,μ] · psi_full[kpQ_index[k], c, s, μ]
```
Energies carry no phase: `eps_c_Q[k,c] = enk_full[kpQ_index[k], cond_indices]`.
This phase is why the shift is not *literally* a permutation; it is a
diagonal-in-(k,μ) unitary applied once at load.

**Exchange at Q≠0.** Read `V_Q = V_qmunu[Q_flat]` (`Q_flat = C_order(Q)`);
skip the rank-1 head. `V_qmunu` already carries `4π/|Q+G|²` for all G including
G=0 (built GW-side in `compute_V_q`), so `V_Q` is exactly `Ṽ^Q_μν`
(energy_loss convention). The matvec's V contraction is unchanged — it just
receives `V_Q` in the `V_q0` slot and `psi_c_Q` in the conduction slots.

**Structure factor spectra (no dipole).** The oscillator amplitude is the G=0
component of the Q pair density, which is precisely the g0 projection already
loaded for the head:
```
ρ_cvk(Q) = Σ_μ conj(psi_c_Q[k,c,s,μ]) ψ_v[k,v,s,μ] g0[μ]        (spin-traced)
S(Q,ω) ∝ Σ_S | Σ_{cvk} A^S_{cvk} ρ_cvk(Q) |²  · L(ω−E_S; η)
```
`g0 = ζ(q=0,μ,G=0)` is `G0_mu_nu` (`bse_io.py:468-469`) — no plane-wave dipole,
no new I/O. Haydock seeds `|ρ(Q)⟩` (one polarization) in place of `|d^α⟩`.

### File-level plan (single-sourced; no parallel paths)

| Change | File | What |
|---|---|---|
| NEW `kgrid_shift_map(kgrid, Q) → (kpQ_index, G_umk)` | `common/symmetry_maps.py` (accessor on the k-grid, plain numpy) | C-order fold + umklapp; the ONE place k+Q arithmetic lives. Reuses the existing C-order convention; no new class. |
| EXTEND loader with `q_shift: tuple[int,int,int]=(0,0,0)` | `bse_io.py:358` (`load_bse_data_from_restart_sharded`) + `:767` (`_load_ring_subset`) | build `phase[k,μ]`, remap `psi_c_*`/`eps_c`, pick `V_Q=V_qmunu[Q_flat]`, gate head on `Q==(0,0,0)`. Conduction ψ remap done inside the existing per-μ-shard `_read_psi_mu_sharded` host reads (`bse_io.py:174-218`) so the shifted cache is built on host and never a jit arg. |
| Parametrize the exchange tile index in `_read_vq0_sharded` | `bse_io.py:221-265` | add `q_index=0` arg; default preserves Q=0. Single reader, not a fork. |
| NEW spectra route `structure_factor.py` | `src/bse/` | `ρ_cvk(Q)` seed from `psi_c_Q`,`psi_v`,`g0`; delegate to existing `run_haydock` / `absorption_eigvecs` sum-over-states with the ρ(Q) seed. Reuses `absorption_common` writers. |
| CLI `--exciton-Q Qx Qy Qz` (integer grid steps) | `bse_jax.py:__main__` (`:349-626`) | thread to loader; `nQ`/`exciton_Q_shifts` in the eigenvector writer (`bse_io.py:45`) filled from Q instead of hardcoded 0. |
| DELETE nothing new; reuse | — | head machinery (`gw/head_correction.py`), FFT helpers, `make_bse_shardings`, all solvers untouched. |

### Sharding + memory plan

No new sharding. `psi_c_Q` inherits `psi_c_X = P(None,None,None,'x')` /
`_Y = P(…,'y')` (`bse_ring_comm.py:46-63`); `V_Q` inherits `V_q0 = P('x','y')`;
the trial vector `X = P(None,'x','y',None)` and every intermediate are
identical to Q=0. The k-remap and umklapp phase are applied **on host, per
μ-shard** inside the existing `_read_psi_mu_sharded` reads
(`bse_io.py:174-218`) — the shifted ψ cache is a read-only host cache pulled
per slice, honoring the io_callback-not-jit-arg rule. Extra memory: the
`(nk, n_rmu_pad)` phase table (tiny) and `Q_flat` scalar. Peak-memory profile
is unchanged from Q=0, so the existing 1-GPU MoS2/Si fixtures fit.

---

## Interactions with the other four designs (shared seams)

1. **Exchange-correctness / single-sourced kernel (B1).** The current exchange
   is k-block-diagonal — `S=einsum("kcvN,bcvk->bNk")` keeps k as a batch index,
   dropping the k′≠k blocks (`kernel_dataflow_trace.md:368-388`,
   `bse_serial.py:62`). The physical exchange (Q=0 **and** finite-Q) is dense in
   (k,k′): `V^Q_A ∝ [pair density at k]·Ṽ^Q·[pair density at k′]`
   (`parallel_bse_algos.md:20`). Finite-Q inherits whatever the kernel design
   lands: once the exchange sums over k, finite-Q needs only `V_q0→V_Q` +
   `psi_c→psi_c_Q`. **Seam: the exchange contraction must stay single-sourced;
   finite-Q adds only the tile index and the shifted conduction cache.** Do not
   fork a `apply_V_Q`.

2. **Restart-loader / format fixes (B3–B6).** The finite-Q loader edits are the
   same functions the loader-fix design touches (`_read_vq0_sharded` 8-D-only
   B3, `_read_wq_sharded` kgrid-attr B4, rank-3 head-inject B5, pad-count B6,
   `kernel_dataflow_trace.md:397-433`). **Seam: land the q_shift arg on the
   *fixed* flat-q loader, not the broken HEAD one.** Both designs modify
   `bse_io.py:358` and `:767`.

3. **Fine-grid interpolation (arbitrary Q).** On-grid Q (this design) is the
   qflag=2 analogue. **Arbitrary off-grid Q requires the interpolation layer**
   (dcc/dvv coarse→fine, `bgw_fine_grid_reference.md §1,§4`) to supply
   `u_{c,k+Q}` between grid points, plus a shifted-grid ψ source. **Seam: this
   design owns the on-grid k+Q remap + umklapp helper
   (`kgrid_shift_map`); the interpolation design consumes it and supplies the
   between-grid ψ.** State the hard dependency: arbitrary Q is out of scope
   here.

4. **Spectra / non-TDA.** The structure-factor route reuses `run_haydock` and
   `absorption_eigvecs` (spectra design's surface); it swaps the seed only.
   Non-TDA finite-Q (B-block with Q) is blocked by the malformed B-block einsum
   B2 (`kernel_dataflow_trace.md:389-396`) — **finite-Q is TDA-only until B2 is
   fixed**, matching the production TDA-only reality
   (`kernel_dataflow_trace.md:297-305`).

Shared table/helper: the ONE `SymMaps` k-table (`bse_io.py:713`) — the
`kgrid_shift_map` accessor lives beside it, never a parallel "shift ψ at Q"
helper (unified-sym-action rule).

---

## Gates (1-GPU validation plan, BGW anchors)

All on 1 GPU with Si-scale fixtures (no 16-GPU gating).

1. **Q=0 regression.** `--exciton-Q 0 0 0` must reproduce the existing Si
   4×4×4 8v×8c eigenvalues bit-for-bit against the current path
   (`STATUS.md` ~3 meV BGW match) — proves the remap is identity at Q=0.
2. **kgrid_shift_map unit test.** For a 4×4×4 grid: `kpQ_index` is a
   permutation, `k + Q − kgrid·G_umk = kpQ` exactly, `Q=0 → identity`,
   `G_umk∈{0,1}³`. Pure numpy, pytest-collected (`tests/`).
3. **Umklapp phase check.** For a wrapped k, verify
   `|ρ_cvk(Q)|` computed from `psi_c_Q` equals the direct centroid overlap
   `Σ_μ conj(u_{c,k+Q}(r_μ))u_{v,k}(r_μ)g0[μ]` from an independently phased
   reference (toy 2-band).
4. **BGW finite-Q dispersion on Si.** Run BGW `absorption.x` with
   `exciton_Q_shift 2 Qx Qy Qz` + `energy_loss` for a few on-grid Q along
   Γ→X; compare lowest exciton energies `E_S(Q)` (Ry→eV, gauge-invariant) to
   LORRAX. Anchor: the qflag=2 valence-shift diagonal `diag.f90:489-495` and
   full-`v(Q+G)` exchange (`kernel_main.f90:287-289`). Tolerance = the ISDF
   compression floor already established at Q=0 (~few meV).
5. **Structure-factor sanity.** `S(Q,ω)` first peak disperses upward with |Q|
   for Si (electron-hole exchange + kinetic), and `S(Q→smallest grid Q,ω)`
   trends toward the Q=0 `ε_2` peak position (not magnitude).

---

## Open questions for Jack (physics/priority decisions only he can make)

1. **Exchange head convention at finite Q.** Default to BGW `energy_loss`
   (keep full `v(Q+G)`, i.e. `V_Q = V_qmunu[Q_flat]` untouched — the natural
   EELS / structure-factor choice)? Or also build an optical-limit variant that
   zeroes the smallest-|Q+G| contribution — which the collapsed (μ,ν) tile
   cannot express and would require a GW-side head subtraction in `compute_V_q`
   for the Q tile? Recommendation: energy_loss default; optical-limit deferred.
2. **Priority of on-grid Q vs the exchange (B1) fix.** Finite-Q exchange is
   only as correct as the Q=0 exchange. Should finite-Q wait on the
   dense-in-(k,k′) exchange fix, or land the (W-correct, exchange-inherits)
   version first so the direct-term dispersion can be validated against BGW
   independently?
3. **Which Q path first** — commensurate on-grid (qflag=2, this design, no new
   DFT) vs arbitrary Q (needs the interpolation layer + shifted NSCF)? On-grid
   covers phonon-scale and zone-boundary excitons cheaply; arbitrary Q is a
   much larger lift.
4. **Structure factor normalization / target.** Bare `S(Q,ω) ∝ |Σ A ρ(Q)|²`
   (BSE two-particle DOS weighted by |ρ(Q)|²), or the full loss function
   `Im[−1/ε(Q,ω)]` (needs the RPA denominator / local-field resummation on top
   of the BSE eigenvectors)? The former reuses Haydock directly; the latter is
   a separate post-processing layer.

---

## LOC estimate + suggested phasing

Net ~450–600 LOC, dominated by the spectra route and tests; the kernel change
is near-zero because the matvec is untouched.

- **Phase 1 — on-grid remap + W/D dispersion (~200 LOC).** `kgrid_shift_map`
  (~60), loader `q_shift` plumbing + umklapp phase + `V_Q` tile (~90), CLI flag
  + eigenvector-writer Q fields (~30), Q=0 regression + shift-map unit tests
  (~20). Deliverable: `E_S(Q)` for the direct+diagonal kernel, gate 1/2/4.
- **Phase 2 — structure-factor spectra (~200 LOC).** `structure_factor.py`
  ρ(Q) seed + Haydock/sum-over-states delegation (~120), umklapp phase gate,
  BGW `S(Q,ω)` comparison harness in `runs/Si/...` (~80). Gate 3/5.
- **Phase 3 (dependent, not owned here).** Arbitrary off-grid Q — blocks on the
  interpolation-layer design; consumes `kgrid_shift_map` and a shifted-grid ψ
  source. Non-TDA finite-Q blocks on the B2 fix.

Hard prerequisite: land on the loader-fix design's flat-q loader (B3–B6), not
HEAD — otherwise every finite-Q path is dead on current restart files for the
same reason Q=0 is (`kernel_dataflow_trace.md:418-420`).
