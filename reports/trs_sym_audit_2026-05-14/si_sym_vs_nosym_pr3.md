# Si sym vs nosym PR3 Σ_X validation — Si 4×4×4 SOC

**Date**: 2026-05-14
**Task**: #30 mirror — exercise PR3 τ-phase code path on non-symmorphic Si Fd-3m
**Source**: `lorrax_B` @ `agent/trs-aware-sym-fix` (PR3 commit `8504994` + `a45f039` + `69ab42c`)
**Run dir**: `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/`
**Companion**: MoS₂ test (PASS, 0.090 meV) — `reports/trs_sym_audit_2026-05-14/sym_vs_nosym_pr3_validation.md`

## Verdict

**FAIL.** max |ΔΣ_X(k, n)| = **160,077 meV** = **160 eV** across 64 k-points × 56 bands = 3584 (k, n) pairs. Pass gate was ≤ 1 meV; observed residual is **160,000× over threshold**. This is a real, catastrophic bug in PR3's τ-phase code path that the MoS₂ test bed could not detect.

## Why Si is the load-bearing test (not MoS₂)

Si diamond structure has space group Fd-3m, which is **non-symmorphic**: 36 of 48 sym operations carry non-trivial fractional translations τ (glide planes + screw axes between the two diamond sublattices). These operations exercise the τ-phase factor `e^{−i(k+G)·τ}` in `unfold_psi` — a code path that PR3 touched.

The MoS₂ test (which PASSED at 0.090 meV) used a sym WFN with only ntran=2 (E + σ_h), both symmorphic (τ=0). The τ-phase branch was therefore never exercised. The MoS₂ pass is a necessary but not sufficient gate.

### τ-phase content of the Si sym WFN

Verified via `mf_header/symmetry/tnp` in `runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5`:

```
ntran = 48
non-symmorphic ops (|τ mod 1| > 1e-6): 36 / 48
example τ's (fractional, mod 1):
  op1: τ = [-0.5,  0,    0   ]
  op2: τ = [ 0,    0,   -0.5 ]
  op3: τ = [ 0,   -0.5,  0   ]
  op4: τ = [ 0.5,  0,    0   ]
  op6: τ = [ 0,    0.5,  0   ]
  op7: τ = [ 0,    0,    0.5 ]
```

Every IBZ k-point sees non-symmorphic operations in its little group:

```
 k0=[0,    0,    0   ]: |G_k|=48, ns_in_Gk=36
 k1=[0,    0,    0.25]: |G_k|=4,  ns_in_Gk=2
 k2=[0,    0,   −0.5 ]: |G_k|=8,  ns_in_Gk=4
 k3=[0,    0.25, 0.25]: |G_k|=2,  ns_in_Gk=0
 k4=[0,    0.25,−0.5 ]: |G_k|=2,  ns_in_Gk=1
 k5=[0,    0.25,−0.25]: |G_k|=4,  ns_in_Gk=2
 k6=[0,   −0.5, −0.5 ]: |G_k|=8,  ns_in_Gk=4
 k7=[0.25,−0.5, −0.25]: |G_k|=4,  ns_in_Gk=2
```

Si also has **inversion** (op 24 = −I). With `no_t_rev=true`, the spatial group fully covers the BZ unfold ⇒ **no TRS-fold rows fire** (PR3's iσ_y·conj rotation is a no-op here). The Si test isolates the τ-phase code path that MoS₂ left untested.

## Test design

Two LORRAX cohsex runs on the SAME Phase-2 code, SAME 384-centroid orbit-closed file (generated on sym WFN), differing only in the symmetry of the input WFN.

| run | WFN source | ntran | sym ops | n_q in V_q | code paths exercised |
|-----|------------|-------|---------|------------|----------------------|
| `run_sym/`   | `05_si_4x4x4_sym/qe/nscf/WFN.h5`    | 48 | Oh+inv, 36 non-symm | n_q_ibz=8 → unfold to 64 | PR1 IBZ tables → PR2 unfold_v_q → PR3 unfold_psi + τ-phase |
| `run_nosym/` | `02_si_4x4x4_nosym/qe/nscf/WFN.h5`  | 1  | E only              | n_q_ibz=64 (=full BZ)    | trivial: IBZ ≡ full BZ, no unfold work |

Both runs converged in ~12 s wall on 4 A100. `gw.out` confirms `n_q_ibz=8 ... unfold=IBZ→full` for sym, `n_q_ibz=64 ... unfold=IBZ→full` for nosym.

**cohsex.in** (identical between runs except WFN link): `x_only=true`, `do_screened=false`, `bispinor=false`, `bare_coulomb_cutoff=25.0`, `nval=8`, `ncond=48`, `nband=56`, `write_wfn_h5=false`.

The MoS₂ template was followed exactly. Code-path A (orbit-closed kmeans, `--orbit` default), 384 centroids on Si sym WFN.

## Verification of test bed

```
WFN              ntran  nk  nspinor   ecutwfc  noncolin  lspinorb  no_t_rev
05 (sym)         48     8   2         25 Ry    true      true      true
02 (nosym)       1      64  2         25 Ry    true      true      true   (nosym=true)
```

DFT eigenvalues match to **10 significant figures** at every k-point.  Both nscf use the same `silicon.save` from `02_si_4x4x4_nosym/qe/scf/`.  V_q traces match to 1% (sym: 2.195e8, nosym: 2.173e8) — both V_q paths produce a consistent bare Coulomb on a 4×4×4 lattice.

```
--- DFT eigenvalue offset (orientation check, should be 0) ---
   max |ΔE_dft|        =     0.0010 meV   (well below SCF tolerance)
```

The kin_ion and V_H columns match exactly between sym and nosym for every (k, n).  Only `x_bare` and the redundant `sex_0` column (a copy of `x_bare` when `x_only=true`) differ.

## Per-k Σ_X table (eV, top of the breakdown)

Σ_X total = `x_bare + sex_0 + coh_0` (the `coh_0` column is identically 0 with `x_only=true`).

```
ik   |G_k| ns max|ΔΣ_X|/meV  mean|ΔΣ_X|/meV  max|Δx_bare|/meV  N_bands
 0    48   36   140,558.87      69,260.86          70,279.44       56
 1    4    2    145,732.76      69,716.62          72,866.38       56
 2    8    4    160,077.73      68,721.31          80,038.87       56
 3    2    0    146,669.08      69,972.59          73,334.54       56
 4    2    1    144,778.27      69,782.28          72,389.14       56
 5    4    2    142,780.18      68,531.02          71,390.09       56
 6    8    4     97,467.63      67,102.08          48,733.82       56
 7    4    2     96,846.00      67,058.50          48,423.00       56
 8    8    4    159,131.53      68,662.92          79,565.77       56
 ...
```

(Full 64-row breakdown in `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/compare_sigma_x.log`.)

**Observations**:
1. Every single one of the 64 BZ k-points fails the gate by 5-6 orders of magnitude.
2. k_3 (sym IBZ k=(0, 0.25, 0.25), little group |G_k|=2, **0 non-symmorphic ops in G_k**) STILL shows 73,334 meV |Δx_bare|. So the bug is not limited to the little group's τ-phase content — it manifests via the unfold path even at k-points whose little group happens to be symmorphic.
3. Σ_X(k, n) on the sym side is systematically **more negative** than on nosym, with a non-constant magnification ratio (2.3× to 50×).

## Top-10 worst rows

```
ik    n   x_bare_sym    x_bare_nosym   Δx_bare/meV    ΔΣ_X(total)/meV
 2   48    -81.638216    -1.599351     -80,038.87      -160,077.73
 2   49    -81.638216    -1.599351     -80,038.87      -160,077.73
 2   46    -81.475836    -1.595093     -79,880.74      -159,761.49
 2   47    -81.475836    -1.595093     -79,880.74      -159,761.49
32   49    -81.460318    -1.590368     -79,869.95      -159,739.90
32   48    -81.460318    -1.590368     -79,869.95      -159,739.90
32   46    -81.293226    -1.586151     -79,707.08      -159,414.15
32   47    -81.293226    -1.586151     -79,707.08      -159,414.15
 8   48    -81.138044    -1.572279     -79,565.77      -159,131.53
 8   49    -81.138044    -1.572279     -79,565.77      -159,131.53
```

The largest residuals (160 eV) sit at the high conduction bands {46-49} at k=2, 8, 32. The nosym side gives small |Σ_X| there (~1.6 eV) — physically reasonable for high-energy conduction states (small valence-band overlap). The sym side gives ~81 eV which is unphysical and ~50× larger.

At occupied bands (n=0,1) the disagreement is smaller (~70 meV/eV ratio 2.3×) but still ~70,000× the gate.

## Sanity check: low-band Σ_X values

For comparison with established physics:

- Diamond Si Σ_X at the VBM Γ point is typically reported around -12 to -15 eV (BGW reference calculations at similar centroid count).
- **nosym Si k=0 n=0: x_bare = -17.0 eV** → in the right ballpark.
- **sym Si k=0 n=0: x_bare = -39.4 eV** → ~2.5× too negative; unphysical.

The nosym path produces physical numbers; the sym path produces broken numbers. The bug is in PR3's sym unfold, not in some independent normalization.

## Diagnosis

- **MoS₂ test (PASS) and Si test (FAIL) share PR3 commit `8504994`**. The difference is the τ-phase content of the input WFN:
  - MoS₂: ntran=2, all symmorphic → τ-phase branch not entered
  - Si: ntran=48, 36 non-symmorphic → τ-phase branch fires at every k
- The failure is **not** in the TRS-rotation half of PR3 (iσ_y·conj on `U_spinor[ntran + s]`): Si has inversion, no_t_rev=true ⇒ no TRS-fold rows in the sym group.
- The failure is in the **spatial unfold τ-phase factor** that PR3's `unfold_psi` applies via `_get_umklapp_vector` (the PR3 commit message explicitly mentions a τ-phase fix there).
- The non-constant magnification ratio (2.3× at low bands, 50× at high bands) suggests a per-G-vector phase error, not a missing prefactor. Different bands sample different G-shells with different |G·τ|, hence different per-band magnitudes of the bug.

## Recommended next steps

1. Add Si 4×4×4 SOC to the trs-aware-sym-fix branch's pytest gate so the τ-phase bug surfaces locally.
2. Audit `unfold_psi` and `_get_umklapp_vector` in `sources/lorrax_B/src/file_io/symmetry_maps.py` for the τ-phase factor: shapes, signs, units (τ is stored in 2π/lattice units in BGW).
3. Hypothesis to test: the τ-phase factor might be applied to (k+G) but should be to ((k+G) folded into the first BZ), or vice versa. A single-q V_q comparison between sym-unfold and direct nosym evaluation would isolate this.
4. Once `unfold_psi` is corrected, rerun this Si test bed (`runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/`) for regression; ≤ 1 meV gate.

## Pipeline timings

| step | NGPU | wallclock |
|------|------|-----------|
| kmeans (sym WFN, orbit-aware, 11 orbits → 384 centroids) | 1 | 9.8 s |
| dipole.h5 / kin_ion.h5 (sym, nosym) | 1 each | ~2 s each |
| run_sym/cohsex   | 4 | 11.9 s |
| run_nosym/cohsex | 4 | 12.9 s |
| compare (Python parsing) | 0 | < 1 s |

Total GPU-min: ~3.

## Notes (separate issues encountered)

- **`write_wfn_h5` crash on full-BZ paths**: `file_io/qp_wfn.py:137` raises `ValueError: U shape (64, 56, 56) inconsistent with (nk=8, nb_active=56)` when the WFN writer is fed a U built on the unfolded full BZ but the writer expects IBZ shape. Worked around by setting `write_wfn_h5=false`. This is **a separate bug** from the τ-phase Σ_X failure reported here; both happen to fire on the same sym WFN.
- SLURM `Requested nodes are busy` errors required 2 retries (`LORRAX_IMMEDIATE=30` worked). Transient.

## Files

- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_sym/sigma_freq_debug.dat`
- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_nosym/sigma_freq_debug.dat`
- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/compare_sigma_x.py`
- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/compare_sigma_x.log` — full 64-row table
- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/manifest.yaml`
- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_sym/gw.out`
- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_nosym/gw.out`

---

## Addendum 2026-05-14 16:05 — Phase-2 bisect: bug is PRE-EXISTING, not Phase-2-introduced

**TL;DR.** The 160 eV failure is reproduced **bit-for-bit** on the
pre-Phase-2 commit `9e644e9` (the V_q TRS fix; pre-PR1/PR2/PR3 — does not
touch `wfn_loader.py` or `symmetry_maps.py::SymMaps.__init__`). Phase 2
neither introduced nor changed Si's broken Σ_X output. The bug existed in
the LORRAX ψ-unfold / Σ_X pipeline before any Phase 2 work landed.

### Verification

Ran the existing test bed (`runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/`)
under `lorrax_B` checked out at commit `9e644e9` (April 2026, pre-Phase-2).
Reused the same `run_nosym/` reference (unchanged HEAD output) and the
same 384-centroid orbit-closed centroid set. Output saved at
`bisect/01_pre_phase2_v3/`.

```
                    sigma_freq_debug.dat       max|ΔΣ_X|/meV   max|Δx_bare|/meV
commit               vs run_nosym               vs run_nosym    vs run_nosym
69ab42c (HEAD/PR3)   run_sym/                   160077.7300     80038.8650
9e644e9 (pre-PR1)    bisect/01_pre_phase2_v3/   160077.7300     80038.8650
```

```bash
$ diff bisect/01_pre_phase2_v3/sigma_freq_debug.dat run_sym/sigma_freq_debug.dat
$ # no output — files are byte-identical (732,301 bytes each).
```

The two `sigma_freq_debug.dat` files **agree to the last printed digit**
for all 3584 (k, n) pairs. Pre-Phase-2 LORRAX produces the exact same
broken Si-sym Σ_X as PR3 HEAD does.

### What this means

- **PR3 is NOT the offender.** The diagnosis in the body above
  ("Diagnosis", lines 138-145) — that the τ-phase code path in
  `unfold_psi` is broken on non-symmorphic spatial ops — is **incorrect**.
  PR3's diff vs pre-Phase-2 for the spatial-only branch (which is the
  ONLY branch Si exercises, since no_t_rev=true + inversion ⇒ no TRS rows
  in any little group) is bit-equivalent by construction:
  - eager path: `sym_krep @ g_bar.T` (pre-PR3) vs
    `S_full @ g_bar.T` with `S_full = sym_mats_k[sym_idx]` (PR3) — same
    operand, same formula, same `exp(-1j (rotated @ tau))`.
  - phdf5 path: for `s_spatial = s` (when `s < ntran`), same
    `sym_mats_k[s] @ g_bar.T` rotation, same `exp(-1j rotated · τ)`,
    same `U_per[k] = sym.U_spinor[s]` value (PR3's `U_spinor_spatial[s]`
    is the same length-ntran slice of pre-PR3's length-2·ntran array).
- **MoS₂ PR3 PASS at 0.090 meV remains valid.** That test exercises
  a different code path (TRS-fold rows on a ntran=2 symmorphic group).
  PR3 fixed the TRS-fold ψ-unfold bug it was designed to fix. The Si
  failure is orthogonal — a separate, much older bug — that PR3 did not
  introduce and that the MoS₂ test cannot detect.
- **The Si 4×4×4 sym path has apparently never been validated against
  nosym before today.** `runs/Si/05_si_4x4x4_sym/manifest.yaml` is
  `pipeline: qe_only` — there is no `00_lorrax/` directory under that
  run; LORRAX was never previously fed the Si sym WFN. The 160 eV
  disagreement was therefore latent — sitting in production code for
  weeks/months.

### Bisect summary

| step | commit | dir | max |ΔΣ_X| (meV) | verdict |
|------|--------|-----|---|---|
| 1 | `9e644e9` (pre-Phase-2; Phase 1 V_q fix only) | `bisect/01_pre_phase2_v3/` | 160 077.73 | FAIL |
| 2 | `69ab42c` (HEAD / PR3+) | `run_sym/` | 160 077.73 | FAIL |
| Δ | byte diff of `sigma_freq_debug.dat` | — | 0 (files identical) | — |

Per the protocol's Step 1 branch:
> If pre-Phase-2 Si sym ALSO disagrees with nosym at the 160 eV level:
> the bug is pre-existing and didn't originate from Phase 2. Document
> that finding and stop — we need to investigate the pre-existing path
> separately, NOT revert Phase 2.

Stopping here per instructions. No further Phase-2-commit bisect steps
run; no source revert proposed.

### What the failure DOES tell us (qualitative diagnosis pointers)

Although the offending commit is not in the Phase 2 chain, the failure
structure (which the body of this report characterized correctly) still
narrows down the search:

1. **All 64 BZ k-points fail by 5-6 orders of magnitude**, including
   k_3 with little group |G_k|=2 and **zero non-symmorphic ops in G_k**
   (73 334 meV |Δx_bare|). Whatever's wrong fires on the *unfolded*
   ψ side, not on a property of k's own little group.
2. **`max |Δsex_0| = max |Δx_bare|` exactly** (both 80 038.87 meV) —
   confirms `sex_0` is a copy of `x_bare` under `x_only=true`, and the
   bug lives in `x_bare` (the bare Σ_X exchange term itself).
3. **Sign and scale**: sym Σ_X is systematically ~2.5× more negative
   than nosym at occupied bands (Γ VBM: -39.4 eV sym vs -17.0 eV
   nosym), and 50× more negative at high conduction bands
   (k=2, n=48: -81.6 eV sym vs -1.6 eV nosym). The magnification ratio
   tracks the |G·τ| content sampled by each band — consistent with a
   τ-phase factor applied with the wrong sign / wrong G-list / in the
   wrong frame.
4. **kin_ion and V_H agree exactly** (column-by-column) — proves the
   bug is NOT in the DFT-mtxel evaluation path or in the matrix
   element evaluation per-se; it fires only when full-BZ-unfolded ψ
   are recombined in a non-translation-invariant integrand like Σ_X.

### Where the bug is likely to live (NOT verified — for the next agent)

The bug is **somewhere in the pre-PR1 sym ψ-unfold path or in a
downstream consumer that uses the unfolded ψ in the Σ_X kernel**. Likely
suspects (NOT diagnosed in this report; verification deferred):

- **`compute_vcoul` / `v_q_tile` (pre-Phase-2)**: the bare-Coulomb /
  Σ_X kernel may not apply the τ-phase consistently on both bra and ket
  sides of a `<m k | ψ_{m',k-q}><ψ_{m',k-q} | n k>` product. If the same
  ψ_{m',k-q} appears twice and the τ-phase is included once but not the
  other side, the phases don't cancel, and Σ_X diverges.
- **G-list / umklapp bookkeeping for non-symmorphic operations**: the
  `unfold_psi` docstring says it returns ψ "on the IBZ G-axis (i.e.
  `cnk_full[b, σ, g]` corresponds to the G-vector
  `sym_mats_k[sym_idx] @ g_kbar[g]` in the full-k basis)." Whether the
  downstream Σ_X kernel correctly *re-maps* these G-vectors to the
  full-k G-list (via the umklapp kg0 vector) is the most plausible site
  of the bug. A G-vector mismatch on the unfolded side would produce
  exactly this kind of "wrong by a phase / scaled-by-overlap" failure
  pattern, with magnitudes that grow with the number of high-|G|
  components participating in Σ_X (matches the band-dependent
  magnification factor 2.5×→50×).
- **`isdf_fitting` / `psi_G_store` on sym WFN**: the ζ-fitting step
  consumes the unfolded ψ. If a τ-phase / U-rotation step is dropped
  there, the resulting ζ would not represent the same orbital products
  as the nosym ζ. The G-flat ζ in `tmp/zeta_q.h5` could be checked
  bit-for-bit against a "nosym-evaluated, then sym-projected" reference
  to confirm or rule this out.
- **`bare_coulomb_cutoff = 25.0`** is set explicitly per the BGW
  convention, so this is not a cutoff mismatch.

### Recommended next steps

1. **Open a separate bug ticket** "Si 4×4×4 sym Σ_X is 160 eV off from
   nosym — pre-existing, not Phase 2." Triage as a P0 correctness bug
   for any non-symmorphic system. CrI3 6×6 / Si / any diamond-structure
   system is potentially affected — every prior Si sym LORRAX result
   should be re-verified.
2. **Add a Si 4×4×4 SOC sym-vs-nosym Σ_X gate to the regression test
   suite** so the next ψ-unfold change can't silently re-break this.
3. **Phase 2 (PR1+PR2+PR3) can merge as planned** — the Si failure is
   not a blocker for Phase 2 (no Phase 2 commit caused it; Phase 2 did
   not regress Si Σ_X by even a single ULP).
4. **Bisect pre-Phase-2 history** to find when this bug landed. Try
   running this same test bed against, e.g., the `main` branch at
   2026-04 (before sym work began), and walk backward via `git bisect`
   on `master`/`main`. A canonical sandbox commit message dated to the
   sym ψ-unfold's first appearance would identify the offending PR.
5. **Compare against BerkeleyGW** on the same Si 4×4×4 sym WFN. If BGW
   Σ_X agrees with LORRAX nosym (which we expect), the disagreement is
   entirely a LORRAX-sym-path bug. If BGW disagrees with both, then
   *Si nosym* is the broken side (much less likely given the physical
   plausibility of the nosym number, but worth confirming).

### Artifacts produced by this addendum

- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/bisect/run_one.sh` — bisect runner
- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/bisect/01_pre_phase2_v3/` — pre-Phase-2 run output (gw.out, sigma_freq_debug.dat, compare.log, summary.txt) on commit `9e644e9`
- `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/bisect/SUMMARY.txt` — one-line summary across attempts

Working tree reset to HEAD `69ab42c` per hard-constraint.
