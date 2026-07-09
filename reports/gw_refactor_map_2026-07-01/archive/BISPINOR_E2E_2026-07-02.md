# Bispinor GW end-to-end verification after the r-space V_q "tile" deletion

**Date:** 2026-07-02
**Source tree:** `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`
**Branch:** `agent/memplanner-cleanup` (HEAD `d4bb3ba` "refactor: remove refs to the deleted r-space V_q subsystem", on top of `8369ecc` "delete the dead r-space V_q tile subsystem (~3k lines)")

## VERDICT: PASS — bispinor GW still works end-to-end after the deletion.

Exit 0, no crash in the V_q dispatch or the bispinor orchestrator, the live g-flat
7-tile path ran, Σ^B is finite/real/eV-scale, and **every output value is bit-identical
to the pre-deletion reference** (eqp0, eqp1, all Σ diag physics, all 9 Σ^B tiles).

---

## What was run

| Item | Value |
|------|-------|
| Example (reference) | `runs/MoS2/C_60Ry_bispinor_cohsex_2026-06-15/` (mos2 3×3, screened-charge COHSEX + bare Breit Σ^B) |
| Fresh working copy | `runs/MoS2/_e2e_bispinor_deletion_check_2026-07-02/` (cohsex.in copied; WFN.h5 / kin_ion.h5 / both centroid files symlinked — run dir NOT mutated) |
| Confirmed bispinor | `bispinor = true`, spinor WFN, **4 ζ channels**: charge μ_L=0 on 640 centroids + current μ_L=1,2,3 on 668 centroids (`centroids_file_current`) |
| Hardware | **4 GPUs** (`LORRAX_NGPU=4`), 1 node, A100. cusolverMp grid 2×2. (Task constraint honored: NOT 16 GPUs.) |
| Command | `LORRAX_NGPU=4 lxrun python3 -u -m gw.gw_jax -i cohsex.in` under `module load lorrax_D lorrax_agent`, `SLURM_JOBID=55417809` |
| Exit status | **0** (`=== EXIT 0 END Thu Jul 2 09:59:43 PM PDT 2026 ===`) |
| Wall time | ~2 min incl. compile (reference core was 64 s) |

## Evidence 1 — the g-flat 7-tile bispinor V_q path ran (not the deleted r-space path)

From `runs/MoS2/_e2e_bispinor_deletion_check_2026-07-02/gw_run.log`:

```
[bispinor g-flat] tile 1/7 (μ_L=0, ν_L=0)  n_rmu_L=640 n_rmu_R=640  same_zeta=True  use_ibz=False   # CC
[bispinor g-flat] tile 2/7 (μ_L=1, ν_L=1)  n_rmu_L=668 n_rmu_R=668  same_zeta=True  use_ibz=True    # TT diag
[bispinor g-flat] tile 3/7 (μ_L=2, ν_L=2)  n_rmu_L=668 n_rmu_R=668  same_zeta=True  use_ibz=True    # TT diag
[bispinor g-flat] tile 4/7 (μ_L=3, ν_L=3)  n_rmu_L=668 n_rmu_R=668  same_zeta=True  use_ibz=True    # TT diag
[bispinor g-flat] tile 5/7 (μ_L=1, ν_L=2)  n_rmu_L=668 n_rmu_R=668  same_zeta=False use_ibz=True    # TT off-diag
[bispinor g-flat] tile 6/7 (μ_L=1, ν_L=3)  n_rmu_L=668 n_rmu_R=668  same_zeta=False use_ibz=True    # TT off-diag
[bispinor g-flat] tile 7/7 (μ_L=2, ν_L=3)  n_rmu_L=668 n_rmu_R=668  same_zeta=False use_ibz=True    # TT off-diag
```

All V_q kernels are `V_q g-flat [V_qmunu_CC / _TT_11 / _TT_22 / _TT_33 / _TT_12 / _TT_13 / _TT_23]`
(the g-flat orchestrator `compute_V_q_bispinor_g_flat_to_h5`). **No** reference to
`compute_V_q_bispinor_to_h5`, no r-space error, no "missing builder factory" — the deleted
subsystem is gone and nothing tried to call it.

Note: on lorrax_D the 6 TT tiles now run `use_ibz=True` (IBZ→full unfold, n_q_ibz=5),
whereas the reference ran them full-BZ (`use_ibz=False`, n_q=9). This is the intended
behavior of the accompanying commits (`0d7ba06`, `1479162` add/honor the IBZ path + its
equivalence gate). Despite the different unfold route, Σ^B is identical (see below) — the
physical Σ^B is covariant, consistent with the "measure Σ^B not V_q tiles" finding.

## Evidence 2 — Σ^B (Breit exchange) is sane AND matches reference exactly

`tr Σ^B` per transverse tile (eV), lorrax_D vs reference (`C_60Ry_bispinor_cohsex_2026-06-15/gw.out`):

| tile (μ_L,ν_L) | lorrax_D | reference | Δ |
|---|---|---|---|
| (1,1) | -0.152598 | -0.152598 | 0 |
| (1,2) | -0.012224 | -0.012224 | 0 |
| (1,3) | -0.012362 | -0.012362 | 0 |
| (2,1) | -0.012224 | -0.012224 | 0 |
| (2,2) | -0.152608 | -0.152608 | 0 |
| (2,3) | -0.012362 | -0.012362 | 0 |
| (3,1) | -0.012362 | -0.012362 | 0 |
| (3,2) | -0.012362 | -0.012362 | 0 |
| (3,3) | -0.143666 | -0.143666 | 0 |

Finite, real, eV-scale (~0.15 eV diagonal), correct symmetric structure. No NaN/Inf anywhere
in the log or outputs.

## Evidence 3 — full output comparison vs pre-deletion reference

Diffs of `runs/MoS2/_e2e_bispinor_deletion_check_2026-07-02/` vs `runs/MoS2/C_60Ry_bispinor_cohsex_2026-06-15/`
(header timestamp line ignored):

- `eqp0.dat` — **BIT-IDENTICAL**
- `eqp1.dat` — **BIT-IDENTICAL**
- `sigma_diag.dat` — sigSX / sigCOH / sigTOT / VH all **bit-identical**. Only difference:
  the newer writer appends a cosmetic `Eo= <bare eigenvalue>` column per line (accounts for
  41 KB vs 37 KB file size). Not a regression.
- 9 k-points, 30 bands each in both; Kramers doubling intact (n=0≡n=1, etc.).

Agreement is 0 meV (exact), far better than the ~meV bar. No regression.

## Conclusion

The ~3000-line r-space V_q tile deletion (removal of `compute_V_q_bispinor_to_h5` + 2 dead
builder factories from `v_q_bispinor.py`, the `if _is_g_flat/else` collapse in `gw_init`, and
the `compute_vcoul` r-space tail) **did not break bispinor GW**. The live path
`compute_V_q_bispinor_g_flat_to_h5` runs cleanly, produces the 7 g-flat tiles, and yields
output bit-identical to the last known-good bispinor run.

## Notes / minor infra observations (not blockers)

- The lmod init path in the task instructions (`/etc/profile.d/z00_lmod.sh`) does not exist on
  this system; the correct one is `/etc/profile.d/zzz-lmod.sh`. This is only in the task prompt,
  not in any skill/doc, so no `KNOWN_SANDBOX_ERRORS.md` entry was warranted.
- `lxrun` is a `set_shell_function` from the `lorrax_agent` module: it is lost if `module load`
  is piped (e.g. `module load ... | tail`), matching the documented "never pipe module load"
  gotcha. Load in a bare statement.
- The session scratchpad under `/tmp/claude-.../scratchpad` is login-node-local and NOT visible
  to compute nodes (srun chdir fails → falls back to /tmp on the node). Bispinor runs must live
  on Lustre (`/pscratch/...`); the working copy was placed there.
- Benign during the run: XLA `Involuntary full rematerialization` warnings from
  `v_q_g_flat.py:94` (sharding-annotation perf note, present in the reference too) — not an error.
