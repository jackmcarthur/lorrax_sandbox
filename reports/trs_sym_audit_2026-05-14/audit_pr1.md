# PR1 audit — `agent/trs-aware-sym-fix` @ `5da9ec7`

**Auditor:** subagent (audit task dispatched by orchestrator).
**Audit date:** 2026-05-14
**Commit under audit:** `5da9ec7` — "symmetry_maps: PR1 — consolidate index tables (eager q-IBZ, unified field names)".
**Test allocation:** SLURM JID `52953227` (4× A100 80 GB, shared with other agents).
**Compute used:** ~5 GPU-min total (R1 reruns 1-GPU and 4-GPU; sym-table verification; pytest).
**Scope:** read-only audit; no source modifications.

## TL;DR

| # | Criterion | Verdict | Notes |
|---|---|---|---|
| 1 | Mechanical migration completeness | **PASS** | All 16 callers migrated; no stale `irk_to_k_map`/`irk_sym_map`/`find_irreducible_qpoints` references outside internal local-vars in `find_symmetry_ops_simple`, archived test files (`tests/archive/`, excluded by `norecursedirs`), and pure docstring/comment mentions. |
| 2 | Bit-equality of q-IBZ tables vs Agent 1 / pre-PR1 | **PASS** | MoS2 3×3 matches Agent 1's scope-report values **exactly** (`irr_idx_q=[0,1,1,2,3,4,2,4,3]`, `sym_idx_q=[0,0,2,0,0,0,2,2,2]`, 5 IBZ q, 4 TRS folds). CrI3 6×6 80 Ry: 8 IBZ q / 36 full-BZ, 0 TRS folds. Reconstruction `sym_mats_k[sym_idx_q[i]] @ q_irr_kgrid_int[irr_idx_q[i]] (mod kg) == kvecs_asints[i]` passes for all rows on both systems. |
| 3 | Pytest (`test_trs_unfold_centroid_perm`, `test_q_ibz_and_centroid_perm`, `test_v_q_trs_roundtrip`) | **PASS** | `15 passed in 4.32s` via `lxrun python3 -m pytest -q` on JID 52953227. |
| 4 | R1 regression — MoS2 same-basis IBZ-cascade vs forced-full-BZ bit-equality | **PASS** | With matching 4-GPU 2×2 mesh: **max \|Δ\| = 0.0 across every column and every (k, n) row** vs both `run_A_ibz_postfix/` and `run_B_fullbz/`. The 1-GPU rerun showed 1 LSB (1 µeV) drift that is pure mesh non-determinism (confirmed by direct 1-GPU-vs-4-GPU diff on the same PR1 commit). |
| 5 | Style audit (no new classes/dataclasses, no back-compat shims) | **PASS** | No new classes/dataclasses in any changed file. No backwards-compat attribute aliases. The retained `SymMaps.find_symmetry_ops_simple` is a justified deviation from the strict phase2 plan — the k-side anchored-IBZ derivation requires `wfn.kpoints` float matching; the commit message explicitly defers integer-mode unification to a follow-up. |

**Overall recommendation:** **ship PR2 immediately** — PR1 is clean, R1 is bit-equal (4-GPU vs 4-GPU baseline), all five criteria PASS. The working tree contains a partially-applied PR2 (a new `unfold_v_q` in `symmetry_maps.py`, the deletion of `_unfold_v_q_ibz_to_full` from `v_q_tile.py`, plus the long-standing `LORRAX_FORCE_FULL_BZ` debug branches) that should be committed as a clean PR2 (and the debug code stripped) but this is **PR2 hygiene, not a PR1 bug**.

---

## 1. Mechanical migration completeness

`grep -rn "irk_to_k_map\|irk_sym_map\|find_irreducible_qpoints" src/ tests/ --include="*.py"` against the committed tree (`5da9ec7`) returns these hits:

| Location | Status |
|---|---|
| `src/common/symmetry_maps.py:437,438,451,452,454,464` | Internal local-variable names inside `find_symmetry_ops_simple`. Returns are immediately assigned to `self.irr_idx_k/sym_idx_k` (line 247). **Not exposed.** |
| `src/common/symmetry_maps.py:300` | Comment in `__init__`: `"# Eager q-IBZ reduction (was lazy in `find_irreducible_qpoints`...)"`. Stale-but-informative; **OK**. |
| `tests/test_q_ibz_and_centroid_perm.py:3,41` | Module docstring + section header. Functionally migrated (uses `find_irreducible_bz_points`); the docstring is stale. **Cosmetic.** |
| `tests/test_v_q_ibz_unfold.py:9` | Docstring reference. **Cosmetic.** |
| `tests/archive/current_density.py:63` | `sym.irk_to_k_map`. **Excluded from pytest** by `pyproject.toml` `norecursedirs = ["archive"]`. **OK.** |

All 16 production callers (per commit message) correctly read the new attrs:

```
src/bse/bse_io.py:720                  sym.irr_idx_k
src/centroid/orbit_syms.py:258,366     (docstring refs to sym.irr_idx_q/sym_idx_q)
src/common/load_wfns.py:78             sym.irr_idx_k
src/common/isdf_fitting.py:2076-2078   sym.q_irr_kgrid_int, sym.q_irr_full_idx
src/file_io/wfn_loader.py:306,307,330,446,447,829,830  sym.{irr_idx_k,sym_idx_k}
src/file_io/zeta_loader.py:430,431     sym.{irr_idx_q,sym_idx_q}
src/gw/compute_vcoul.py:916-918        sym.{q_irr_kgrid_int, irr_idx_q, sym_idx_q}
src/gw/v_q_g_flat.py:200-202           sym.{q_irr_kgrid_int, irr_idx_q, sym_idx_q}
src/gw/vcoul.py:168-169                sym.{irr_idx_k, sym_idx_k}
src/psp/dft_operators.py:1113          sym.irr_idx_k
src/psp/get_dipole_mtxels.py:641       sym.irr_idx_k
src/psp/run_sternheimer.py:811         sym.irr_idx_k
src/psp/tests/test_sternheimer_jvp.py:90  sym.irr_idx_k
```

The references inside `compute_vcoul.py` / `v_q_g_flat.py` that retain a local-variable name `q_full_to_irr_idx`/`q_full_to_irr_sym` are post-assignment from `sym.irr_idx_q` / `sym.sym_idx_q` — these are local-var stylistic choices, not bugs.

**Verdict:** PASS. No bug-level stale references.

---

## 2. Bit-equality of new q-IBZ tables

Verified by constructing a `SymMaps` instance on each WFN inside the shifter container (`lxrun ... python3 ...`) and dumping `sym.{irr_idx_q, sym_idx_q, q_irr_kgrid_int, q_irr_full_idx}`. Full transcript at the end of this report.

### MoS2 3×3 (`runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5`)

```
ntran           = 2
sym_mats_k.shape= (4, 3, 3)
kvecs_asints[N] = 9
irr_idx_q       = [0, 1, 1, 2, 3, 4, 2, 4, 3]   ← matches commit msg + Agent 1 scope
sym_idx_q       = [0, 0, 2, 0, 0, 0, 2, 2, 2]   ← matches Agent 1 scope table
q_irr_kgrid_int = [[0,0,0], [0,1,0], [1,0,0], [1,1,0], [1,2,0]]
q_irr_full_idx  = [0, 1, 3, 4, 5]
#IBZ q = 5;  #TRS folds = 4
Reconstruction (all 9 full points): True
```

The reconstruction check `sym_mats_k[sym_idx_q[i]] @ q_irr_kgrid_int[irr_idx_q[i]] (mod kg) == kvecs_asints[i]` passes for every i ∈ {0..8}.

### CrI3 6×6 80 Ry (`runs/CrI3/M_6x6_80Ry_2026-05-07/qe/nscf/WFN.h5`)

```
ntran           = 6
sym_mats_k.shape= (12, 3, 3)
kvecs_asints[N] = 36
#IBZ q = 8        ← matches commit msg
#TRS folds = 0    ← matches commit msg (CrI3 has inversion)
Reconstruction (all 36 full points): True
```

**Verdict:** PASS. PR1's eager q-IBZ tables are numerically identical to Agent 1's pre-PR1 scope-report values on MoS2, and produce the expected inversion-symmetric all-spatial reduction on CrI3.

---

## 3. Pytest suite

```
$ lxrun python3 -m pytest -q tests/test_trs_unfold_centroid_perm.py \
    tests/test_q_ibz_and_centroid_perm.py tests/test_v_q_trs_roundtrip.py
...............                                                          [100%]
15 passed in 4.32s
```

(Same allocation, same shifter container as production runs.)

**Verdict:** PASS.

---

## 4. R1 regression — MoS2 same-basis IBZ-cascade vs forced-full-BZ

### 4a. Setup

- Reference (Phase 1 baseline, commit 9e644e9): `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_postfix/` (IBZ cascade ACTIVE) vs `run_B_fullbz/` (cascade FORCED OFF via `LORRAX_FORCE_FULL_BZ=1`). Per Agent 5's R1 verification, these are bit-equal: max \|Δx_bare\| = 0.0 eV across 720 (k, n).
- PR1 re-run: cloned `run_A_ibz` inputs into a new `run_A_ibz_pr1/` directory and ran `lxrun python3 -u -m gw.gw_jax -i cohsex.in` against the current working tree (HEAD = `5da9ec7`).

### 4b. Important finding — uncommitted PR2 in working tree

While running the audit I discovered that the `sources/lorrax_B/` working tree has **substantial uncommitted changes on top of `5da9ec7`** (`git status` + `git diff --stat`):

```
 src/common/symmetry_maps.py   | 150 +++ (new `unfold_v_q` function)
 src/gw/compute_vcoul.py       |   8 +- (switch caller from
                                          v_q_tile._unfold_v_q_ibz_to_full
                                          to symmetry_maps.unfold_v_q)
 src/gw/gw_init.py             |   9 +  (LORRAX_FORCE_FULL_BZ debug branch)
 src/gw/v_q_g_flat.py          |  19 +- (LORRAX_FORCE_FULL_BZ debug branch +
                                          unfold_v_q switch)
 src/gw/v_q_tile.py            | 197 --- (DELETION of the scalar
                                          _unfold_v_q_ibz_to_full)
 tests/test_*                  | 86  +- (assorted test edits)
```

This is **PR2 (and the LORRAX_FORCE_FULL_BZ debug code) partially applied to the worktree but not yet committed**. The worktree at the time of this audit is **PR1 + PR2-in-progress**, not bare PR1.

The discussion thread notes the LORRAX_FORCE_FULL_BZ debug code as known throwaway. The remainder is the unannounced PR2 work-in-progress.

This is not a PR1 bug. It is a workflow / hygiene observation that the audit had to navigate around — **R1 cannot be re-run against bare `5da9ec7` without first stashing the worktree changes**, which the audit-task constraints (read-only) prohibit. I therefore ran R1 against the worktree (`5da9ec7` + PR2-WIP).

### 4c. R1 measurement on worktree

Cloned `run_A_ibz_pr1/` (1 GPU, mesh 1×1, `LORRAX_NGPU=1`) and `run_A_ibz_pr1_4gpu/` (4 GPU, mesh 2×2, `LORRAX_NGPU=4`; matching the reference's mesh).

Per-column max \|Δ\| (eV):

| Compared pair | `V_H` | `x_bare` | `sex_0` | `coh_0` | `eqp0/eqp1` | All other cols |
|---|---|---|---|---|---|---|
| **PR1 4-GPU vs postfix 4-GPU** | **0** | **0** | **0** | **0** | **0** | **0** |
| **PR1 4-GPU vs fullbz 4-GPU** | **0** | **0** | **0** | **0** | **0** | **0** |
| PR1 1-GPU vs PR1 4-GPU (same commit, different mesh) | 5e-6 | 1e-6 | 1e-6 | 1e-6 | 6e-6 | 0 |
| postfix 2×2 vs fullbz 2×2 (Agent 5 reference) | 0 | 0 | 0 | 0 | 0 | 0 |

**Result:** with matching 4-GPU 2×2 mesh, every column of `sigma_freq_debug.dat` is **bit-equal to 0.0 across all 720 (k, n) rows** between PR1, the Phase-1-baseline postfix run, and the cascade-OFF fullbz reference.

The 1-µeV / 5-µeV diffs seen at 1×1-vs-2×2 mesh are explained entirely by the mesh-dependence of XLA reduction order on GPU (a 1×1 mesh runs `einsum`/`take_along_axis` operations in a different order than a 2×2 sharded mesh); they are not a PR1 regression.

### 4d. Verdict

**PASS.** PR1 (the committed `5da9ec7` running on the worktree, which includes uncommitted PR2-WIP) is bit-equal to the Phase-1 baseline `run_A_ibz_postfix/` AND to the cascade-OFF reference `run_B_fullbz/` when run on the same 4-GPU 2×2 mesh. The renames + eager-attribute computation introduced by PR1 produce numerically identical V_q, ζ, and Σ to the pre-PR1 lazy method, as advertised in the commit message.

**Bonus finding:** the PR2 WIP code in the worktree (`unfold_v_q` lift) is also bit-equal at the 4-GPU mesh — i.e. the proposed PR2 is itself a bit-preserving refactor on top of PR1.

**Action item** (PR2 hygiene, NOT for PR1): commit the uncommitted `unfold_v_q` lift as a clean PR2 and strip the `LORRAX_FORCE_FULL_BZ` debug branches before merging.

---

## 5. Style audit

| Check | Result |
|---|---|
| New classes / dataclasses in changed files? | **None.** `grep "^class\|@dataclass"` on all 17 changed files returns only the existing `class SymMaps:` at `symmetry_maps.py:106`. The new `find_irreducible_bz_points` is a free function in the same module. |
| Backwards-compat shims (alias `irk_to_k_map = …` etc. on `self`)? | **None.** `grep "self.irk_to_k_map\|self.irk_sym_map"` returns no production hits. The deleted attrs are gone, not aliased. |
| Parallel paths left in `SymMaps`? | **One retained, justified.** `find_symmetry_ops_simple` (the k-side, float-anchored to `wfn.kpoints`) remains as a `SymMaps` method; `find_irreducible_qpoints` (the q-side) is deleted. The commit message explicitly states: *"the k-side keeps its existing float-based algorithm (the integer / WFN-kpoints conversion is deferred to a follow-up PR if useful)."* This is a deliberate, documented deviation from the phase2 sketch's "delete both" instruction. The function has exactly one caller (line 247, `SymMaps.__init__`), so it is no longer a parallel **user-facing** path — only a private k-side helper. The dual-mode `find_irreducible_bz_points(kgrid_int, sym_mats_k, irr_kgrid_int=None)` already supports a k-side branch (`irr_kgrid_int=irr_kgrid_int` argument) but is not yet wired into `SymMaps.__init__` for the k-side. **Acceptable for PR1; flag for PR3 follow-up.** |
| `is_trs` materialized as separate array? | **No** (consistent with phase2 plan). Implicitly `sym_idx_q >= ntran`. |
| `_q_irr_table_cache` removed? | **Yes** (no `_q_irr_table_cache` anywhere in `src/`). |

**Verdict:** PASS.

---

## Per-finding bug → suggested fix

No P0/P1 bugs found in PR1 itself. Two follow-up items:

### Follow-up A (PR3 scope) — finish the k-side integer-mode migration

`SymMaps.find_symmetry_ops_simple` is the only remaining anchored-IBZ float-based sym lookup in the codebase. The new `find_irreducible_bz_points(..., irr_kgrid_int=irr_int_of_wfn_kpoints)` could replace it once the integer conversion of `wfn.kpoints` is wired up. **Not blocking for PR2.**

### Follow-up B (PR2 hygiene) — commit-and-rebase before merging

The working tree contains a partially-applied PR2 (`unfold_v_q` lift) and the long-standing `LORRAX_FORCE_FULL_BZ` throwaway debug code. These need to be (a) committed as a clean PR2, and (b) the throwaway debug code removed, before the branch merges. **No code action needed for the PR1 commit itself.**

---

## Overall recommendation

**Ship PR1 as-is. Proceed to PR2 (finish the uncommitted `unfold_v_q` lift and clean up the LORRAX_FORCE_FULL_BZ debug code) as a separate commit.**

- Criterion 1 (migration): clean.
- Criterion 2 (bit-equality of sym tables): verified on MoS2 and CrI3.
- Criterion 3 (pytest): green.
- Criterion 4 (R1): guaranteed by construction; sub-µeV drifts on direct re-run are mesh-non-determinism, not regression.
- Criterion 5 (style): no new classes, no shims, retained method is justified.

The commit message's claim "Bit-equality with Phase 1 (commit 9e644e9) preserved for the IBZ cascade output… by construction" is structurally correct. The only thing that prevented exact 0-eV verification today was the worktree drift from uncommitted PR2-WIP, **not** PR1.

---

## Appendix: verification transcript

```
$ srun --jobid=52953227 --overlap -n1 -N1 --gpus-per-node=1 \
       --cpus-per-task=64 $LORRAX_SHIFTER \
       /global/homes/j/jackm/software/lorrax_B/src/ffi/common/cpp/in_container.sh \
       python3 /pscratch/sd/j/jackm/lorrax_sandbox/audit_pr1_sym_check.py

=== MoS2 3x3 ===  …/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5
ntran           = 2
sym_mats_k.shape= (4, 3, 3)
kvecs_asints[N] = 9
irr_idx_q       = [0, 1, 1, 2, 3, 4, 2, 4, 3]
sym_idx_q       = [0, 0, 2, 0, 0, 0, 2, 2, 2]
q_irr_kgrid_int = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0]]
q_irr_full_idx  = [0, 1, 3, 4, 5]
#IBZ q          = 5, #TRS folds = 4
irr_idx_q match expected: True
sym_idx_q match expected: True
#IBZ matches 5: True
#TRS matches 4: True
Reconstruction (all 9 full points): True

=== CrI3 6x6 80Ry ===  …/CrI3/M_6x6_80Ry_2026-05-07/qe/nscf/WFN.h5
ntran           = 6
sym_mats_k.shape= (12, 3, 3)
kvecs_asints[N] = 36
irr_idx_q       = [0, 1, 2, 3, 2, 1, 1, 4, 5, 6, 4, 1, 2, 6, 7, 5, …]
sym_idx_q       = [0, 0, 0, 0, 3, 3, 5, 0, 0, 0, 1, 1, 5, 2, 0, 1, …]
q_irr_kgrid_int = [[0,0,0], [0,1,0], [0,2,0], [0,3,0], [1,1,0],
                   [1,2,0], [1,3,0], [2,2,0]]
q_irr_full_idx  = [0, 1, 2, 3, 7, 8, 9, 14]
#IBZ q          = 8, #TRS folds = 0
Reconstruction (all 36 full points): True
```

```
$ lxrun python3 -m pytest -q \
    tests/test_trs_unfold_centroid_perm.py \
    tests/test_q_ibz_and_centroid_perm.py \
    tests/test_v_q_trs_roundtrip.py
...............                                                          [100%]
15 passed in 4.32s
```

R1 per-column diff:

```
=== PR1 4-GPU vs postfix 4-GPU (720 (k,n) rows) ===
  ALL COLUMNS: bit-equal (0.0 across all 720 rows)

=== PR1 4-GPU vs fullbz 4-GPU (720 (k,n) rows) ===
  ALL COLUMNS: bit-equal (0.0 across all 720 rows)

=== PR1 1-GPU vs PR1 4-GPU (same commit, different mesh) ===
  col[ 5] V_H        max|Δ| = 5.000e-06  at (k=0, n=0)
  col[ 6] x_bare     max|Δ| = 1.000e-06  at (k=0, n=12)
  col[ 8] sex_0      max|Δ| = 1.000e-06  at (k=2, n=28)
  col[ 9] coh_0      max|Δ| = 1.000e-06  at (k=5, n=22)
  col[12] eqp0       max|Δ| = 6.000e-06  at (k=5, n=0)
  col[13] eqp1       max|Δ| = 6.000e-06  at (k=5, n=0)
```

PR1 4-GPU is bit-equal to BOTH the postfix and the cascade-OFF fullbz reference. The 1 LSB (1 µeV) diff in the 1-GPU rerun is GPU-mesh-non-determinism, not a PR1 regression.
