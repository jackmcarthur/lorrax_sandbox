# Agent 1 — Core Physics / Algorithm Reference Docs

**Slice**: `docs/PHYSICS_COMPREHENSIVE.md`, `docs/ZETA_V_Q_ALGORITHMS.md`,
`docs/SYMMETRY_COMPREHENSIVE.md`, `docs/MEMORY_MODEL.md`, `docs/MINIMAX_QUADRATURE.md`

**Branch audited**: `agent/install-blitz-integration` on `sources/lorrax_D` (HEAD `3079a1f`)

---

## 1. Scope

I read all five docs in full plus spot-checked ~15 code citations against current source
in `src/`. Out of scope: sibling agent slices (environment docs, cohsex.in reference,
the GN-PPM guide, install/CI docs).

---

## 2. Per-doc verdict

| Doc | Lines | Verdict | One-sentence rationale |
|---|---:|---|---|
| `PHYSICS_COMPREHENSIVE.md` | 978 | **KEEP §1-2, §6-11; ARCHIVE §3-5 body (keep tombstone)** | §1-2 and §6-11 are accurate and dense; §3-5 describe a pipeline (r-space ζ-on-disk) replaced 2-3 months ago and are now misleading if read as current implementation. §11 is already a pointer stub — works as-is. |
| `ZETA_V_Q_ALGORITHMS.md` | 1145 | **KEEP** (small fixes needed — see §3) | Accurate, detailed, recently authored; the canonical current-implementation reference. Minor: source citations say `sources/lorrax_B` (not `lorrax_D`) and a few line numbers have drifted. |
| `SYMMETRY_COMPREHENSIVE.md` | 666 | **KEEP** | No material overlap with PHYSICS that isn't already cross-referenced. All three non-symmorphic gates verified 2026-05-15; code evidence in §1.4 solid. One `lorrax_B` ref in §9 header. |
| `MEMORY_MODEL.md` | 1004 | **KEEP** (AOT section stale — see §3) | The G-flat planner sections (§G-flat Memory Model) are current and validated. The AOT model section is explicitly flagged stale in the doc itself — honest, not misleading. No action needed other than a note. |
| `MINIMAX_QUADRATURE.md` | 340 | **KEEP** | Standalone math reference, no code-level citations that can drift, no supersession. Slightly overlaps PHYSICS §6.3 (CTSP) but the treatment is complementary (PHYSICS cites it), not duplicative. |

---

## 3. Per-section verdicts (where granularity matters)

### 3.1 `PHYSICS_COMPREHENSIVE.md` — section-level

| Section | Lines (approx) | Verdict | Notes |
|---|---|---|---|
| Header block (status note, §3-5 tombstone) | 1-32 | KEEP | The note accurately describes current state: §3-5 are the scalar narrative, §11 is the pointer. |
| §1 Wavefunctions and Notation | 33-73 | KEEP | Correct; foundational. |
| §2 ISDF Theory / pair products | 74-108 | KEEP | Correct; scalar narrative that §11 quick-ref refers back to. |
| §3 Galerkin System | 109-162 | ARCHIVE-IN-PLACE (keep with tombstone) | Describes spin-traced rank-3 pair density; current pipeline is rank-5 open-spin. The math is still correct as a conceptual entry point but function names `compute_CCT_from_left_right` / `compute_ZCT_from_left_right_zchunk` and reference to `common/isdf_fitting.py` **no longer exist** — renamed to `c_q_from_psi_sm` / `z_q_from_psi_sm`. Anyone reading §3 as implementation guidance will be confused. Recommend prepending a one-line `[HISTORICAL — functions renamed; see ZETA_V_Q_ALGORITHMS.md §11.3]` callout rather than deleting (the math derivation is genuinely useful). |
| §4 Zeta Fitting and V_q | 163-318 | ARCHIVE-IN-PLACE (same caveat as §3) | Several stale citations: `solve_zeta_from_L_q()` and `fit_zeta_chunked_to_h5()` do not exist (renamed `solve_zeta` / `fit_zeta_to_h5`); `compute_all_V_q_from_zeta_h5()` does not exist (replaced by `compute_all_V_q` / `compute_all_V_q_g_flat`). Disk layout described (`(nq, n_rtot, n_rmu)` flat-q) is the old r-space layout; G-flat layout is `(n_q_disk, n_rmu, ngkmax)`. Add the same tombstone callout. |
| §5 Chunking Strategy | 319-399 | ARCHIVE-IN-PLACE | All chunk formulas here pre-date the G-flat planner. `compute_optimal_chunks()` still runs but no longer controls `band_chunk` / `chunk_r` — the G-flat planner does. Mention of `docs/MEMORY_MODEL.md` at end of §5.3 is fine. |
| §6 Green's Functions and Self-Energy | 428-721 | KEEP | §6.1-6.8 describe static COHSEX and are accurate. §6.9 GN-PPM is accurate and has up-to-date code references. §6.7 "important frequency-dependent caveat" is load-bearing. |
| §7 JAX Sharding Summary | 722-779 | KEEP | Still accurate sharding map; tables match current source. |
| §8 File Organization | 780-828 | KEEP | Minor note: `gw/compute_vcoul.py` entry says `compute_all_V_q_from_zeta_h5()` — this function was renamed `compute_all_V_q`. Should update. |
| §9 Typical Workflow | 829-861 | KEEP | Accurately describes current workflow. |
| §10 Known Issues | 862-871 | KEEP | Still current. |
| §11 Bispinor-aware G-flat pipeline (pointer) | 872-977 | KEEP | §11.1 is a correct 1-paragraph context summary; §11.2 quick-reference table is useful. The pointer-to-ZETA_V_Q block is the right design. |
| Appendix quick reference | 963-977 | KEEP | Accurate formula table. |

### 3.2 `MEMORY_MODEL.md` — section-level

| Section | Verdict | Notes |
|---|---|---|
| Stage Summary (top table) | KEEP | Accurate for post-Round-6 pipeline. |
| Band Chunk / R-Chunk / Q-Chunk / μ-Chunk | KEEP | Accurate; matches current planner. |
| G-Flat Memory Model (Peaks A/B/C/D, planner algorithm) | KEEP | Validated by Si 4×4×4 and CrI3 runs. |
| Round-8 efficiency findings | KEEP | Good historical context; accurate. |
| AOT Memory Model (architecture, covered kernels) | **KEEP with a flag** | Doc honestly says `fit_one_rchunk` artifact is stale. The note at the bottom of §AOT Status is accurate — leave it. The section is not misleading because it calls out its own staleness explicitly. |
| Predicted-vs-realized faithfulness (Round-7 audit) | KEEP | The 7-8× over-prediction finding is important for anyone running the planner. |
| Appendix: Persistent Arrays (`live_arrays()` probe table) | KEEP | Dense but load-bearing; this is where you go when you see an unexpected shape in HBM. |

### 3.3 `ZETA_V_Q_ALGORITHMS.md` — section-level

All sections KEEP. The doc is well-structured and accurate. Sub-issues noted in §4 below.

### 3.4 `SYMMETRY_COMPREHENSIVE.md` — section-level

All sections KEEP. The gate table in §8 is verified and current. §12 Deferred Work is honest and accurate.

### 3.5 `MINIMAX_QUADRATURE.md` — section-level

All sections KEEP. The doc is stable math; no code citations that can drift.

---

## 4. Cross-cutting recommendations

### 4.1 Stale code citations (spot-check of ~15 citations)

The main drift is in **PHYSICS §3-5** and **SYMMETRY §9**:

| Doc | Cited | Actual (in `lorrax_D`) | Verdict |
|---|---|---|---|
| PHYSICS §4.5 "Triangular solve" | `solve_zeta_from_L_q()` in `common/isdf_fitting.py` | `solve_zeta()` at line 1214 | STALE — function renamed |
| PHYSICS §4.5 "Implementation" | `fit_zeta_chunked_to_h5()` | `fit_zeta_to_h5()` at line 1835 | STALE — function renamed |
| PHYSICS §4.6 "Implementation" | `compute_all_V_q_from_zeta_h5()` in `gw/compute_vcoul.py` | `compute_all_V_q()` at line 846 | STALE — function renamed |
| PHYSICS §8 file table | `compute_all_V_q_from_zeta_h5()` | `compute_all_V_q()` | STALE |
| ZETA_V_Q §11 header | `sources/lorrax_B` | canonical checkout is `sources/lorrax_D` | STALE (cosmetic) |
| SYMMETRY §9 header | `LORRAX (sources/lorrax_B/)` | `sources/lorrax_D` | STALE (cosmetic) |
| SYMMETRY §2.1 | `SymMaps` at `symmetry_maps.py:416-1143` | Class starts at line **729** (not 416) | STALE — line range drifted significantly |
| SYMMETRY §9 table | `SymMaps` at `src/common/symmetry_maps.py:416-1143` | line 729 | STALE |
| SYMMETRY §3.1 | `unfold_psi` at `symmetry_maps.py:306-413` | line **618** | STALE |
| ZETA_V_Q §11.3 | `_make_fit_one_rchunk_kernel` at `isdf_fitting.py:1467-1624` | `1556` | within the cited range but range start is stale |
| ZETA_V_Q §11.3 | `fit_one_rchunk` at `isdf_fitting.py:1627-1717` | `1697` | stale |
| ZETA_V_Q §11 file table | `_make_read_q` at `v_q_g_flat.py:234-287` | function is `_make_read_all_ibz` at line 232 | STALE — renamed |
| SYMMETRY §9 | `compute_centroid_sym_perm` at `orbit_syms.py:229-437` | line **180** | stale start |
| SYMMETRY §9 | `compute_rgrid_sym_perm` at `orbit_syms.py:444-551` | line **395** | stale start |
| MEMORY §G-flat | `accumulate_rchunk_to_gflat` at `wfn_transforms.py:439-626` | line **1006** (file is 1357 lines, not a short module) | range completely wrong |

**Recommendation**: Do not maintain per-line citations in prose docs. The §3-5 citations
are the most dangerous because they describe functions that no longer exist by name.
Convert to function-name citations only (drop line ranges), and add a top-of-file note
in ZETA_V_Q that its source citations target `lorrax_D` not `lorrax_B`.

### 4.2 Equation numbering

No cross-doc equation reuse conflict — docs use boxed display equations without sequential
numbering. No renumbering risk if sections move. No action needed.

### 4.3 Symbol consistency

| Issue | Where | Status |
|---|---|---|
| `Σ_X` vs `Σ^X` | Mixed throughout PHYSICS (both forms appear) | Low priority; conventional in physics writing |
| `ζ` vs `zeta` | Both forms used in ZETA_V_Q (prose uses ζ̃, code uses `zeta`) | Intentional: math uses ζ̃, code names use `zeta`. Clear. |
| `μ_L = i` (bispinor indefinite CCT) | PHYSICS §11.1, ZETA_V_Q §11.3.2 | Consistent: both say pivoted-LU for μ_L≠0. MEMORY MEMORY.md pin `project_bispinor_isdf.md` says "μ_L=i CCT is indefinite, must use LU" — matches docs. |
| `q_irr_frac` sign convention | SYMMETRY §2.4, ZETA_V_Q §11.6.1 | Consistent: both use BGW `(-1/2, 1/2]` wrap. |
| `extend_trs=True` | SYMMETRY §4.4, ZETA_V_Q §11.6.1, SYMMETRY §5.5 | Consistent: all three describe the same `compute_centroid_sym_perm` argument. |

### 4.4 Internal cross-references

I checked the main "see X §Y.Z" pointers:

| Pointer | Valid? |
|---|---|
| PHYSICS §5.3 → `docs/MEMORY_MODEL.md` | Yes — MEMORY_MODEL has the G-flat planner section |
| PHYSICS §6.3 → `docs/MINIMAX_QUADRATURE.md` | Yes |
| PHYSICS §11.2 quick-ref → `ZETA_V_Q_ALGORITHMS.md` | Yes |
| PHYSICS §8 doc-table → `SYMMETRY_COMPREHENSIVE.md` | Yes — §2 and §5 match |
| PHYSICS §8 doc-table → `MEMORY_MODEL.md` | Yes |
| ZETA_V_Q §11.2 → `MEMORY_MODEL.md` | Yes — dedicated §Memory Model section exists |
| ZETA_V_Q §11.6.3 → `SYMMETRY_COMPREHENSIVE.md` | Yes — §4.4 `extend_trs` description matches |
| SYMMETRY §5.5 → `project_lorrax_ibz_cascade.md` | Pointer to memory file, not a doc; acceptable for now |
| MEMORY §IBZ Cascade Memory → `gw/v_q_g_flat.py :: _resolve_ibz_q_list` | Function exists at line 152 |
| ZETA_V_Q §11.10.2 → `reports/zeta_v_q_g_flat_reference_2026-05-12/report.md` | Not checked (report dirs are in sandbox, not lorrax_D), but living reference so OK |

All intra-doc section references are valid (e.g. "§7 Sharding Summary" in PHYSICS §8
exists; "§3-5" referenced by §11 exists).

### 4.5 Overlap between docs (deduplication opportunities)

| Overlap | Magnitude | Recommendation |
|---|---|---|
| PHYSICS §3-5 vs ZETA_V_Q §11 | Large — the algorithm pipeline is described in both, but for different versions of the code | Not a pure duplicate; §3-5 is the HISTORICAL path, ZETA_V_Q is CURRENT. Keep both with clearer tombstones on §3-5. |
| PHYSICS §11.2 quick-ref table vs ZETA_V_Q §11.11 quick-ref | Moderate — both contain formula tables | They're nearly identical (~10 rows each, same formulas). Reasonable to delete PHYSICS §11.2's table body and replace it with "see ZETA_V_Q_ALGORITHMS.md §11.11 for the current formula table." Saves ~25 lines of maintenance surface. |
| MEMORY §IBZ Cascade Memory vs ZETA_V_Q §11.6 | Small — MEMORY §IBZ gives disk-size numbers; ZETA_V_Q §11.6 gives algorithm | Complementary, not duplicative. No action. |
| PHYSICS §6.3 CTSP / minimax vs MINIMAX_QUADRATURE | Complementary — PHYSICS §6.3 describes the GW driver's use, MINIMAX_QUADRATURE is the solver math. The "Full derivation in MINIMAX_QUADRATURE.md" pointer makes this explicit. No action. |
| SYMMETRY §5.4 JIT body vs ZETA_V_Q §11.6.3 | Small — both describe `unfold_v_q`. SYMMETRY gives the derivation + implementation notes; ZETA_V_Q gives the algorithm-level context. Keep both; they serve different readers. |

---

## 5. What I'd write fresh

None of the gaps in this slice are urgent. The docs are dense and largely accurate for
the current pipeline. Two small additions would pay off:

**5.1 A "quick-orient" header for ZETA_V_Q_ALGORITHMS.md** (~10 lines).
The file opens with "This section is the single source of truth..." (the old §11 heading
is still the document title "# 11. ζ-fit and V_q ..."). A reader landing here for the
first time doesn't know whether to read top-to-bottom or jump to a stage. A 3-5 bullet
"reading guide" (§11.1 = overview, §11.3 = kernel details, §11.6 = IBZ cascade, §11.7 =
sharding/collectives, §11.8 = memory) would reduce orientation cost significantly.
Target: ~10 lines.

**5.2 A one-page "where to look" primer** that maps physics questions to the right doc.
The current split (PHYSICS = conceptual + scalar narrative; ZETA_V_Q = current
implementation; SYMMETRY = conventions; MEMORY = sizing) is correct but not stated
anywhere. A 20-line table in PHYSICS §8 (or a new top-level `docs/INDEX.md`) would save
time for anyone re-entering the codebase. The information is implicit in PHYSICS §8's
doc table; making it explicit is ~30 minutes work. Not urgent if the author already holds
this map in memory.

---

## 6. Open questions

1. **Is PHYSICS §3-5 safe to permanently delete?** The math in §3 (Galerkin derivation,
   CCT/ZCT derivation) is still the correct conceptual underpinning of the G-flat
   pipeline — just not the implementation. Deleting it removes the only place where the
   Galerkin normal equations are derived from first principles. My recommendation is to
   tombstone in-place, but the author may prefer to delete and rely on the paper (Lu &
   Ying JCTC 2015) for the derivation.

2. **`compute_all_V_q_from_zeta_h5` in PHYSICS §4.6 and §8**: This function no longer
   exists. The replacement `compute_all_V_q` (at `compute_vcoul.py:846`) is a legacy
   r-space path; the current G-flat path is `compute_all_V_q_g_flat` in
   `gw/v_q_g_flat.py`. Should PHYSICS §4.6 be updated to point to `v_q_g_flat.py` or
   left as a historical description of the r-space path? The former risks making §4.6
   redundant with ZETA_V_Q; the latter is cleaner.

3. **AOT model re-sampling priority**: MEMORY §AOT Status (2026-05-15) says
   `fit_one_rchunk` artifact is stale against the post-Round-6 fused kernel. Is this on
   a near-term roadmap, or should the AOT section be shortened to remove the
   `fit_one_rchunk` table entry until it's re-sampled?

4. **`accumulate_rchunk_to_gflat` attribution**: ZETA_V_Q §11.4 cites
   `wfn_transforms.py:439-626` but the function is actually at line 1006 in a 1357-line
   file. This range appears in multiple places (ZETA_V_Q §11.8 memory model, §11.7.3
   collective inventory). All need updating — but someone who knows the recent file
   reorganization history should confirm whether the file was extended or the citation
   was always wrong.

5. **`lorrax_B` vs `lorrax_D` in ZETA_V_Q and SYMMETRY**: The docs say "refer to
   `sources/lorrax_B`" in their headers. This is the clone letter at time of authorship.
   Should citations be agent-letter-agnostic (just `src/`) or should the header note be
   updated? Since `lorrax_D` is the integration branch checkout, updating to `src/`
   (relative to the repo root) would be more stable.
