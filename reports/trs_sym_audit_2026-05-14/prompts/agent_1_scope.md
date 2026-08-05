# Agent 1 — Scope: every TRS-blind sym site

You are Agent 1 of a four-agent team auditing TRS handling in LORRAX. Read `reports/trs_sym_audit_2026-05-14/STATUS.md` first. Working tree is `sources/lorrax_B` (do not edit code yet).

## Your job

Enumerate every site in the LORRAX source where the choice of `sym_matrices` (length `ntran`, spatial only) vs. `sym_mats_k` (length `2·ntran`, TRS-augmented) matters. For each site, determine:

1. Which sym table it currently uses.
2. Which sym table it SHOULD use given the physical operation.
3. If they don't match: what's the failure mode (wrong number? exception? silent clip? something else)?
4. Whether the site is currently exercised in production runs (vs. dead/unused).

## Where to look (starting points — not exhaustive)

- `src/common/symmetry_maps.py` — `find_symmetry_ops_simple`, `find_irreducible_qpoints`, anything that builds `sym_mats_k` or `sym_matrices`
- `src/centroid/orbit_syms.py` — `compute_centroid_sym_perm`, `unfold_orbit_unique_with_id`, `snap_orbits_to_grid`
- `src/gw/v_q_tile.py:_unfold_v_q_ibz_to_full` — the known bug site
- `src/gw/v_q_g_flat.py:_resolve_ibz_q_list` — the gate that wires sym_perm into V_q
- `src/common/isdf_fitting.py:fit_zeta_to_h5` — the IBZ-only ζ solve path (lines ~1591-1610) and the q_irr_full_idx call site
- `src/file_io/zeta_loader.py` and Σ_X consumer paths — does the Σ_X ζ reader unfold IBZ→full? If yes, how?
- `src/common/load_wfns.py` — ψ unfolding from IBZ k to full-BZ k. Does it apply complex conjugation when sym >= ntran?
- Any bispinor-specific TRS handling (Pauli-channel TRS rotation is non-trivial; γ̃ might transform differently)

For each site, **grep for both `sym_matrices` and `sym_mats_k`** uses and the index pattern (`[:ntran]`, `[ntran:]`, direct indexing by an integer that came from `find_irreducible_qpoints` or `find_symmetry_ops_simple`).

## Output

Write your report to `reports/trs_sym_audit_2026-05-14/agent_1_scope_report.md` with this structure:

```markdown
# Scope: TRS-blind sym sites

## Summary table

| # | File:line | Function | Current sym table | Correct sym table | Failure mode if TRS row hit | Exercised? |
|---|-----------|----------|-------------------|-------------------|------------------------------|------------|
| 1 | v_q_tile.py:1452 | _unfold_v_q_ibz_to_full | sym_perm (spatial, ntran) | needs TRS+conj | silent OOB clip → wrong V_q | YES (IBZ cascade) |
| ... | ... | ... | ... | ... | ... | ... |

## Per-site detail

### Site #1: `_unfold_v_q_ibz_to_full`
... (5-15 lines explaining the math, the code, and what TRS handling would look like)
```

## Hard constraints

- **NO code edits** — read-only audit. Save the actual patching for Agent 2.
- For each finding, write out the per-element math (NOT just "uses sym_mats_k"). See [[agent-audit-failure-modes]] in memory for why this matters — prior audits failed by pattern-matching strings without index-tracking.
- **Verify each finding with one concrete example**: e.g. "On MoS2 3×3, this site sees `sym_idx=2` (TRS), which clips to `sym_perm[1]` (mirror), producing wrong output X for input Y."
- If you discover a site I didn't list, INCLUDE IT. The starting list isn't exhaustive.

## Done criterion

Report exists at the path above, with a populated summary table covering ≥6 sites, and detail sections for every site marked "Exercised? YES". Ping `discussion.md` with `[Agent 1] scope complete; N sites found, M exercised in production` and stop.

## What you have access to

The full source tree at `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/`. Run files in `/pscratch/sd/j/jackm/lorrax_sandbox/runs/` for cross-checking. SLURM alloc `52953227` is alive if you need to actually run something (you probably don't for a read-only audit; if you do, use `lxrun`).
