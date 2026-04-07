# Memory Assay: Handoff (SUPERSEDED)

**This document is superseded by
`reports/memory_model_assay_2026-04-06/report.md`.**

The plan, goals, and pitfalls of previous work are documented there.
The files in this directory remain as raw data but their conclusions
(particularly the "9x shard" model) should not be used — they were
measured in single-process mode which is invalid for production.

## What to keep from this directory

- `run_assay.py` — useful as a code template for the sweep script,
  but must be rewritten to run multi-process and cover the full pipeline
- `cohsex.in`, `centroids_frac_240.txt` — input files, still valid
- `tmp/` — ISDF tensor outputs from the 4-proc run, can verify pipeline

## What to discard or ignore

- `assay_results.json`, `assay.out` — single-process, invalid
- `detailed_trace.out` — single-process, invalid
- `collective_buffers.out` — single-process, incomplete
- `assay_report.md` — superseded
- `detailed_buffer_inventory.md` — single-process, decompositions are
  inferred (not from XProf), and some entries are speculative
