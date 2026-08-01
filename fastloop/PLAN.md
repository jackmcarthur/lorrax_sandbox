# fastloop — single-process mini-deck pipeline

Status 2026-07-31: SCAFFOLD. Nothing here is certified; no jobid backs any
timing claim. This file states exactly what exists and what is planned.

## Problem

A semantic check today needs container + frozen bundle + sbatch: 30-120
minutes per iteration. Agents therefore reason instead of testing, which
is how keyword-mismatch-class bugs ship. The fix is a deck small enough
that the FULL pipeline (wfn ingest -> centroids -> ISDF -> chi/W -> sigma)
runs in ONE process in minutes, with sharding still exercised via
`XLA_FLAGS=--xla_force_host_platform_device_count=4` (2x2 mesh, host
devices, no MPI). The repo's `tools/probe_w_densifier_hlo.py` proves the
technique works for a single L2 routine; this area scales it to the
pipeline.

## What exists

- `run_minideck.py` — runner skeleton: env setup, device/mesh assertions,
  ordered stage stubs. Refuses to run past the first unimplemented stage
  and says so. Runs in-container on a compute node (interactive or a
  1-node dev job), never on login.

## What is planned (in order)

1. Deck generator: tiny 2x2-kgrid MoS2 deck (few bands, minimal cutoff).
   Two candidate routes — downsample the pinned 4x4 deck's WFN.h5, or
   synthesize free-electron-like wavefunctions with the right symmetries.
   Decision criterion: whichever preserves the code paths (spinor
   structure, symmetry maps, finite-q head) with less code.
2. Stage wiring: call the L1 drivers with the mini `cohsex.in`, one stage
   per flag, each stage dumping its artifact for the next.
3. Parity harness: per-stage reference values captured once from a
   certified full-size run (jobids recorded in CLAIMS.md), tolerance per
   stage; plus the HLO forbid-gate
   (`tools/hlo/analyze_hlo_dump.py --forbid all-gather,all-to-all`)
   on the sigma and W stages. Scope the forbid list per stage: the
   invariant is "no gather-class collective on an N_mu^2-class operand",
   and a stage that consumes a volume-preserving staged reshard
   (`common.staged_reshard` / `contract_bands_block_reshard`)
   legitimately emits exactly 2 shard-sized `all-to-all`s — gate that
   stage on count and operand size instead of the bare opcode
   (see `docs/HLO_HOWTO.md`).
4. Only after 1-3 hold: wire into the checkpoint skill as the mandatory
   pre-commit semantic check.

## Rules

- The mini-deck proves plumbing and sharding semantics, not physics
  accuracy: parity tolerances are loose, and no physics conclusion may
  cite a fastloop number.
- Any speedup or coverage claim from here goes through CLAIMS.md with the
  session's evidence like every other claim.
