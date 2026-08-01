# fastloop — mini-deck driver-chain loop

Status 2026-07-31: BUILT AND CERTIFIED (CLAIMS rows 17-19). Deck built by
job 7884926 (QE leg green); pins + both check legs green in job 7884936:
p1 rerun exact-0 on every compared quantity, shard4 (2x2 host-device
mesh) worst delta 1.15e-13 eV (sigma_mnk.h5), parity verdict PASS.
Warm-loop chain wall: 76.6 s (p1), 107.5 s (shard4, includes its fresh
4-device compiles). This file states exactly what exists, what is
certified (with jobids), and what remains.

## Usage (the whole point — run this before committing driver changes)

    cd /scratch2/08271/jackmc/lorrax_sandbox
    sbatch fastloop/run_fastloop.sbatch        # 1-node dev job, ~5 min
    # ... or, inside ANY existing allocation (idev / a bigger job's step):
    bash fastloop/run_fastloop.sbatch

Exit 0 = the full chain reproduced the pinned baseline on BOTH legs.
Exit 1 = numeric drift (per-file deltas printed in the parity report).
Exit 2 = a driver stage failed rc!=0 (last 25 log lines echoed; full
logs under `fastloop/work/check.<jobid>/<leg>/logs/`).
Exit 3 = refusal (login node, missing deck/pins).

- Tests YOUR working tree: `LORRAX_SRC=/path/to/src` (default: the live
  repo src). Knobs: `FASTLOOP_MODE=pin|check`, `FASTLOOP_ARGS`,
  `FASTLOOP_THREADS`. Direct runner help:
  `run_fastloop.py --help` (in-container only).
- Never edit `pins_mini.json` or `reference/` by hand; re-pin only from
  a certified run (`FASTLOOP_MODE=pin FASTLOOP_ARGS=--allow-repin`) and
  record the jobid in CLAIMS.md.
- Deck rebuild (only if the deck itself must change):
  `FASTLOOP_REBUILD=1 sbatch fastloop/build_minideck.sbatch` — reruns
  the QE leg, re-pins, re-checks, all in one job.

## Problem

A semantic check used to need container + frozen bundle + multi-node
sbatch: 30-120 minutes per iteration. Agents therefore reasoned instead
of testing, which is how runtime-only defects ship — both of this week's
shipped defects (scorecard BB.4; the f96c180-bundle fix) were invisible
to every AST gate. The fix is a checked-in miniature deck small enough
that the FULL L1 driver chain runs in ONE process in about a minute,
with sharding semantics still exercised via
`XLA_FLAGS=--xla_force_host_platform_device_count=4` (2x2 mesh of host
devices, no MPI) — the repo `tools/probe_w_densifier_hlo.py` trick,
scaled from one L2 routine to the whole chain. Every driver resolves its
mesh through `common.collectives.resolve_mesh`, which takes the
most-square factorisation of `jax.devices()`, so 4 host devices give
every driver a real 2x2 mesh with no code changes.

## What exists (this session)

- `deck_mini/` — the checked-in miniature system. MoS2 primitive cell
  (3 atoms), 2x2x1 k-grid, 40 NSCF bands, sigma window nval=26 + ncond=4
  (mirrors the b300 contract: nval = nelec so the eqp -> htransform
  override covers every interpolated band), FFT grid 24x24x80 (inherited:
  the certified QE leg reuses the ONE converged SCF density, which pins
  cell + cutoff; the deck is "mini" in k-points, bands, centroids and
  sigma window, not in the r-grid). Built by `build_minideck.sbatch`
  using the job-7884642 recipe (`qe_deck_b300.sbatch`), shrunk.
- `run_fastloop.py` — the runner. Chain per leg:
  kmeans -> dipole -> kin-ion -> gw -> eqp-convert -> htransform(dft)
  -> htransform(qp), each driver a fresh subprocess with the certified
  single-process env (`JAX_PROCESS_COUNT=1` pinned — the job-7884642
  SLURM_NTASKS lesson). Legs: `p1` (1 device) and `shard4` (4 host
  devices, 2x2 mesh). Modes: `pin` (write `reference/` +
  `pins_mini.json` from a certified run) and `check` (compare every
  stage output against the pins; exit nonzero on ANY drift or stage
  failure). Parity: full-file numeric compare for .dat/centroids
  (NaN-aware, `compare_valsmoke.py` lineage), strided point-wise
  sample + max-abs signatures for .h5, VBM/CBM/gap scalars from both
  bandstructure files, and the ht-qp "Using EQP energies" log gate
  (the post_b300 90/91 gate). Exit codes: 0 pass | 1 drift | 2 stage
  failure | 3 refusal.
- `run_fastloop.sbatch` — standard form: tiny 1-node dev job; also runs
  as `bash run_fastloop.sbatch` inside any existing allocation. Refuses
  login nodes (jax cannot import there: glibc 2.17 vs wheel 2.28, repo
  `docs/environment/overview.md` layer 2 — re-verified this session:
  the venv interpreter is container-internal `/usr/local/bin/python3.12`).
- `build_minideck.sbatch` — one-time deck build + first certification
  (QE leg, deck assembly, `--mode pin`, then `--mode check` on both legs
  in the same job: rerun determinism and device invariance certify the
  instrument itself before it gates anyone).
- `pins_mini.json` + `reference/` — pinned values, tolerances, and
  reference outputs. Written by pin mode only; never by hand.

## Rules

- The mini-deck proves plumbing and sharding semantics, NOT physics
  (4 conduction bands cannot converge screening): no physics conclusion
  may cite a fastloop number.
- Tolerance defaults are the `compare_valsmoke.py` precedent (1e-8 eV;
  5-6 orders above the measured FFI-swap noise floor of 2.5e-14 eV) and
  are recalibrated only from measured deltas, recorded in the pins.
- The runner reads the LIVE src tree by design (`--src`): it is a
  pre-commit semantic check of the working tree. Production jobs keep
  reading frozen bundles (INVARIANTS row 5 is about jobs whose results
  are records; a fastloop FAILURE is a record, a pass is just a gate).
- Warm compile cache is the default (outputs measured byte-identical
  warm: jobs 7884869/7884871); anything HLO/collective-shaped must be
  cache-cold (CLAIMS row 5) — `--hlo-diag` does that itself.
- Any speedup/coverage claim from here goes through CLAIMS.md with
  jobids, like every other claim.

## Certified (2026-07-31)

- Deck build (QE leg): job 7884926 — NSCF 3 s / pw2bgw / wfn2hdf green;
  WFN_mini.h5 = 40 bands, 4 IBZ k, nspinor 2, 10.3 MB; deck 12 MB.
- Pin run: job 7884936, src @ 24e4dc3. Stage walls (s, semi-cold):
  kmeans 26.0, dipole 16.4, kin_ion 6.3, gw 40.0, eqp 0.1, ht_dft 7.7,
  ht_qp 6.4 — leg 102.8. kmeans: 400 requested -> 394 kept, rank 37,
  gate 37/37 PASS (reproduced identically in 4 further runs, both legs).
- Check run (same job): p1 leg 76.6 s warm, ALL deltas exact-0; shard4
  leg 107.5 s, all exact-0 except sigma_mnk.h5 at 1.15e-13 eV. The
  pinned tolerances (1e-8 eV files / 1e-6 eV scalars, x10 on shard4)
  therefore carry >=5 orders of margin over measured noise.
- Two real defect classes caught DURING certification (both invisible to
  AST gates, the fastloop's reason to exist — CLAIMS rows 18-19):
  slab_io=auto aborting in MPI_Init on a bare launch, and a gw_jax
  interpreter-teardown hang at P=1.
- The pinned scalar triple per bandstructure file is (E_vbm_raw,
  E_cbm_raw, difference) of window bands 25/26 in the file's own units —
  a drift detector, NOT a physics gap claim (see Rules).

## What remains (honest list)

1. `run_fastloop.sbatch` has not yet run as its OWN sbatch job (the
   certification job invoked the identical runner + env through
   `build_minideck.sbatch`; the wrapper adds only module load + apptainer
   exec + the login guard, and is bash -n checked). First standalone use
   should confirm rc=0 and add the jobid here.
2. HLO forbid-gate as a hard failure: `--hlo-diag` only summarizes
   (cache-cold dump + `tools/hlo/analyze_hlo_dump.py`). It cannot be a
   hard `--forbid all-gather` gate yet because known-open gathers exist
   in the chain (CLAIMS row 16 zeta-apply; KNOWN_LORRAX_ISSUES htransform
   SVD family). When those close, scope the gate per stage: the invariant
   is "no gather-class collective on an N_mu^2-class operand", and a
   stage consuming a volume-preserving staged reshard legitimately emits
   exactly 2 shard-sized `all-to-all`s (see `docs/HLO_HOWTO.md`).
3. Wire into `skills/checkpoint/SKILL.md` as the mandatory pre-commit
   semantic check (certification now holds; wiring is a docs change the
   next checkpoint session should make).
4. BSE/absorption drivers are NOT in the chain (kmeans/dipole/kin-ion/
   gw/htransform only). Extending needs a decision on which BSE outputs
   to pin.
5. Two repo-side findings to fix in src (REPORTED, not patched — sandbox
   discipline): (a) `slab_io=auto` resolves to PHDF5_FFI whenever the
   host .so exports the handler and then dies in `MPI_Init_thread` when
   no PMI environment exists (bare single-process launch); auto should
   probe MPI bootstrapability, or demote with an announcement, the way
   the GPU router degrades (job 7884926). (b) `gw.gw_jax` completes
   `main()` and then hangs indefinitely in interpreter teardown on a
   bare P=1 launch (>=4 min observed, job 7884928); the fastloop works
   around it with an `os._exit` wrapper (`GW_WRAPPER` in
   `run_fastloop.py`) — suspects: FFI/MKL or XLA:CPU client teardown
   (gw is the only chain driver exercising the FFT/GEMM FFI paths).
6. Superseded: the old `run_minideck.py` scaffold (in-process stage
   stubs, synthetic-deck plan) was replaced by this runner; the
   downsample-vs-synthesize question resolved itself — the certified QE
   leg regenerates a real tiny NSCF instead, preserving every code path
   (spinor structure, symmetry maps, finite-q head) with zero synthetic
   code.
