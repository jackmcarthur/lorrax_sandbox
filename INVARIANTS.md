# INVARIANTS.md — cross-cutting preconditions

Conditions that couple physics choices to machine behavior. Violating one
produces a wrong-but-plausible result or a distant failure, not a local
error. Each row names the enforcing refusal where one exists; where the
column says "convention", nothing stops you — check it yourself.

| # | Invariant | Why | Enforced by |
|---|---|---|---|
| 1 | The centroid-selection band window must span the sigma band window | Otherwise the ISDF overlap spectrum is rank-capped and every pseudo-inverse silently loses rank | `src/common/rank_criterion.py` (kappa_cap criterion; refusal names the band-window cause) |
| 2 | `warm_mesh_cliques(mesh)` runs before the first jitted mpi collective, warming x + y + world cliques | XLA:CPU issues collective thunks from pool workers; communicator creation off the main thread refuses (CLAIMS row 8) | `src/common/collectives.py::warm_mesh_cliques`; refusal is the MPI thread-main error |
| 3 | `dipole.h5` is regenerated on any band-window change | Stale dipole matrix elements are consistent in shape but wrong in content | Convention (SIZE campaign brief, `lorrax_setup/wk_REL/`) |
| 4 | Collective/HLO tables are produced cache-cold (`ISDF_JAX_CACHE_DIR=""`) | Cache-hit modules never re-dump HLO; warm tables report "no violation" falsely | Convention (CLAIMS row 5) |
| 5 | Jobs read a frozen source bundle, never the live tree | Mid-job edits and half-synced trees produce unattributable results | `config/frontera/build_cpu_runtime_bundle.sh` + sbatch template |
| 6 | No N_mu^2-class tile materializes on one rank | The scaling doctrine; a single gather of W-class operands caps the reachable system size | AST gates (repo `tests/`), HLO probes (`docs/HLO_HOWTO.md`), CLAIMS rows 2, 9 |
| 7 | `.h5` written in-container (HDF5 1.14) is not opened with login tools (HDF5 1.8) | 1.8 CLI tools fail on 1.14 superblocks; the error looks like corruption | Convention (AGENTS.md rule 5) |
| 8 | apptainer binds include `/opt/intel`, never `/dev` | MKL/IMPI resolution needs the former; binding `/dev` breaks the container | Convention (certified sbatch template) |
| 9 | A perf candidate states its scaling over the design envelope (natoms to hundreds, N_mu to tens of thousands, P to thousands, both backends) before implementation | Deck-tuned wins invert at scale; the canonical rejected example is DFT-as-matmul for nk=16 | Owner rule, 2026-07-28 |
