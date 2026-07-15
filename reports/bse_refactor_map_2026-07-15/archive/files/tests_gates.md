# src/bse/test_bse.py (382 LOC) + src/bse/test_davidson_bse.py (309 LOC) — deep-read notes + repo-wide BSE gate audit

Audit date: 2026-07-15, lorrax_D checkout. Working tree is at `adc2197`
(branch `agent/ppm-fit-conditioning`); the prescribed audit base `e18d0e5`
(`agent/slate-linalg-ffi`) differs from HEAD only in
`tests/test_ffi_linalg_contract.py` (`git diff e18d0e5..HEAD --stat -- src/bse/
tests/ pytest.ini pyproject.toml` → 1 file, 8 lines) — **no BSE or
pytest-config difference**, so these notes are valid for both commits.

## Purpose

Neither file is a pytest test. Both are **argparse CLI scripts** exercising
the TDA BSE matvec/eigensolver stack against a COHSEX/GW restart bundle
(`isdf_tensors_*.h5`):

- `test_bse.py` — single-device smoke/benchmark: loads a small
  (n_val×n_cond) band window, times the serial matvec
  (`bse_serial.apply_bse_hamiltonian_single_device`), runs single-vector
  Lanczos (`bse_lanczos.solve_bse`), prints eigenvalues and residuals,
  optionally writes `eigenvectors.h5`. **Stale**: reads a restart schema
  (`psi_l/psi_r/enk_l/enk_r`, 8-D `V_qmunu`) that the current writer no
  longer produces (see Suspects).
- `test_davidson_bse.py` — multi-device smoke + BGW comparison: loads the
  sharded restart (`bse_io.load_bse_data_from_restart_sharded`), builds the
  plain-jit sharded matvec (`bse_simple.build_bse_simple_matvec`), runs the
  shape-agnostic block Davidson (`solvers.davidson.davidson`), then compares
  the lowest `n_eig` eigenvalues (meV diff) and gauge-invariant per-state
  densities `|A^S[c,v,k]|²` (cosine similarity) against a BGW
  `eigenvectors.h5`. Docstring (lines 10-12): "smoke + numerical sanity
  test, NOT a unit test (**no asserts**)".

Physics exercised, per-element as written in the kernels these scripts call
(axis names: b=batch/trial, k=flat k index, c=cond band, v=val band,
s,t=spinor, M/μ N/ν=ISDF centroid; all energies Ry):

```
H_TDA X = D + V − W                       (bse_serial.py:80, bse_simple.py:175)

D[b,c,v,k]   = (ε_c[k,c] − ε_v[k,v]) · X[b,c,v,k]
                                          (energy_diff_cv_k, bse_preconditioner.py:41;
                                           bse_simple.py:82 identical: eps_c.T[None,:,None,:] − eps_v.T[None,None,:,:])

M[k,c,v,μ]   = Σ_s conj(ψ_c[k,c,s,μ]) · ψ_v[k,v,s,μ]     ("kcsm,kvsm->kcvm",
                                           bse_serial.py:29 == bse_jax.py:95 duplicate)

V (exchange, q=0 kernel), bse_serial.py:62-64:
  S_V[b,N,k]  = Σ_{c,v} M[k,c,v,N] · X[b,c,v,k] / √Nk    ("kcvN,bcvk->bNk")
  U_V[b,M,k]  = Σ_N V_q0[M,N] · S_V[b,N,k]               ("MN,bNk->bMk")
  V[b,c,v,k]  = Σ_M conj(M[k,c,v,M]) · U_V[b,M,k] / √Nk  ("kcvM,bMk->bcvk")
  ⇒ net: V[b,c,v,k] = (1/Nk) Σ_{M,N} M*[k,c,v,M] V_q0[M,N] Σ_{c',v'} M[k,c',v',N] X[b,c',v',k]
  NOTE: k appears in every output subscript → the exchange couples (c,v)↔(c',v')
  at the SAME k only. There is NO Σ_{k'}. See Suspects §k-diagonal exchange.

W (direct, screened), bse_serial.py:69-78:
  R[b,c,k,s,N] = Σ_v conj(ψ_v[k,v,s,N]) · X[b,c,v,k]     ("kvsN,bcvk->bcksN")
  T[b,M,N,t,s,k] = Σ_c ψ_c[k,c,t,M] · R[b,c,k,s,N]       ("kctM,bcksN->bMNtsk")
  T_R = ifftn(T over kx,ky,kz, norm='ortho');  W_R = ifftn(W_q over q axes, 'ortho')
  U_R[b,M,N,t,s,R] = W_R[M,N,R] · T_R[b,M,N,t,s,R]        (k-convolution: Σ_k' W(k−k') T(k'))
  U_q = fftn(U_R, 'ortho')
  A[b,c,N,s,k] = Σ_{M,t} conj(ψ_c[k,c,t,M]) · U[b,M,N,t,s,k]  ("kctM,bMNtsk->bcNsk")
  W[b,c,v,k]  = Σ_{N,s} ψ_v[k,v,s,N] · A[b,c,N,s,k] / √Nk     ("kvsN,bcNsk->bcvk")
```

`bse_simple.build_bse_simple_matvec` (what test_davidson_bse runs) is the
same index math with `with_sharding_constraint` tags instead of hand
collectives (`bse_simple.py:89-175`); the ring version
(`bse_ring_comm.apply_V_ring:228-301`) also matches (verified per-element:
A[b,c,N,k] = Σ_{v,s} conj(ψ_c)[k,c,s,N]·ψ_v[k,v,s,N]·X[b,c,v,k] accumulated
over the y-ring, then Σ_c, so S[b,N,k] = Σ_{c,v} M[k,c,v,N]X[b,c,v,k] —
identical k-diagonal exchange).

Category: **diagnostics / manual gates** (CLI smoke + BGW-comparison
scripts, not collected by pytest).

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox runs/skills/scripts)

| symbol | callers (grep evidence) |
|---|---|
| `bse.test_bse.main` | `python -m bse.test_bse` only. Documented in `src/bse/context/README.md:145,156,180` and own docstring lines 4, 11. Not in pytest testpaths. No hits in sandbox `skills/`, `scripts/`, `docs/`, `CHANGELOG.md`; `find runs -maxdepth 4 (*.sh|*.md|*.yaml) | xargs grep test_bse` → none. |
| `bse.test_davidson_bse.main` | `python -m bse.test_davidson_bse` only (own docstring line 15, `LORRAX_NGPU=4 lxrun ...`). Referenced as design mirror (comments, not imports) by `src/bse/davidson_absorption.py:9,91,121`. No run-script/skill hits (same greps as above). |
| `test_matvec` / `test_pair_amplitude` / `test_lanczos` (test_bse) | only `test_bse.main` lines 345-354. **Not pytest-collectable**: they take required positional args (`data`), would error "fixture 'data' not found" if ever collected. |
| `load_bse_data_from_restart` (test_bse:42) | only `test_bse.main:341`. Parallel/stale sibling of `bse_io.load_bse_data_from_restart_sharded`. |
| `find_restart_file` (test_bse:288) | only `test_bse.main:337`. Duplicates `bse_io._find_restart_file` (see Suspects). |
| `_load_data_and_matvec`, `_load_bgw_eigvecs`, `_cosine_similarity_density` (test_davidson_bse) | only `test_davidson_bse.main` (216, 278, 298). |

## Function table — test_bse.py

### `load_bse_data_from_restart(restart_file, n_val=4, n_cond=4, fermi_energy=0.0)` — lines 42-168
- Reads `V_qmunu` (line 63, expects 8-D `(1,1,1,nkx,nky,nkz,μ,ν)` via
  `shape[3:6]` at line 85), `W0_qmunu` if `attrs["W0_ready"]` (64-67),
  `psi_l/psi_r` (71-72), `enk_l/enk_r` (75-76). **All four ψ/enk datasets
  and the 8-D V layout are the pre-format-v2 restart schema** — current
  writer (`file_io/tagged_arrays.py:91-96`) emits only
  `V_qmunu` (flat-q) / `psi_full_y` / `enk_full`. See Suspects.
- Band selection: `mean_enk = mean over k`; valence = `mean_enk <
  fermi_energy` (default 0.0 → assumes VBM-referenced enk);
  `val_indices = argsort(val_energies)[-n_val:]` (line 120) — ascending, so
  selected v=0 is the deepest valence of the window = LORRAX-internal
  convention (STATUS.md "Index ordering" item 1).
- `V_q0 = V_qmunu[0,0,0,0,0,0,:,:]` (143) — flat-q index 0 must be q=0.
  `W_q = W_src[0,0,0].transpose(3,4,0,1,2)` → `(μ,ν,nkx,nky,nkz)` (149).
  Falls back to bare V as W when W0 absent (148) — RPA-with-bare-V
  placeholder, printed but not flagged in output data.
- Returns plain dict of **unsharded device arrays** (`jnp.asarray` of the
  full datasets — entire V_qmunu is materialized on one device, line 63).

### `test_matvec(data, n_warmup=2, n_bench=10)` — lines 171-215
- Random normalized complex trial `X (1,nc,nv,nk)` float64 pairs (181-183);
  warmup + benchmark loops around `apply_bse_hamiltonian_single_device`,
  wrapped in `timing.section` / `jax_profile.step_annotation`.
- Prints `<X|H|X>` (211-213). **No assert** (not even Hermiticity/realness).

### `test_pair_amplitude(data)` — lines 218-235
- Calls `compute_pair_amplitude` (the `bse_jax.py:94` duplicate, not the
  `bse_serial.py:27` one — identical einsum), prints shape and `|M|_max`.
  **No assert**.

### `test_lanczos(data, n_eig=5, max_iter=50, use_jit_lanczos=True)` — lines 238-285
- `solve_bse(..., use_block=False, use_jit_lanczos=..., n_reorth=10)`;
  signature verified against `bse_lanczos.solve_bse:30-46` (all kwargs
  exist). Prints eigenvalues in Ry and eV (RYD2EV=13.6056980659).
- Verification loop (273-283): for the first 3 states recomputes `<ψ|H|ψ>`
  and `|Hψ − Eψ|`, **prints only** — a residual of O(1) would pass silently.
- Note `n_reorth=10` — the `bse_jax` CLI default is `-1` (full reorth) with
  a warning that small windows "give ghost eigenvalues that destroy
  per-state oscillator strengths" (`bse_jax.py:437-445`). The test hardwires
  the ghost-prone setting.

### `find_restart_file(input_file)` — lines 288-306
- Candidates: `tmp/isdf_tensors_*.h5`, `isdf_tensors_*.h5`, then
  `tmp/taggedarrays600.h5`, `tmp/taggedarrays.h5`, `taggedarrays.h5`
  (296-300). Drifted duplicate of `bse_io._find_restart_file:756-765`,
  which accepts only `isdf_tensors_*.h5` and raises otherwise.

### `main(argv=None)` — lines 309-378
- Order: `timing.reset()` → device print → find restart → load →
  pair_amplitude → matvec → lanczos → optional
  `write_eigenvectors_h5` (357-371, kpts from `generate_kpts_grid` =
  unshifted `ix/nkx` MP grid, `write_eigenvectors.py:156-173`) →
  `timing.report`.

## Function table — test_davidson_bse.py

### `_load_data_and_matvec(input_file, n_val, n_cond, n_occ, eqp_file, include_W)` — lines 53-134
- `_find_restart_file` → `create_mesh_2d()` (all `jax.devices()`, factored
  px×py with px=⌊√n⌋ divisor, `bse_ring_comm.py:31-44`; 1 device → 1×1 mesh,
  so **1-GPU capable**) → `make_bse_shardings` → sharded loader with
  `n_occ` forwarding.
- EQP override (76-93): rereads `enk_full`, applies
  `bse_io.apply_eqp_corrections` (symmetry-mapped IBZ→full-BZ when
  `input_file` given, else nearest-energy match, `bse_io.py:698-753`;
  eV→Ry divide at 725/750), band-slices with `n_val_eff/n_cond_eff =
  data["n_val"/"n_cond"]` (the capped values — comment at 73-75 explains
  the einsum-shape failure this avoids), then pads eps_v to multiple of
  grid_y and eps_c to grid_x (90-91) matching the loader
  (`bse_io.py:450-453`).
  - `n_occ_eff = n_occ if n_occ is not None else int((mean_enk < 0.0).sum())`
    (line 85) — hand-rolled fallback, diverges from `resolve_n_occ` used by
    the two production copies of this block (`bse_jax._preview_lanczos:252`,
    `davidson_absorption.main:103`) and contradicts `BGW_COMPARE.md` rule 5
    ("No silent auto-detect"). Unreachable from the CLI anyway: `--n-occ`
    defaults to 4, not None (line 193).
- W precompute (106-113): `W_R = jit(make_sharded_ifftn_3d(mesh, sh.W.spec,
  sh.W.spec, axes=(2,3,4), 'ortho'))(W_q)`, block_until_ready to drop W_q.
  With `--no-W`, `W_R = data["W_q"]` unused by the matvec (`include_W=False`
  short-circuits at `bse_simple.py:133`).
- Returns `apply_H(X)` closure binding the 9-arg jit'd matvec (128-132).

### `_load_bgw_eigvecs(bgw_h5, n_eig, n_val, n_cond, nk)` — lines 141-162
- BGW h5 → numpy shape `(nq, N, nk, nc, nv, ns, 2)` (STATUS.md item 3);
  takes `[0, :n_eig]`, real+i·imag, drops ns via `[..., 0]`, **flips the
  valence axis** `[..., ::-1]` (line 158, BGW iv=1=highest ↔ LORRAX v=0=
  deepest — correct per STATUS.md item 1), transposes to `(n, nc, nv, nk)`.
  Returns eV eigenvalues (BGW convention) + `|A|²` density.
- Parameters `n_val`, `n_cond`, `nk` are **never used** in the body — cruft.

### `_cosine_similarity_density(lorrax_density, bgw_density)` — lines 165-180
- Flattens per-state densities over (c,v,k), normalizes, `L @ B.T`,
  `max(axis=1)` = best-match BGW state per LORRAX state. Gauge-safe
  (densities are U(1)-invariant; complex overlaps would not be — STATUS.md
  "Gauge-invariant comparison checks").
- Requires identical (nc,nv,nk) window on both sides; a mismatched BGW
  reference dies with a shape error in the matmul, not a diagnostic.

### `main()` — lines 187-305
- Flow: load+matvec → `init_bse_subspace` (lowest-ΔE unit vectors +
  `--n-random-init` Gaussian tail, `bse_davidson_helpers.py:61-138`) →
  `bse_diagonal_precond` (1/(ΔE−λ+1e-3 Ry), sharding = sh.X minus batch
  axis `P("x","y",None)`, lines 234-239) → `warmup_davidson_jit` →
  `davidson(apply_H, n_eig, precond_fn, X0=V0, m_max=4·n_eig, tol)` →
  eigenvalue diff vs BGW in meV (282-285) → density cosine table (287-303).
- Missing BGW file → prints "skipping comparison" and returns (274-276);
  again no assert anywhere, all pass/fail judgment is left to the reader.
- Host fetch: re-replicates eigvecs via `jit(lambda x: x,
  out_shardings=NamedSharding(mesh, P()))` (292-294), then reaches into the
  solver's private helper `from solvers.davidson import _to_host` (295) —
  private reach-in, same pattern class as the exemplar's
  `_make_cohsex_kernels` import.
- Density slice `[:, :n_cond, :n_val, :]` (296) uses the **user-requested**
  `args.n_val/args.n_cond`, not the capped `data["n_val"]/data["n_cond"]`
  used everywhere else in the file; when a request exceeds availability the
  numpy slice silently clamps to nc_pad/nv_pad and the comparison proceeds
  against a differently-shaped BGW density (matmul shape error) or, worse,
  against padded-zero rows when nc_pad > n_cond_eff (cosine computed over
  windows that include pad bands).

## Flags / CLI args / env consumed

| flag | file | meaning | default |
|---|---|---|---|
| `-i/--input` | both | cohsex.in used only to locate `tmp/isdf_tensors_*.h5` (+ WFN.h5 for kgrid/n_occ/cell_volume in the sharded loader) | required |
| `--n-val` / `--n-cond` | both | band window (Kramers pairs under SOC — BGW_COMPARE.md rule 4) | 4/4 (test_bse), 8/8 (davidson) |
| `--n-eig` | both | eigenvalues to converge | 10 / 20 |
| `--max-iter` | both | Lanczos / Davidson iteration cap | 50 / 60 |
| `--n-warmup`, `--n-bench` | test_bse | matvec benchmark loop counts | 2, 10 |
| `--no-jit-lanczos` | test_bse | `simple_lanczos_eig` python loop instead of `lanczos_eig_jit` | False |
| `--write-eigenvectors PATH` | test_bse | write via `write_eigenvectors.write_eigenvectors_h5` (NO valence flip — see Suspects) | None |
| `--n-occ` | davidson | occupied SP-state count, forwarded to loader and EQP slicing | 4 (Si-specific) |
| `--n-random-init` | davidson | random tail of the initial subspace | 5 |
| `--tol` | davidson | Davidson residual tolerance | 1e-6 |
| `--eqp PATH` | davidson | BGW eqp1.dat QP corrections (eV→Ry inside `apply_eqp_corrections`) | None |
| `--bgw-h5 PATH` | davidson | reference eigenvectors.h5; default is a **hardcoded /pscratch sandbox path** (lines 200-204) | Si 4×4×4 8×8 run |
| `--no-W` | davidson | drop the W (direct) term — "RPA-like, debug only" | False |
| env `JAX_ENABLE_X64` | test_bse | `os.environ.setdefault(...,"1")` line 22 + `jax.config.update` line 29 (davidson does only the config update, line 35) | on |
| env `ISDF_JAX_PROFILE_DIR` | test_bse | enables `common.jax_profile` tracing; echoed at line 333-334 | unset |
| env `LORRAX_NGPU` | davidson docstring | consumed by the `lxrun` wrapper, not by this code | — |

## Sharding / residency

- test_bse: **no mesh, no PartitionSpec** — everything single-device;
  entire `V_qmunu`/`psi_l`/`psi_r` datasets are `jnp.asarray`'d to the
  default device before band-slicing (lines 63-76), so device memory scales
  with the full restart, not the selected window.
- test_davidson_bse: canonical BSE shardings from
  `bse_ring_comm.make_bse_shardings:46-62` on the (x,y) mesh —
  `X P(None,'x','y',None)` (c on x, v on y), `psi_x P(None,None,None,'x')`
  / `psi_y P(...,'y')` (μ/ν sharded, bands replicated), `V P('x','y')`,
  `W P('x','y',None,None,None)`, `eps P(None,None)` (replicated),
  intermediates `S P(None,'y',None)`, `U_mu P(None,'x',None)`,
  `T P(None,'x','y',None,None,None)`. eps padding: v-axis to multiple of
  grid_y, c-axis to grid_x — must mirror `bse_io.py:450-453` and does
  (lines 90-91). Multi-process safety is delegated to the helpers
  (`process_allgather` in `bse_davidson_helpers._gather_to_host:41-58`,
  call-time eps forwarding in `bse_diagonal_precond:182-201`).
  Final eigvecs are re-replicated then host-fetched (292-296).

## TDA vs full BSE

Both scripts are **TDA-only**: the state vector is the resonant amplitude
`X[b,c,v,k]` and every matvec they can reach computes `D + V − W` on that
block only. The non-TDA machinery (`use_tda=False`, 2-component `X_full
P(None,None,'x','y',None)` sharding, J-metric) lives exclusively in the
FEAST/KPM/pseudopole paths (`bse_feast.py`, `bse_kpm.py`,
`bse_pseudopoles.py`) and is untouched (and ungated) here.
`write_eigenvectors_h5` hardcodes `use_tda=1` (`write_eigenvectors.py:130`).

## Spin / nspinor

- Exciton amplitude carries **no spin axis** — spinor pairs, `ns=1` on file
  (`write_eigenvectors.py:58,127`, `spin_kernel=3` line 79/125).
- V term traces spin inside the pair amplitude
  (`M = Σ_s ψ_c*·ψ_v`); W term keeps the full spinor pair `(t,s)` through
  encode/convolve/decode (T[b,M,N,t,s,k]) — correct spinor structure that
  degrades gracefully to nspinor=1. nspinor is read from `psi.shape[2]`,
  never assumed (test_bse:88, bse_serial:53).
- BGW-side: `_load_bgw_eigvecs` drops BGW's ns=1 axis (line 156). SOC band
  counting caveat (BGW SP bands vs LORRAX Kramers pairs, factor 2) is
  documented in BGW_COMPARE.md rule 4; `--n-occ 4` default in
  test_davidson_bse is the Si-SOC Kramers-pair count while the docstring
  usage example says `--n-occ 4` with `--n-val 8 --n-cond 8` — mixed
  conventions in one command line (works only post-`bse-band-slicing-fix`).

## Coupling to gw/ and isdf/

- No direct `gw.` or `isdf.` import in either file. Indirect:
  `bse_io.load_bse_data_from_restart_sharded` calls
  `gw.head_correction.apply_q0_head_rank1_sharded` (bse_io.py:504-507) for
  the q=0 head injection, and `file_io.WfnLoader` for kgrid/ifmax/
  cell_volume fallbacks (bse_io.py:403,497). The restart bundle itself
  (`isdf_tensors_*.h5`) is produced by the gw_jax ISDF pipeline
  (`file_io/tagged_arrays.write_restart_state_to_h5`).
- `solvers/`: `davidson`, `warmup_davidson_jit`, private `_to_host`
  (test_davidson_bse:38,295); `bse_lanczos` re-exports `solvers.lanczos`
  kernels for test_bse's path.
- `common/`: `timing`, `jax_profile` (test_bse:31-32),
  `fft_helpers.make_sharded_ifftn_3d` (test_davidson_bse:37).

---

# Gate audit (BSE test coverage repo-wide)

## What pytest actually collects

`pyproject.toml:59-65`: `testpaths = ["tests"]`, `norecursedirs =
["archive"]`, `addopts = "-m 'not extra'"`. There is no `pytest.ini`.
Consequently **`src/bse/test_bse.py` and `src/bse/test_davidson_bse.py` are
never collected** — they are reachable only as `python -m bse.<name>`. Had
they been collected, both would fail structurally (module-level `test_*`
functions with required positional args; test_bse would also fail at data
load, see Suspects).

Repo-wide grep of `tests/` for BSE imports (word-boundary; earlier
substring hits were `subset`/`bse_isdf`-archive false positives):

```
tests/test_eqp_bgw.py:78:  from bse.bse_io import read_bgw_eqp
```

That is the **entire pytest-collected BSE surface**: one round-trip test
(`test_reader_skips_provenance_header`) asserting
`write_bgw_eqp → read_bgw_eqp` allclose at 1e-9, plus byte-identity tests of
the gw-side writer. Pure numpy, no GPU marker, runs anywhere (1 GPU rule:
trivially satisfied). It is a real value gate — for exactly one 60-line
reader.

## Non-pytest gates (opt-in CLI, print-only)

| gate | where | asserts? | 1-GPU? |
|---|---|---|---|
| `python -m bse.bse_jax --ring-test` | `ring_matvec_smoke_test`, bse_ring_comm.py:853-899 | none — prints HX sharding only | needs px·py=4 devices, but message suggests `XLA_FLAGS=--xla_force_host_platform_device_count` (line 858) → 1-GPU/CPU capable |
| `python -m bse.bse_jax --ring-check [-‑components]` | `ring_matvec_correctness_check`, bse_ring_comm.py:901-996 | **prints** rel-err ring-vs-serial (and per-term D/V/W with `--components`), no threshold, exit 0 regardless | same as above; needs restart data |
| `python -m bse.bse_jax --debug-parallelism` | `_main_random_demo`, bse_jax.py:163-200 | none | yes (random data) |
| `python -m bse.test_davidson_bse` | this file | none (prints meV diff + cos-sim) | yes (1×1 mesh) but needs run artifacts + BGW h5 |
| `python -m bse.test_bse` | this file | none | broken (Suspects §1) |

`ring_matvec_correctness_check --components` is the closest thing to a real
matvec gate in the package (per-term serial-vs-ring rel-err on real restart
data) — it only needs a numeric threshold + assert + a tiny checked-in
fixture to become a pytest test.

## UNGATED modules (zero pytest coverage at HEAD)

Every `src/bse/*.py` except the `read_bgw_eqp` function of `bse_io.py`:

`bse_jax.py` (626 LOC driver + CLI), `bse_serial.py` (serial matvec — the
reference everything else is checked against, itself checked against
nothing), `bse_simple.py`, `bse_ring_comm.py` (996 LOC of hand-rolled
collectives), `bse_lanczos.py`, `bse_davidson_helpers.py`,
`bse_preconditioner.py`, `bse_io.py` (932 LOC: sharded readers, padding,
head injection, EQP application, eigenvector streaming — all unread by
tests except read_bgw_eqp), `bse_feast.py`, `bse_feast_dense_debug.py`,
`bse_kpm.py`, `bse_pseudopoles.py`, `pseudopoles_eval.py`,
`pseudopoles_sweep.py`, `feast_sweep.py`, `feast_ellipse_mixed_sweep.py`,
`feast_zolo_sweep.py`, `bse_w_exact.py`, `absorption_common.py`,
`absorption_eigvecs.py`, `absorption_haydock.py` (STATUS.md: "*the* method
to use vs BGW" — ungated), `davidson_absorption.py`, `eigvals_to_eps2.py`,
`write_eigenvectors.py`. Also BSE-adjacent `solvers/davidson.py` and
`solvers/lanczos.py` (tests/ imports only `solvers.projectors` and
`solvers.sternheimer_precond`, per `grep -rn "from solvers" tests`).

Nothing in the missing coverage inherently needs big hardware: serial-vs-
sharded matvec identity, Davidson/Lanczos vs `numpy.linalg.eigh` on a random
Hermitian TDA-shaped operator, `init_bse_subspace`/precond shape+sharding
checks, eigenvector-writer round-trip (incl. the valence flip), and
`apply_eqp_corrections` symmetry mapping are all 1-GPU-or-CPU-sized. The
gap is absence, not hardware.

## Suspects

### 1. test_bse.py cannot run against any current restart (stale schema) — bug/cruft
`load_bse_data_from_restart` reads `f['psi_l']/f['psi_r']/f['enk_l']/
f['enk_r']` (lines 71-76) and assumes 8-D `V_qmunu` via `shape[3:6]`
(line 85). The canonical writer (`file_io/tagged_arrays.py:91-96`,
`restart_format_version=2` at line 75) writes only flat-q `V_qmunu` +
`psi_full_y` + `enk_full`; repo-wide grep shows `'psi_l'` is read nowhere
else and written nowhere (`grep -rn "'psi_l'" src` → test_bse.py:71 only).
Concrete failure: `KeyError: 'psi_l'` on every `isdf_tensors_*.h5` produced
at HEAD; even given legacy ψ datasets, flat-q V_qmunu (3-D) makes
`V_qmunu.shape[3:6]` return an empty/short tuple → unpack ValueError.
History: deleted in cleanup `a0da0a5`, restored in `fe5e3e8` — restored
without rebasing onto the format-v2 loader. The sharded loader
(`bse_io.py:389-411`) has the compat shim; this file predates it.

### 2. k-diagonal exchange in every matvec these scripts exercise — bug-suspect (PLAUSIBLE, needs adjudication in the bse_serial/bse_simple notes)
Per-element (bse_serial.py:62-64, identical in bse_simple.py:101-131 and
bse_ring_comm.py:269-301):
`V[b,c,v,k] = (1/Nk) Σ_{M,N} M*[k,c,v,M] V_q0[M,N] Σ_{c',v'} M[k,c',v',N] X[b,c',v',k]`
— k is a batch index throughout ("kcvN,bcvk->bNk" keeps k in the output),
so the exchange couples pairs **at the same k only**. The ISDF-exact
exchange is `K^x = M† V_q0 M` dense over (cvk)×(c'v'k'):
`(K^x X)[cvk] = (1/Nk) Σ_{MN} M*_cvk[M] V_q0[M,N] Σ_{c'v'k'} M_c'v'k'[N] X[c'v'k']`
(pair density ψ_ck ψ_vk* is q=0-periodic for every k, so ⟨cvk|v|c'v'k'⟩ ≠ 0
for k≠k'). The implemented form zeroes all k≠k' exchange blocks; with the
1/Nk prefactor it equals the exact form only when both M[ν] and X are
k-independent. Not documented as an approximation anywhere (STATUS.md and
BGW_COMPARE.md silent; no comment at any of the three sites). Counter-
evidence to "badly wrong": `davidson_absorption.py:10` reports lowest-20
Si-8×8 eigenvalues within ~3 meV of BGW **with the default (exchange-
including) BGW kernel** (`runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8/
absorption.inp` has no `spin_triplet`) — but bulk-Si low excitons have
meV-scale exchange splittings, so 3 meV agreement cannot discriminate.
A material with strong exchange (e.g. MoS2, CrI3) would expose it.
No test gates this either way — the only correctness check in the package
(`--ring-check`) compares ring vs serial, i.e. two copies of the same
formula.

### 3. Two eigenvector writers with opposite valence conventions — bug (wrong output vs spec)
`bse_io.write_eigenvectors_stream` flips the valence axis on write
(`bse_io.py:98-100`, "v=0 at the deepest valence — flip on write") so its
`eigenvectors.h5` matches the BGW spec, as promised by STATUS.md
("Index ordering" item 1 and the closing sentence). The older
`write_eigenvectors.write_eigenvectors_h5` — the writer test_bse.py uses at
line 361 — performs **no flip** (verified: no `[::-1]`/flip anywhere in
write_eigenvectors.py; the transpose at line 96 reorders axes only) yet
writes the same `exciton_data/eigenvectors` dataset advertised as
"BerkeleyGW format" (docstring line 3) with `use_tda=1`. Concrete wrong
output: a consumer applying the STATUS.md convention (e.g.
`test_davidson_bse._load_bgw_eigvecs`, which flips at line 158) to a
test_bse-written file gets every state's valence character reversed
(v=0 ↔ v=n_val−1) — silent for degenerate |A|² only. Classic forbidden
parallel old/new path; the old writer also duplicates
`generate_kpts_grid` usage and eigenvalue-unit handling (writes Ry where
BGW files carry eV — second convention mismatch in the same file,
`write_eigenvectors.py:141` writes `eigenvalues` in Ry vs
`bse_io.write_eigenvectors_stream` which converts Ry→eV per STATUS.md).

### 4. Redundancy: three copies of the EQP-override band-slice block
`test_davidson_bse._load_data_and_matvec:76-93` ≈
`bse_jax._preview_lanczos:242-262` ≈ `davidson_absorption.main:95-112`,
with the test copy hand-rolling the n_occ fallback (`(mean_enk<0).sum()`,
line 85) instead of `resolve_n_occ` — drift already; the fallback is dead
from the CLI (`--n-occ` default 4). Likewise `test_bse.find_restart_file:
288-306` vs `bse_io._find_restart_file:756-765` (the test copy still
accepts `taggedarrays*.h5`, which the canonical loader was deliberately
narrowed to reject).

### 5. Cruft (small)
- `_load_bgw_eigvecs` unused params `n_val, n_cond, nk`
  (test_davidson_bse.py:141).
- Hardcoded default `--bgw-h5=/pscratch/sd/j/jackm/...` sandbox path inside
  shipped source (test_davidson_bse.py:200-204); ships a dead default to
  every non-Perlmutter user.
- Density slice with `args.n_val/args.n_cond` instead of the capped
  `data[...]` values (test_davidson_bse.py:288-296) — inconsistent with the
  file's own line-73 comment about exactly this trap.
- `compute_pair_amplitude` defined twice with identical bodies
  (bse_jax.py:94 and bse_serial.py:27); test_bse imports the bse_jax one.
- test_bse hardwires `n_reorth=10` (line 259), the ghost-eigenvalue-prone
  setting the production CLI defaults away from (bse_jax.py:437-445).

### Recommended minimal gate set (all 1-GPU/CPU)
1. Pytest-ify `--ring-check --components` with a tiny checked-in restart
   fixture + rel-err threshold (serial vs ring vs simple — three-way).
2. Random-Hermitian TDA-shape Davidson/Lanczos vs `numpy.linalg.eigh`
   (catches solver + helpers; no physics data needed).
3. Writer round-trip: `write_eigenvectors_stream` → `_load_bgw_eigvecs`
   convention check (valence flip + eV), and retire
   `write_eigenvectors.py` or fold the flip in.
4. An exchange-term k-structure test: dense-build `K^x = M†V M` for a
   4-band 2-k toy and compare against `apply_V` — adjudicates Suspect 2
   in milliseconds.
