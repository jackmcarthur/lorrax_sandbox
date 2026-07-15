# src/bse/bse_jax.py — deep-read notes (626 LOC)

Audit date: 2026-07-15, lorrax_D checkout. Stated base e18d0e5
(agent/slate-linalg-ffi); working HEAD at audit time was 218aeb8
(agent/ppm-fit-conditioning) — `git diff e18d0e5 218aeb8 -- src/bse/` is
empty, so the notes are valid for both.

## Purpose

The BSE **CLI entry point and solver dispatcher**: `python -m bse.bse_jax`
routes to FEAST (default), KPM DOS, or the Lanczos/Davidson preview
(`_preview_lanczos`), the latter being the production route in
STATUS.md / BGW_COMPARE.md ("--bse --lanczos --tda"). It also acts as the
**package facade** (`__all__` re-exports the serial matvec, ring matvec,
Lanczos solvers, and eigenvector writer) and carries a **legacy jit'd
sharded matvec trio** (`apply_bse_hamiltonian`/`apply_V`/`apply_W`) that is
dead and non-functional (see Suspects).

Physics implemented (TDA, Q=0, per-element, exactly as written; X is the
exciton amplitude batch `X[b,c,v,k]`, all energies in Ry):

```
H·X = D·X + V·X − W·X                                    (line 86)

D:   HX_D[b,c,v,k] = (eps_c[k,c] − eps_v[k,v]) · X[b,c,v,k]
     (energy_diff_cv_k = eps_c.T[:,None,:] − eps_v.T[None,:,:], bse_preconditioner.py:29-42)

pair amplitude (spin-traced per vertex):
     M[k,c,v,m] = Σ_s conj(ψ_c[k,c,s,m]) · ψ_v[k,v,s,m]   (line 95)

V (bare exchange, q=0), einsums verbatim (lines 108-121):
     S[b,N,k]  = Σ_{c,v} M_Y[k,c,v,N] · X[b,c,v,k] / √nk        ("kcvN,bcvk->bNk", psum over "x")
     U[b,M,k]  = Σ_N V_q0[M,N] · S[b,N,k]                        ("MN,bNk->bMk",  psum over "y")
     VX[b,c,v,k] = Σ_M conj(M_X[k,c,v,M]) · U[b,M,k] / √nk       ("kcvM,bMk->bcvk", psum_scatter "x" on c)
  ⇒ net: VX[b,c,v,k] = (1/nk) Σ_{M,N} M*_X[k,c,v,M] V_q0[M,N] Σ_{c',v'} M_Y[k,c',v',N] X[b,c',v',k]
     — NOTE: k appears on BOTH sides of every einsum ⇒ the exchange as
     implemented is DIAGONAL in k (couples (c'v'k)→(cvk) at the same k only).
     See Suspects: the reference formula has Σ_{k'}.

W (screened direct, FFT convolution over k−k'), lines 142-160:
     R[b,c,k,s,N]      = Σ_v conj(ψ_v_Y[k,v,s,N]) · X[b,c,v,k]        ("kv sN,bcvk->bcksN")
     T[b,M,N,t,s,k']   = Σ_c ψ_c_X[k',c,t,M] · R[b,c,k',s,N]          ("kctM,bcksN->bMNtsk", psum "x")
     U[b,M,N,t,s,k]    = (1/√nk) Σ_{k'} W_q[M,N,k−k'] · T[…,k']       (ortho ifftn/pointwise/fftn over 3 k axes)
     A[b,c,N,s,k]      = Σ_{M,t} conj(ψ_c_X[k,c,t,M]) · U[b,M,N,t,s,k] ("kctM,bMNtsk->bcNsk")
     WX[b,c,v,k]       = Σ_{N,s} ψ_v_Y[k,v,s,N] · A[b,c,N,s,k] / √nk  ("kvsN,bcNsk->bcvk", psum "y", psum_scatter "x")
  ⇒ net: WX[b,c,v,k] = (1/nk) Σ_{k'} Σ_{M,N,t,s} ψ*_c[k,c,t,M] ψ_v[k,v,s,N] W[M,N,k−k'] ·
                         Σ_{c',v'} ψ_c[k',c',t,M] ψ*_v[k',v',s,N] X[b,c',v',k']
     — matches Henneke 2020 Eq. (4-6) including the Σ_{k'} (the ortho
     ifft·ifft·fft chain yields (1/√N)Σ_{k'} W_{k−k'} T_{k'}; the trailing
     /√nk completes 1/nk).
```

Spin convention: `H = D + V − W` with V coefficient **1** (spinor
convention; the restricted-singlet textbook form is D + 2V − W). Documented
in `src/bse/context/README.md:20`.

Category: **pipeline stage — BSE solver driver / CLI dispatcher** (plus a
dead in-file matvec that duplicates bse_serial/bse_ring_comm/bse_simple).

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox runs/skills/scripts)

| symbol | callers (grep evidence) |
|---|---|
| `python -m bse.bse_jax` (`__main__`, lines 349-626) | production CLI: `runs/Si/B_centroid_sweep_2026-04-27/run_bse_sweep.sh:46`; `runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/bse_profile*.out` (`run_module:bse.bse_jax`); recipes at `src/bse/STATUS.md:100`, `src/bse/BGW_COMPARE.md:58` |
| facade re-exports (`apply_bse_hamiltonian_single_device`, `solve_bse`, `compute_pair_amplitude`) | `src/bse/test_bse.py:34-38` (uses re-exported names, incl. the file-local `compute_pair_amplitude` at test_bse.py:226) |
| `apply_bse_hamiltonian` / `apply_V` / `apply_W` / `apply_D` (file-local jit trio) | **NONE FOUND** — `grep -rn "apply_bse_hamiltonian\b" src tests tools scripts` hits only bse_jax.py:46/68 and a pseudocode sketch in `context/README.md:130`; bse_ring_comm.py:25 imports the *bse_serial* `apply_D`/`apply_bse_hamiltonian_single_device`, not these |
| `_preview_lanczos` | internal only (bse_jax.py:609); referenced in comments at `davidson_absorption.py:90`, `test_davidson_bse.py:72` |
| `_main_random_demo` | internal only (`--debug-parallelism`, bse_jax.py:504) |

`src/bse/__init__.py` is empty (docstring only) — no package-level
re-export path into this module.

## Function table

### module top — lines 1-64
- Lines 10-12: `os.environ.setdefault("JAX_ENABLE_X64","1")` then
  `runtime.init_jax_distributed()` **at import time, before `import jax`**
  (comment: ring matvec psum/ppermute is silent-wrong without a shared
  distributed runtime). `init_jax_distributed` (src/runtime/__init__.py:109)
  is idempotent via the `_LORRAX_JAX_DISTRIBUTED_DONE` env sentinel and
  passes `local_device_ids` derived from `CUDA_VISIBLE_DEVICES` (one GPU
  per rank under Cray MPICH). Deliberate, documented.
- Lines 18-40: imports from `.bse_ring_comm`, `.bse_io` (twice — lines 27
  and 39, split import cruft), `.bse_serial`, `.bse_lanczos`,
  `.bse_preconditioner`. Imports private names `_find_restart_file`,
  `_load_ring_subset` across module boundary.
- Lines 44-64: `__all__` — 20 names; 5 are file-local
  (`apply_D/apply_V/apply_W/apply_bse_hamiltonian/compute_pair_amplitude`),
  the rest re-exports.

### `apply_bse_hamiltonian(X, nkx, nky, nkz, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_q, V_q0)` — lines 67-86
- `@jax.jit` with **no static_argnums**: `nkx/nky/nkz` are traced.
- Returns `apply_D + apply_V − apply_W`.
- **Cannot execute as decorated** (two independent trace-time failures):
  (a) `apply_V`/`apply_W` use `lax.psum(..., axis_name="x"/"y")` — unbound
  axis names under plain jit outside shard_map; (b) even inside a
  shard_map, `apply_W` line 146 does
  `T.reshape(..., nkx, nky, nkz)` on traced ints → ConcretizationTypeError.
- Zero callers (see entry-points table). Dead.

### `apply_D(X, eps_c, eps_v)` — lines 89-91
- `delta_E[c,v,k]·X[b,c,v,k]`, delta from
  `bse_preconditioner.energy_diff_cv_k` (returns `(nc,nv,nk)`).
- Byte-identical duplicate of `bse_serial.apply_D` (bse_serial.py:32-35),
  which is the one actually imported elsewhere (bse_ring_comm.py:25).
- Only caller: the dead `apply_bse_hamiltonian` above.

### `compute_pair_amplitude(psi_c, psi_v)` — lines 94-95
- `M[k,c,v,m] = Σ_s ψ*_c[k,c,s,m] ψ_v[k,v,s,m]` (`"kcsm,kvsm->kcvm"`).
- Duplicate of `bse_serial.compute_pair_amplitude` (bse_serial.py:27-29).
- Live caller **via this module**: test_bse.py:37/226 imports it from
  `.bse_jax`.

### `apply_V(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, V_q0, nk)` — lines 98-121
- Exchange matvec, per-element formula in Purpose. k-diagonal (see
  Suspects #1). Also structurally incompatible with the standard shardings:
  `M_Y` is built from band-replicated `psi_*_Y` (full nc, nv extents) while
  X's local block under `sh.X = P(None,"x","y",None)` has nc/px, nv/py —
  the `"kcvN,bcvk->bNk"` einsum would shape-mismatch inside shard_map for
  any px·py > 1. Abandoned pre-ring sketch; superseded by
  `bse_ring_comm.apply_V_ring` (ring over band blocks) and
  `bse_simple` (XLA auto-partitioned). Dead.

### `apply_W(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_q, nkx, nky, nkz)` — lines 124-160
- Direct-term matvec via FFT convolution (formula in Purpose; einsum
  `"kv sN,bcvk->bcksN"` at line 142 contains a stray space — einsum ignores
  whitespace, harmless). Same shard_map shape problem as `apply_V`
  (T einsum needs full-band X), plus the traced-reshape crash. Dead.

### `_main_random_demo()` — lines 163-200
- `--debug-parallelism` handler; despite the name it is **serial**: random
  (nk=8, nc=nv=4, nspinor=2, nμ=32) data through
  `apply_bse_hamiltonian_single_device` and `solve_bse` (n_eig=5,
  max_iter=30). Prints ⟨X|H|X⟩ and eigenvalues in Ry and eV
  (`ryd2ev = 13.6056980659`).

### `_preview_lanczos(input_file, n_val, n_cond, n_eig=5, write_eigs=None, max_lanczos_iter=None, include_W=True, eqp_file=None, n_occ=None, block_size=1, rtol=0.0, check_every=4, matvec_kind="ring", n_reorth=-1, solver_kind="lanczos")` — lines 203-345
- The production eigensolve driver (`--lanczos` path). Splits on
  `jax.device_count() > 1` (line 221-222):
- **Sharded branch (224-293)**: `create_mesh_2d()` (auto-factorizes device
  count; `--px/--py` NOT consulted) →
  `bse_io.load_bse_data_from_restart_sharded(restart, n_val, n_cond,
  mesh_xy, input_file, n_occ)` → plumb `data["matvec_kind"]` (string
  dispatch consumed at bse_lanczos.py:155) → optional EQP override
  (242-261): re-reads `enk_full` via inline `__import__('h5py')`, applies
  `bse_io.apply_eqp_corrections` (BGW eqp1.dat semantics),
  `resolve_n_occ`, re-slices `eps_v/eps_c` with
  `val_idx = arange(n_occ_eff−n_val, n_occ_eff)`,
  `cond_idx = arange(n_occ_eff, n_occ_eff+n_cond)` (LORRAX internal
  ordering: v=0 = deepest valence; the BGW valence flip happens only in the
  writer), re-pads with `_pad_axis_to_multiple` (eps_v→grid_y,
  eps_c→grid_x, matching the loader's c↔x / v↔y convention) →
  `max_lanczos_iter` default `max(30, min(200, bse_dim//2))`; block path
  divides by block_size; `n_reorth=-1` resolves to full reorth
  (= block_max_iter, line 282) → `solve_bse_sharded(...)` (3-tuple return).
- **Serial branch (294-328)**: `bse_io._load_ring_subset(restart, n_val,
  n_cond, 1, 1, eqp_file, n_occ, input_file)` (px=py=1 so band padding is a
  no-op) → `solve_bse(psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky,
  nkz, n_eig, max_iter, include_W)`. **Does not forward** block_size, rtol,
  n_reorth, matvec_kind, solver_kind (see Suspects #4).
- Prints eigenvalues in Ry and eV; `--write-eigs` →
  `bse_io.write_eigenvectors_stream("eigenvectors.h5", eigenvalues,
  eigenvectors, n_val, n_cond, nkx, nky, nkz, n_write)` — passes the
  **user-requested** n_val/n_cond, not the loader-clamped effective values
  (Suspects #6).

### `__main__` CLI — lines 349-626
- `args, _ = parser.parse_known_args()` (line 484) — unknown flags are
  silently discarded.
- Dispatch order: `--ring-test` → `ring_matvec_smoke_test()` (default 2×2
  mesh, args not passed); `--ring-check` →
  `ring_matvec_correctness_check(input, n_val, n_cond, px, py, components)`;
  `--debug-parallelism` → `_main_random_demo`; `--kpm-dos` →
  `bse_kpm.main(argv-list)`; **default (no `--lanczos`)** →
  `bse_feast.main(argv-list)`; `--lanczos` → TDA gate (line 606:
  `raise SystemExit("Lanczos preview currently supports TDA only...")`) →
  `_preview_lanczos`.
- Kernel selection: `use_rpa = args.rpa or not args.bse` (lines 515/542)
  and `include_W = not (args.rpa or not args.bse)` (line 616) — **the
  default kernel is RPA (D+V only); `--bse` must be passed explicitly to
  get D+V−W**.
- The FEAST/KPM sub-drivers are invoked by **rebuilding a string argv**
  (lines 516-536, 543-603) — every forwarded value round-trips through
  `str()`. Flags NOT in the forwarded lists (notably `--eqp`, `--n-occ`,
  `--write-eigs`, `--n-eig`, `--matvec-kind`, `--solver`) are silently
  dropped for those paths; `bse_feast.main` (bse_feast.py:1073-1146) and
  `bse_kpm.main` (bse_kpm.py:323-355) have no `--eqp`/`--n-occ` arguments
  at all.

## CLI flags consumed (argparse, lines 353-483)

| flag | meaning / default | consumed by |
|---|---|---|
| `-i/--input` | cohsex.in; used only to locate `tmp/isdf_tensors_*.h5` (bse_io.py:756-764), WFN path, head overrides | all data paths |
| `--n-val` / `--n-cond` | band window (4/4). SOC counting caveat: BGW_COMPARE §4 | all paths |
| `--px` / `--py` | process grid (1/1) | forwarded to FEAST/KPM; `--ring-check`; **ignored by the lanczos path** (`create_mesh_2d()` auto-factorizes) |
| `--n-eig` | eigenpair count (5) | lanczos path only |
| `--tda` | TDA (default False = full BSE) | required gate for lanczos; forwarded to FEAST/KPM |
| `--rpa` / `--bse` | kernel; **default RPA** (D+V), `--bse` → D+V−W; `--rpa` wins over `--bse` | `use_rpa`, `include_W` |
| `--lanczos` | route to `_preview_lanczos` instead of FEAST (False) | dispatch |
| `--write-eigs [N]` | write eigenvectors.h5 (const −1 = n_eig) | lanczos only |
| `--max-lanczos-iter` | total Krylov dim (auto `max(30,min(200,dim//2))`) | lanczos |
| `--block-size` | block Lanczos size (1) | sharded lanczos only |
| `--lanczos-rtol` / `--lanczos-check-every` | convergence-driven exit (0.0 = fixed / 4) | sharded lanczos only |
| `--n-reorth` | reorth window; −1 = full (help: essential for spinor BSE) | sharded lanczos only |
| `--matvec-kind` | ring / gather / simple (ring) | sharded lanczos only (dict plumbed, bse_lanczos.py:155) |
| `--gather-t` | deprecated alias for `--matvec-kind=gather` | line 622 |
| `--solver` | lanczos / davidson (lanczos) | sharded lanczos only |
| `--eqp` | BGW eqp1.dat QP corrections (None) | lanczos only; **silently dropped for FEAST/KPM** |
| `--n-occ` | occupied-band count (None → WFN ifmax) | lanczos only; **silently dropped for FEAST/KPM** |
| `--feast-n-lanczos/-buffer/-n-quad1/-n-quad2/-quadrature/-units-ev-per-ry/-ritz-count/-window1/-window2` | FEAST params | forwarded to `bse_feast.main` |
| `--gmres-max-iter/-tol/-seed/--gmres-fp32` | FEAST shifted solves | forwarded to FEAST |
| `--kpm-dos` | run KPM Chebyshev DOS and exit (False) | dispatch to `bse_kpm.main` |
| `--kpm-window-count/-n-moments/-n-random/-n-lanczos/-emin-ev/-emax-ev/-plot-file` | KPM params | forwarded to KPM and FEAST(--windows-kpm) |
| `--ring-test` / `--ring-check` / `--components` | ring matvec diagnostics | bse_ring_comm |
| `--ring-timing` | **parsed at line 481, read nowhere** (handler existed historically: `git log -S ring_timing` → 906dd31 etc.) | dead |
| `--debug-parallelism` | random-data serial demo | `_main_random_demo` |

Env: `JAX_ENABLE_X64` (setdefault "1", line 10);
`init_jax_distributed` reads SLURM vars, `CUDA_VISIBLE_DEVICES`,
`_LORRAX_JAX_DISTRIBUTED_DONE` at import. No cohsex.in keys are read here
(head overrides `vhead`/`whead` and `wfn_file` are parsed in bse_io).

## Sharding / PartitionSpec assumptions

Mesh: 2D `("x","y")` from `create_mesh_2d` (bse_ring_comm.py:31-43; px =
largest divisor ≤ √ndev). Canonical specs (`make_bse_shardings`,
bse_ring_comm.py:46-63): `X P(None,"x","y",None)` (c on x, v on y);
`psi_*_X P(None,None,None,"x")` (μ on x, bands replicated); `psi_*_Y`
(ν on y); `V_q0 P("x","y")`; `W_q P("x","y",None,None,None)`; eps
replicated. Band axes are zero-padded to multiples of the mesh dims
(loader lines bse_io.py:450-453; eqp branch bse_jax.py:260-261 repeats it
with the same c↔x, v↔y pairing). The q=0 head is injected at load time as
a rank-1 (μ,ν) update using dual-sharded `g0_X`/`g0_Y` copies
(bse_io.py:463-513, `gw.head_correction.apply_q0_head_rank1_sharded`).
The file-local `apply_V/apply_W` predate these conventions and are
inconsistent with them (band-replicated M vs band-sharded X).

## Host vs device residency

- Sharded path: `psi_full_y` is read from HDF5 **per-device slice**
  (`_read_psi_mu_sharded`) directly into the μ- and ν-sharded layouts —
  ψ is held in **two device copies** (X: μ-on-x; Y: ν-on-y). `W_q`
  `(μ,μ,nkx,nky,nkz)` and `V_q0` `(μ,μ)` device-sharded on (x,y).
  `W_R = ifft(W_q)` is precomputed once outside the Lanczos loop
  (bse_lanczos.py:167-181). No io_callback/host-cache pattern anywhere in
  BSE — everything is device-resident.
- EQP override re-reads `enk_full` on host (numpy) and re-uploads eps
  slices replicated (bse_jax.py:248-261).
- Serial path: `_load_ring_subset` loads everything with `jnp.asarray`
  (single device), including the full `V_qmunu` (nq,μ,μ).
- Eigenvectors are streamed to `eigenvectors.h5` one vector at a time
  (`jax.device_get(eigenvectors[i])`, bse_io.py:87-103).

## TDA vs full BSE

- This file's lanczos path is **TDA-only**, enforced at line 606-607
  (`SystemExit` unless `--tda`). The Hamiltonians here (dead local trio,
  `apply_bse_hamiltonian_single_device`, ring/simple matvecs) are the TDA
  block `A = D + V − W`.
- `--tda` default is False, i.e. the **default FEAST route is full
  non-TDA**; the full-BSE machinery (`build_bse_ring_matvec_full`, with
  the B-block `_encode_T_B`/`_apply_V_ring_B` couplings,
  bse_ring_comm.py:487-716) is only re-exported here and consumed by
  `bse_feast`/`bse_kpm`.

## Spin / nspinor

ψ arrays carry an explicit spinor axis `(nk, nb, nspinor, nμ)`. Pair
amplitudes are spin-traced per vertex (`Σ_s`); the W term keeps separate
spinor sums per vertex (t at the conduction vertex, s at the valence
vertex — the 2×2 "spin matrix" T of context/README.md). The kernel is the
**spinor convention** `D + V − W` (coefficient 1 on V); there is no
`nspin==1` branch anywhere in the BSE matvecs, so a scalar-relativistic
(nspinor=1) singlet calculation would be missing the factor 2 on V.
`write_eigenvectors_stream` stamps `ns=1`, `spin_kernel=3` and performs the
BGW valence-axis flip + Ry→eV conversion on write (bse_io.py:40-103,
LORRAX-internal v=0 = deepest valence; this flip is a BGW-compat
convention, not a bug — see STATUS.md "Index ordering").

## Coupling to gw/ and isdf/

- Input is the **gw_jax ISDF restart** `tmp/isdf_tensors_*.h5`
  (`psi_full_y`, `enk_full`, `V_qmunu`, `W0_qmunu`+`W0_ready` attr,
  `G0_mu_nu`, `vhead`, `whead`): BSE is entirely downstream of a prior
  `gw.gw_jax` pass; falls back from W0 to V (bare) silently
  (bse_io.py:376-379). ISDF centroids never appear directly — only via
  these tensors.
- `gw.head_correction.apply_q0_head_rank1(_sharded)` for the q=0 head
  (bse_io.py:504/818); BGW `vcoul`-matched heads per BGW_COMPARE §3.
- `common.fft_helpers.make_sharded_{i,}fftn_3d` shared with gw
  (bse_simple.py:39-42, bse_lanczos.py:178).
- `runtime.init_jax_distributed` — same bootstrap as gw_jax.
- `file_io.WfnLoader` for kgrid / cell_volume / ifmax resolution.
- `common/jax_compile_cache.py:4` lists `bse.bse_jax` among the cached-jit
  drivers.

## Suspects

### 1. BUG (high-value): exchange (V) term is diagonal in k — missing Σ_{k'}
Per-element, all three live matvecs and the serial reference compute
`S[b,N,k] = Σ_{c,v} M[k,c,v,N]·X[b,c,v,k]` (`"kcvN,bcvk->bNk"`:
bse_serial.py:62, bse_simple.py:101-104, bse_ring_comm.py:262-289, and the
dead bse_jax.py:108) and back-project **at the same k**, i.e.
`VX[cvk] = (1/nk) M*(k) V_q0 Σ_{c'v'} M(k) X(c'v'k)`.
The reference the code itself cites (context/README.md:41-48 → "Henneke
(2020)"; the paper is vendored at `context/Henneke-2020-...md`) defines the
exchange as a **full (k,k') matrix** (Eq. 2-16) and its matvec (Eq. 4-5)
has an explicit inner `Σ_{k'}`:
`[V_A X](i_v i_c k) = (1/N_k) Σ_{μν} ū_{i_c k}(r̂_μ)u_{i_v k}(r̂_μ) Ṽ_{A,μν} · (Σ_{k'} Σ_{j_c} u_{j_c k'}(r̂_ν) Σ_{j_v} ū_{j_v k'}(r̂_ν) X(j_v j_c k'))`.
The correct einsum chain would be `"kcvN,bcvk->bN"` then `"kcvM,bM->bcvk"`.
The transcription error is already present in the design note
(context/README.md:44 writes `S(ν,k)` with a per-k output) and propagated
to every implementation, so ring-vs-serial "correctness checks"
(`ring_matvec_correctness_check`, bse_ring_comm.py:960-962 uses the same
formula) cannot detect it. Why the BGW validation didn't catch it: in bulk
Si the exchange contribution to low-lying exciton **eigenvalues** is
sub-meV-to-few-meV (below the documented ~3 meV ISDF agreement floor,
STATUS.md "Results"), and spectra were compared at η = 0.15 eV with ~1.5%
peak tolerance. Systems with significant exchange (2D materials,
molecular crystals, singlet-triplet splittings) would get silently wrong
singlet energies. The W term is NOT affected (its Σ_{k'} is done by the
FFT convolution, matching Eq. 4-6).

### 2. DEAD + broken: file-local `apply_bse_hamiltonian`/`apply_V`/`apply_W`/`apply_D` (lines 67-160)
Zero callers by repo-wide grep of every mechanism (imports, `__init__`
re-exports — bse package `__init__` is empty, `-m` invocations in sandbox
runs/skills/scripts, string dispatch — the only string dispatch is
`data["matvec_kind"]` into bse_lanczos, tests). Doubly non-functional as
written: unbound `axis_name="x"/"y"` collectives under plain `@jax.jit`,
and `reshape(..., nkx, nky, nkz)` on traced ints (line 146) even if an
axis environment existed; also shape-incompatible with the canonical
band shardings (band-replicated `M_Y` einsum'd against band-sharded X).
Superseded by bse_ring_comm (ring/gather) and bse_simple (default).
`apply_D`/`compute_pair_amplitude` are byte-duplicates of bse_serial's
(violates the no-redundancy rule); `compute_pair_amplitude` is the only
one with a live consumer (test_bse.py:37 imports it via this module).

### 3. DEAD flag: `--ring-timing` (line 481)
Parsed, never read anywhere in the file or repo
(`grep -rn ring_timing src` → definition only). Handler existed in older
trees (`git log -S ring_timing` → 906dd31, a82d91b). Per the
"parsed-but-unread ≠ dead config" rule this was checked for lost wiring:
the timing capability now lives behind `common.timing` sections inside
`build_bse_ring_matvec` (bse_ring_comm.py:453-462), so the flag is
genuinely orphaned, not lost wiring.

### 4. BUG (footgun): single-device path silently drops solver flags
`_preview_lanczos` line 221-222 branches on `jax.device_count() > 1`; the
serial branch (325-328) calls `solve_bse` without `block_size`, `rtol`,
`n_reorth`, `matvec_kind`, or `solver_kind`. `solve_bse` then uses its
default `n_reorth=10` (bse_lanczos.py:45). The CLI's own `--n-reorth` help
(lines 438-445) says partial-reorth windows "give ghost eigenvalues that
destroy per-state oscillator strengths" for spinor BSE. Concrete failure:
the STATUS.md-recommended command with `--n-reorth -1 --solver davidson`
run on 1 GPU silently executes single-vector Lanczos with n_reorth=10 —
ghost eigenvalues, no warning.

### 5. BUG (silent physics drop): `--eqp`/`--n-occ` not forwarded to FEAST/KPM
The default path (no `--lanczos`) is FEAST (line 539); the KPM path is
`--kpm-dos`. Neither forwarded argv list (lines 516-536, 543-603) includes
`--eqp`/`--n-occ`, and `bse_feast.main` (bse_feast.py:1073-1146) /
`bse_kpm.main` (bse_kpm.py:323-355) define no such flags. Concrete
scenario: `python -m bse.bse_jax -i cohsex.in --eqp eqp.dat` (forgetting
`--lanczos`) runs FEAST on **uncorrected DFT energies** with no error —
BGW_COMPARE §2 quantifies the resulting silent offset at 50-200 meV.
Compounded by `parse_known_args` (line 484) swallowing typo'd flags.

### 6. BUG (edge, crash): user-arg vs loader-clamped band counts
Both loaders clamp `n_val/n_cond` to availability with a warning
(bse_io.py:426-427, 887-888), but (a) `--write-eigs` passes the
**unclamped** user values to `write_eigenvectors_stream`
(bse_jax.py:335-345), whose dataset is shaped `(1,n_write,nk,n_cond,
n_val,ns,2)` (bse_io.py:83) while the vectors have the clamped extents →
broadcast ValueError after the entire solve; (b) the sharded EQP branch
re-slices with unclamped `cond_idx = arange(n_occ_eff, n_occ_eff+n_cond)`
(bse_jax.py:255) → IndexError past nb_total, or eps/psi band-extent
mismatch downstream. Trigger: `--n-val 10` when only 8 valence bands
exist.

### 7. BUG (latent, cross-file): `_pad_axis_to_multiple` returns the PRE-pad size
`return jnp.pad(...), size` (bse_io.py:146) — the second element is the
original extent, yet callers bind it to `n_val_pad`/`n_cond_pad`
(bse_io.py:450-451, 899-900) and `solve_bse_sharded` sizes its trial
vectors from `data["n_cond_pad"]/["n_val_pad"]` (bse_lanczos.py:140-148).
Whenever n_val (n_cond) is not a multiple of grid_y (grid_x), psi/eps are
padded but the Lanczos vector keeps the unpadded extent → einsum shape
mismatch at the first matvec (e.g. `--n-val 5 --px 2 --py 2`). All
validated runs used multiples (4/8 bands on 2×2), so only the pad==0 case
is exercised. Secondary hazard if the shapes were fixed: bands are
zero-padded (ψ=0, ε=0), so padded pairs form an exact ΔE=0 null space
below the physical spectrum — Davidson's diagonal preconditioner /
`init_bse_subspace` would select those ghost slots first.

### 8. Convention (not bugs — document, don't "fix")
- **Default kernel is RPA and default solver is FEAST**: a bare
  `python -m bse.bse_jax -i X` runs FEAST with D+V only, full non-TDA.
  `--bse --lanczos --tda` is the validated production combination.
- Ry internally; eV only at print/write boundaries
  (`ryd2ev = 13.6056980659`, lines 191/329; writer converts, bse_io.py:41).
- Valence-axis flip and eigenvalue eV units in `eigenvectors.h5` are
  BGW-compat (STATUS.md "Index ordering"), applied in the writer only.
- `--px/--py` are ignored by the lanczos path (mesh auto-factorized);
  they only matter for FEAST/KPM/ring-check. The BGW_COMPARE recipe passes
  them anyway (harmless at 4 GPUs = 2×2).
- Import-time `init_jax_distributed()` before `import jax` is deliberate
  and required (comment lines 7-9); a refactor tripwire.

### 9. Redundancy / cruft
- Duplicate `.bse_io` imports (lines 27 and 39); `import numpy as _np` and
  `__import__('h5py')` inside the eqp branch (lines 244-248).
- The EQP branch duplicates loader logic ("Simpler: reload enk and
  re-apply slicing", comment lines 245-247) instead of the loader taking
  an `eqp_file` argument like `_load_ring_subset` already does
  (bse_io.py:773/799-800) — the serial path applies EQP inside the loader,
  the sharded path outside; two parallel EQP paths for one concept.
- `--gather-t` deprecated alias kept alongside `--matvec-kind`.
- `docs/architecture/codebase.md:115` still lists `bse_isdf.py`, which no
  longer exists (stale doc, noted in passing).
