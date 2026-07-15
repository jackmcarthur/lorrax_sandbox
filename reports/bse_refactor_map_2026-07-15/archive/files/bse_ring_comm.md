# src/bse/bse_ring_comm.py — deep-read notes (996 LOC)

Audit date: 2026-07-15, lorrax_D checkout. Stated base e18d0e5
(agent/slate-linalg-ffi); working checkout was on agent/ppm-fit-conditioning
@ adc2197 — verified `git diff e18d0e5..HEAD -- src/bse/` is **empty** and
working tree clean for src/bse/, so all findings hold at the stated base.

## Purpose

Distributed (multi-GPU) BSE Hamiltonian matvec kernels on a 2D device mesh
`(x, y)`: conduction bands and ISDF centroid axis μ sharded on `x`, valence
bands and centroid axis ν sharded on `y`. Implements the TDA matvec
`HX = D + V − W` and the full (non-TDA) two-block matvec
`S = [[A, B], [−B†, −A†]]`, plus the shared sharding vocabulary
(`make_bse_shardings`) consumed by every other bse/ module, and two
density-space probe operators used by the W-exact/pseudopole tooling.
Communication is hand-rolled: `lax.ppermute` rings ("ring", low memory) or
`lax.all_gather` ("gather", faster) inside `shard_map`, with
`psum_scatter`/`psum` reductions.

Physics as written (all energies Ry, LORRAX-internal band order — v=0 is the
*deepest* valence band; the BGW valence flip happens only in
`bse_io.write_eigenvectors_stream`, see `src/bse/STATUS.md` "Index ordering"):

```
D:  (DX)[b,c,v,k]  = (ε_c[k,c] − ε_v[k,v]) · X[b,c,v,k]                       (l.330-333, 433-435)

V (q=0 exchange; NOTE: k-DIAGONAL as implemented — see Suspects):
    S[b,ν,k]  = Σ_{c,v,s} conj(ψ_c[k,c,s,ν]) ψ_v[k,v,s,ν] X[b,c,v,k] / √Nk    (l.269-270, 281, 287)
    U[b,μ,k]  = Σ_ν V_q0[μ,ν] S[b,ν,k]        (psum over y)                   (l.289-290)
    M[k,c,v,μ] = Σ_s conj(ψ_c[k,c,s,μ]) ψ_v[k,v,s,μ]                          (l.296)
    (VX)[b,c,v,k] = Σ_μ conj(M[k,c,v,μ]) U[b,μ,k] / √Nk                       (l.297-300)

W (direct term, k−k′ convolution over the BZ):
    R[b,c,k,s,ν]   = Σ_v conj(ψ_v[k,v,s,ν]) X[b,c,v,k]                        (l.91  "kv sN,bcvk->bcksN")
    T[b,μ,ν,t,s,k] = Σ_c ψ_c[k,c,t,μ] R[b,c,k,s,ν]                            (l.121 "kctM,bcksN->bMNtsk")
    T_R = ifftnₒ(T_k over (kx,ky,kz));  U_R = W_R[μ,ν,R]·T_R;  U_q = fftnₒ    (l.416-419)
    A[b,c,ν,s,k]   = Σ_{t,μ} conj(ψ_c[k,c,t,μ]) U[b,μ,ν,t,s,k]                (l.422 "kctM,bMNtsk->bcNsk")
    (WX)[b,c,v,k]  = Σ_{s,ν} ψ_v[k,v,s,ν] A[b,c,ν,s,k] / √Nk                  (l.423-424)

    With norm='ortho' on both transforms and the final /√Nk, the net W action
    is (WX)(k) = (1/Nk) Σ_{k′} [conj(ψ_c)ψ_c′](μ) W_{μν}(k−k′) [conj(ψ_v′)ψ_v](ν) X(k′).

TDA:   HX = D + V − W                                                          (l.337, 450)
Full:  X_out = A·X + B·Y ;  Y_out = −B(X) − A(Y)    (assumes A† = A, B† = B)   (l.662-670)
       B's V uses conj'd ket pair: apply_V_ring(X, conj(ψ_c_Y), conj(ψ_v_Y), ψ_c_X, ψ_v_X, V_q0)  (l.572-583)
```

Sign/conjugation convention: the V path computes bra = Φ_cvk = ψ_c conj(ψ_v)
and ket = conj(Φ) (i.e. the transpose-conjugate of the textbook
`conj(Φ)·V·Φ` form). Equivalent for Hermitian `V_q0`; acknowledged in
`bse_simple.py:95` ("Original forward (apply_V_ring) is conj(ψ_c)·ψ_v·X").
Not a bug.

Category: **physics: distributed BSE-Hamiltonian matvec kernels (solver-agnostic
core of the BSE stage) + variant axis (ring/gather/simple, TDA/full)**.

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox runs/skills/scripts)

| symbol | callers (grep evidence) |
|---|---|
| `create_mesh_2d` | `bse_jax.py:229`, `absorption_haydock.py:162`, `davidson_absorption.py:76`, `test_davidson_bse.py:62` |
| `make_bse_shardings` | 10 modules: `bse_jax.py:23`, `bse_simple.py:67`, `bse_lanczos.py:139`, `bse_feast.py:501,733`, `bse_kpm.py:27`, `bse_pseudopoles.py:258`, `bse_w_exact.py:113`, `absorption_haydock.py:163`, `davidson_absorption.py:78`, `feast_zolo_sweep.py:120`, `feast_ellipse_mixed_sweep.py:95`, `test_davidson_bse.py:63` — the de-facto BSE sharding vocabulary |
| `build_bse_ring_matvec` | `bse_lanczos.py:162` (solve_bse_sharded, kind ring/gather), `absorption_haydock.py:206`, `bse_feast.py:486,734`, `bse_kpm.py:122`, `bse_w_exact.py:104`, `bse_pseudopoles.py:233`, `feast_zolo_sweep.py:121`, `feast_ellipse_mixed_sweep.py:96`; production XLA traces exist (runs/Si/04_si_4x4x4_bse/.../profile_sharded_v2/xla_dump/module_0039… references bse_ring_comm.py source_lines) |
| `build_bse_ring_matvec_full` | `bse_feast.py:494`, `bse_kpm.py:131`, `bse_w_exact.py:106`, `bse_pseudopoles.py:242` (all `use_tda=False` paths) |
| `apply_bse_hamiltonian_ring` | **no call sites anywhere** — only imported + `__all__`-re-exported (`bse_jax.py:19,47`; stale sandbox copy `scripts/bse_src_patches/bse_jax.py:12` same). Grepped src/tests/tools/scripts/docs + sandbox runs/skills/scripts |
| `apply_W_ring` | only `apply_bse_hamiltonian_ring` (l.335) → transitively dead |
| `apply_V_ring` | internal (l.334, 384, 562, 573) |
| `build_realspace_random_transition_generator` | `bse_w_exact.py:18,110` |
| `build_density_snapshot_operator` | `bse_w_exact.py:19,112`, `bse_pseudopoles.py:30,266` |
| `ring_matvec_smoke_test` | `bse_jax.py:487` (CLI `--ring-test`) |
| `ring_matvec_correctness_check` | `bse_jax.py:493` (CLI `--ring-check` [+ `--components`]) |

No pytest coverage: `tests/` has no bse tests; `src/bse/test_bse.py` is
single-device only (uses `apply_bse_hamiltonian_single_device`), and
`src/bse/test_davidson_bse.py` uses `bse_simple.build_bse_simple_matvec`.
The only exerciser of the ring kernels is the `--ring-check` CLI, which
compares against `bse_serial.apply_bse_hamiltonian_single_device`
(same formulas, so it validates *distribution*, not physics).

## Function table

### `create_mesh_2d(devices=None)` — lines 31–43
- px = largest integer ≤ √n dividing n_devices, py = n/px; `Mesh(dev.reshape(px,py), ("x","y"))`.
- n=8 → 2×4; n=1 → 1×1 (rings degenerate to self-permute, still correct).

### `make_bse_shardings(mesh_xy)` — lines 46–63
- SimpleNamespace of NamedShardings; the module-wide axis contract:
  `X` (b,c,v,k)=P(None,'x','y',None); `X_full` (2,b,c,v,k)=P(None,None,'x','y',None);
  `psi_x`/`psi_y` (k,band,spinor,rμ)=P(None,None,None,'x'/'y') — **band and k axes
  replicated, only the ISDF centroid axis sharded**; `V` (μ,ν)=P('x','y');
  `W` (μ,ν,qx,qy,qz)=P('x','y',None,None,None); `eps` (k,band)=replicated;
  `T`/`U` (b,μ,ν,t,s,k)=P(None,'x','y',None,None,None); `S` (b,ν,k)=P(None,'y',None);
  `U_mu` (b,μ,k)=P(None,'x',None); `d_mu` (b,μ)=P(None,'x'); plus `R`, `A`, `D`
  intermediates.

### `_ring_perm(axis_size)` — lines 66–67
- Ring permutation `((0,1),(1,2),…,(n−1,0))`.

### `_ring_sum_valence(X, psi_v_Y, v_chunk, py, nu_local)` — lines 70–96
- Ring over y contracting X's valence chunk against conj(ψ_v) at local ν:
  step i on y-rank j processes the X-chunk that originated at
  `origin=(j−i) mod py`, `dynamic_slice`-ing ψ_v's replicated band axis at
  `origin·v_chunk`. Einsum l.91 (note stray space in subscripts, harmless):
  `R[b,c,k,s,N] += Σ_v conj(ψ_v[k,v,s,N])·X[b,c,v,k]`.
- Assumes global padded n_val = py·v_chunk (loader's `_pad_axis_to_multiple`).

### `_ring_sum_conduction(R, psi_c_X, c_chunk, px, mu_local)` — lines 99–126
- Ring over x: `T[b,M,N,t,s,k] += Σ_c ψ_c[k,c,t,M]·R[b,c,k,s,N]` (l.121; ψ_c NOT
  conjugated — it is the ket-side pair member; t is the new ket-conduction
  spinor index, s the ket-valence spinor).

### `_ring_sum_conduction_first(X, psi_c_Y, c_chunk, px, nu_local)` — lines 129–157
- B-block analog, conduction summed first over the x ring:
  `R[b,v,k,s,N] += Σ_c conj(ψ_c[k,c,s,N])·X[b,c,v,k]` (l.152). Valid einsum.

### `_ring_sum_valence_second(R, psi_v_X, v_chunk, py, mu_local)` — lines 160–188
- **BROKEN**: einsum l.183 `"kvsM,bvksN->bMNtsk"` — output subscript `t`
  appears in NO input (`kvsM` binds ψ_v's spinor to `s`). numpy/jax einsum
  raises `ValueError: einstein sum subscripts string included output
  subscript 't' which never appeared in an input` (reproduced with numpy,
  this audit). Intended string is almost certainly `"kvtM,bvksN->bMNtsk"`
  (ψ_v ket spinor = new index t, mirroring `_ring_sum_conduction` l.121).

### `apply_W_ring(X, psi_c_X, psi_v_Y, W_R, nkx,nky,nkz, px, py)` — lines 191–225
- Monolithic W term inside one shard_map body: ring encode → plain
  `jnp.fft.ifftn/fftn` (local inside shard_map, fine) → ψ contractions →
  `psum_scatter(x, dim=1)` then `psum_scatter(y, dim=2)`. Only caller is the
  dead `apply_bse_hamiltonian_ring` — superseded by the builder's split
  encode_T + `_apply_W_from_T` pipeline.

### `apply_V_ring(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0, nk, px, py)` — lines 228–300
- q=0 exchange, formulas in Purpose. Structure: y-ring accumulates
  `A[b,c,N,k]` for the local c-chunk (ψ_c_Y band axis sliced at own
  x-position, l.253-257); x-ring all-reduces Σ_c (every rank ends with the
  full band sum, l.279-285); `U = psum_y(V_q0·S)`; back-projection through
  `conj(M_X)` at local μ with ψ_v sliced at own y-position (l.292-296);
  `psum_scatter(x, dim=1)` completes Σ_μ and re-scatters c.
- `nk_local` (l.248) is a misnomer: ψ arrays carry the FULL k axis
  (P(None,None,None,'y') leaves k replicated), so nk_local == nk.
- **k index is a batch index throughout** — see Suspects #3.

### `apply_bse_hamiltonian_ring(X, nkx,nky,nkz, ψ×4, eps_c, eps_v, W_R, V_q0, px, py)` — lines 303–337
- Old single-shard_map full-TDA matvec (D + apply_V_ring − apply_W_ring),
  with the only divisibility ValueError in the file (l.320-321). Uses
  `lax.axis_index` so it can only run inside shard_map — and nothing wraps
  it. **Dead** (see Entry points).

### `build_bse_ring_matvec(mesh_xy, nkx,nky,nkz, timed=False, low_mem=True, include_W=True)` — lines 340–484
- Factory for the production TDA matvec
  `matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)`.
  Note the 9-arg positional protocol takes **W_R (real-space)**: callers
  precompute `W_R = ifftn_ortho(W_q, axes=(2,3,4))` once outside the
  iteration loop (bse_lanczos.py:169-186, absorption_haydock.py:210-212,
  bse_feast.py:503-505 into `data["W_R"]`).
- Internals: `_encode_T` (ring, `low_mem=True`) vs `_encode_T_gather`
  (`lax.all_gather` on y then x, `low_mem=False`) both under shard_map with
  in_specs `(P(None,'x','y',None), P(None,None,None,'x'), P(None,None,None,'y'))`,
  out_spec `P(None,'x','y',None,None,None)`; `_apply_V_ring_only` shard_map;
  `_apply_W_from_T` **plain jit** (in_shardings sh.T/psi_x/psi_y/sh.W, out
  sh.X, `donate_argnums=(0,)` — T consumed); `_apply_D_term` jit.
- FFTs in `_apply_W_from_T` use `common.fft_helpers.make_sharded_ifftn_3d`/
  `make_sharded_fftn_3d` on the 8D spec P(None,'x','y',None×5) — comment
  l.394-402: plain jnp.fft on a sharded tensor makes XLA all-gather the whole
  array even with unsharded FFT axes (~5 s / 200 matvecs on Si 4×4×4,
  profile_sharded_v2/trace_summary.md).
- `timed=True` returns an UN-jitted variant with `timing.section` blocks
  (l.452-468) — **no caller passes timed=True anywhere** (grep).
- `include_W=False` → returns D + V only (RPA-style; but see Suspects #2/#3
  — the RPA V needs the Σ_k-coupled contraction that no longer exists).

### `build_bse_ring_matvec_full(...)` — lines 487–716
- Non-TDA factory: `matvec(X_full, ψ×4, eps_c, eps_v, W_R, V_q0)` with
  `X_full` (2,b,c,v,k) sharded sh.X_full; `X=X_full[0]`, `Y=X_full[1]`.
- Duplicates verbatim from the TDA builder: `_encode_T`/`_encode_T_gather`
  (501-529 ≡ 353-381), `_apply_V_ring_only` (561-570 ≡ 383-392), FFT factory
  + its 14-line comment (593-606 ≡ 394-407), `_apply_W_from_T` (608-630 ≡
  409-431), `_apply_D_term` (632-640 ≡ 433-441).
- B-specific: `_encode_T_B`/`_encode_T_B_gather` (531-559) — **both crash at
  trace time** via `_ring_sum_valence_second` l.183 / the same bad einsum
  inline at l.551. `apply_V_ring_B` (572-591) passes conj(ψ_c_Y), conj(ψ_v_Y)
  — consistent with the module's conjugation convention (ket pair for the
  deexcitation block).
- `_apply_B` (651-657): `V_B − W_B` — no D term in B, correct.
- `_matvec_impl` (659-670): `Y_out = −B(X) − A(Y)` justified by comment
  "A and B are Hermitian in this formulation" (l.666). True for A given
  Hermitian V_q0 and symmetrized W (bse_serial.symmetrize_W_q enforces
  W(q)=W(−q)†); for B this is an extra assumption — with SOC/complex ψ the
  coupling block is complex-symmetric, not Hermitian, in the standard
  formulation. Untestable today because the B W-path crashes (above); only
  RPA (include_W=False) full mode ever ran.

### `build_realspace_random_transition_generator(mesh_xy, nkx,nky,nkz, n_cond_pad, n_val_pad)` — lines 719–772
- shard_map `r(b,ν,k) → X(b,c,v,k)`: `U = psum_y(V_q0·r)`;
  `M_X[k,c,v,m] = Σ_s conj(ψ_c[k,c,s,m])ψ_v[k,v,s,m]` on own (c,v) chunks;
  `X = Σ_M conj(M_X)·U / √Nk` (l.744-765). Seeds density-space probes for
  the W-exact column solver (bse_w_exact.py:110). Validates divisibility
  explicitly (l.736-737).

### `build_density_snapshot_operator(mesh_xy, nkx,nky,nkz)` — lines 775–850
- shard_map `s(b,c,v,k) → d(b,μ) = Σ_k [V_q0 · Σ_{cv} M s / √Nk](μ,k)`.
  The y-ring/x-ring body (809-837) is a verbatim copy of apply_V_ring's
  step_y/step_x (259-287); diverges only after: `d_mu = Σ_k U[b,μ,k]` (l.842).
  Out spec P(None,'x').

### `ring_matvec_smoke_test(px=2, py=2)` — lines 853–899
- Random 2×2×1-k, nc=4px, nv=4py, nspinor=2 problem; builds matvec, prints
  `HX.sharding`. Requires px·py devices (error message suggests
  `XLA_FLAGS=--xla_force_host_platform_device_count`). Same PRNG key reused
  for real and imag parts (l.868-874) → ψ = a·(1+i), rank-deficient test
  data (cosmetic).

### `ring_matvec_correctness_check(input_file, n_val=4, n_cond=4, px=2, py=2, component_check=False)` — lines 901–994
- Loads a real restart subset via `bse_io._find_restart_file` +
  `_load_ring_subset`, computes reference
  `bse_serial.apply_bse_hamiltonian_single_device`, prints relative error of
  ring vs serial. `component_check=True` isolates D/V/W by zeroing the other
  inputs (`eps*0`, `W_R*0`, `V_q0*0`, l.979-983) and re-derives serial V/W
  by hand (958-973 — same k-diagonal V formula as bse_serial).
- l.975 rebuilds `comp_matvec = build_bse_ring_matvec(...)` although the
  identical `matvec` from l.948 is in scope → duplicate compile (minor).

## Flags / CLI args / env consumed

None read directly in this file (no argparse, no LorraxConfig, no os.environ).
Reached through callers:

- `bse_jax --ring-test` → `ring_matvec_smoke_test()` (bse_jax.py:487).
- `bse_jax --ring-check [-i FILE] [--n-val N] [--n-cond N] [--px P] [--py P] [--components]`
  → `ring_matvec_correctness_check` (bse_jax.py:493).
- `--matvec-kind {ring,gather,simple}` (bse_jax/absorption_haydock CLIs) →
  `low_mem=(kind=='ring')` and ring-vs-simple dispatch
  (bse_lanczos.py:157-165, absorption_haydock.py:203-208).
- `--tda` / `--rpa` (bse_w_exact.py:101-106, bse_feast, bse_kpm,
  bse_pseudopoles) → TDA vs full builder; `include_W = not rpa`.
- `XLA_FLAGS=--xla_force_host_platform_device_count=N` — suggested by the
  smoke/check error paths for CPU testing (l.858, 913).
- Note: `bse_jax --ring-timing` (bse_jax.py:481) is parsed but its consumer
  `ring_matvec_timing` was deleted from this file in a0da0a5 — orphaned flag
  (bse_jax's problem, recorded here since the consumer lived here).

## Sharding / residency

- All large arrays are device-resident jax.Arrays; loading/sharding is done
  once by `bse_io.load_bse_data_from_restart_sharded` (ψ dual-sharded as
  psi_*_X and psi_*_Y — i.e. each ψ set is held TWICE, once per mesh axis;
  per-device footprint nk·nb_pad·ns·(n_rμ/px + n_rμ/py) complex128 each for
  c and v). No host caches / io_callback here.
- Band and k axes of ψ are fully replicated; only the centroid axis is
  sharded. Ring kernels rely on this to `dynamic_slice` remote bands locally.
- Divisibility contract: padded nc = px·c_chunk, nv = py·v_chunk, n_rμ
  divisible by px and py — enforced by the loader's `_pad_axis_to_multiple`
  (and only double-checked in the dead `apply_bse_hamiltonian_ring` and the
  transition generator).
- W_q/W_R q-axes and T k-axes are replicated → local FFTs via
  `make_sharded_*fftn_3d` (common/fft_helpers, shared with gw/).

## TDA vs full-BSE

- `build_bse_ring_matvec` = TDA (resonant block A only).
- `build_bse_ring_matvec_full` = S=[[A,B],[−B†,−A†]] on stacked (2,…) vectors;
  B† and A† realized as B and A under a Hermiticity assumption (l.666).
- **Full-BSE with W has never been runnable** (Suspects #1); only the RPA
  (include_W=False) full path is exercisable, matching history: the only
  full-matvec test that ever existed was `tests/test_bse_ring_matvec_full_rpa.py`
  at ancestor commit 81ca040 (RPA-only), deleted since.

## Spin / nspinor

- A spinor axis `s`/`t` (size nspinor) is threaded through every kernel; the
  W term is spin-diagonal on each pair vertex (t on the c–c′ vertex, s on the
  v–v′ vertex, contracted diagonally l.121/422/423); V uses spin-summed
  pair densities (l.270/296). Works for nspinor∈{1,2}.
- No singlet ×2 factor on V anywhere: `A = D + V − W` is the **spinor (SOC)
  convention**. For an nspinor=1 spin-restricted singlet run BGW would use
  D + 2K^x − W; callers do not inject the 2 either (checked bse_lanczos,
  absorption_haydock). All validated comparisons were spinor Si (STATUS.md),
  so this is a latent convention gap for scalar-wfn runs, not a bug today.
- No collinear nspin=2 axis at all.

## Coupling to gw/ and isdf/

- Direct imports: `common.timing`, `common.fft_helpers.make_sharded_{i,}fftn_3d`
  (same custom-partitioned FFT helpers as gw/), `bse_io` (restart loading),
  `bse_serial` (reference kernels).
- No direct gw/ or isdf/ imports; the gw coupling is upstream via bse_io:
  the restart bundle is the cohsex/ISDF pipeline's `isdf_tensors_*.h5`
  (ψ at centroids, W_q/V_q0 in the ISDF product basis), and the q=0 head is
  injected at load with `gw.head_correction.apply_q0_head_rank1_sharded`
  (bse_io.py:504-507) — by the time arrays reach this module, heads are in.

## Suspects

### Bugs

1. **Full-BSE W path crashes at trace time — invalid einsum** (l.183 and
   l.551): `"kvsM,bvksN->bMNtsk"` has output subscript `t` in no input;
   numpy/jax raise ValueError (reproduced). Every call of the matvec built
   by `build_bse_ring_matvec_full(..., include_W=True)` (bse_feast.py:494,
   bse_kpm.py:131, bse_w_exact.py:106, bse_pseudopoles.py:242 with
   `use_tda=False`) dies in `_apply_B` → `encode_T_ring_B` (ring AND gather
   variants). Fix: `"kvtM,bvksN->bMNtsk"` (ψ_v ket spinor = t, mirroring the
   valid A-side l.121).

2. **Lost `v_couples_k` wiring — TypeError in kpm/pseudopoles callers**:
   `bse_kpm.py:120-137` and `bse_pseudopoles.py:231-248` pass
   `v_couples_k=bool(not include_W)` to both builders, whose signatures
   (l.340-348, 487-495) have no such parameter → immediate
   `TypeError: unexpected keyword argument`. The parameter existed in the
   ancestor `src/isdf/bse_isdf/bse_ring_comm.py` (commit 81ca040, l.364:
   `v_couples_k: bool = False` → `apply_V_ring(..., couple_k=...)`, which
   summed `S_total = Σ_k S[·,·,k]` — "Dyson-consistent RPA/Hartree
   contraction for q=0") and was dropped in the port to src/bse/, while the
   restored callers (fe5e3e8) still use it. Classic
   parsed-but-unread/lost-wiring: RPA mode *needs* the Σ_k-coupled V.

3. **V (exchange) term is k-diagonal — deviates from the module's own spec
   and from standard BSE**. Per-element, the code computes
   `(VX)[b,c,v,k] = (1/Nk) Σ_μν conj(M[k,c,v,μ]) V_q0[μ,ν] Σ_{c′v′} M̄[k,c′,v′,ν] X[b,c′,v′,k]`
   — k appears on BOTH sides of every einsum ("kcvN,bcvk->bNk" keeps k as a
   batch index; same in bse_serial.py:63-65 and bse_simple.py:99-102), i.e.
   K^x_{cvk,c′v′k′} ∝ δ_{kk′}. The Q=0 BSE exchange couples ALL (k,k′):
   K^x = (1/Nk)·M*_cvk(G) v(G) M_c′v′k′(G) (Rohlfing–Louie / BGW bsexmtxel
   computes all (ik,ikp) blocks), and this module's own context doc
   (`src/bse/context/tda_and_pseudopoles.md` §2.2) specifies
   `ρ[r_ν] = Σ_t R[t,r_ν] Z[t]` with t the FULL transition index (v,c,k).
   The ancestor's `couple_k=True` branch implemented exactly the missing
   Σ_k. Consequence: exchange → 0 as Nk grows (only 1/Nk of the physical
   operator survives). The STATUS.md Si validations (3 meV eigenvalues,
   1.5 % peak ε₂ vs BGW singlet absorption with exchange ON) cannot rule
   this out — bulk-Si low-exciton exchange is meV-scale under 0.15 eV
   broadening — but for 2D systems (MoS2/CrI3) or any singlet–triplet
   analysis the missing k-coupling is an O(exchange) error. Needs a
   dedicated gate (BGW `spin_triplet` A/B, or dense kernel diff) before the
   BSE program trusts singlet numbers.

### Dead

- `apply_bse_hamiltonian_ring` (l.303-337) + its only callee `apply_W_ring`
  (l.191-225): zero call sites at HEAD (only bse_jax import/__all__
  re-export); superseded by the builder pipeline. ~80 LOC.
- `timed=True` branches of both builders (l.452-468, 672-700): no caller
  passes timed=True (grep `timed=True` over src/bse → empty). ~50 LOC.
- Unused import `load_bse_data_from_restart_sharded` (l.24; only
  `_find_restart_file`/`_load_ring_subset` are used, l.909/919).
- Unused locals: `sh = make_bse_shardings(...)` in
  `build_realspace_random_transition_generator` (l.733) and
  `build_density_snapshot_operator` (l.786) — both use literal P(...) specs.

### Redundancy

- `build_bse_ring_matvec` vs `build_bse_ring_matvec_full` duplicate ~90 LOC
  verbatim (encode_T/gather, V wrapper, FFT factory incl. its 14-line
  comment, `_apply_W_from_T`, `_apply_D_term`) — the TDA builder is exactly
  the A-block subset of the full builder. Single-source-of-truth violation.
- `build_density_snapshot_operator._map` (l.809-837) copy-pastes
  apply_V_ring's y-ring/x-ring (l.259-287).
- Three coexisting matvec families (ring/gather here, "simple" in
  bse_simple.py) are an acknowledged variant axis (STATUS.md: simple is
  default/fastest; ring kept for memory-tight runs) — not cruft, but the
  ring/gather split inside THIS file doubles every encode definition.

### Weird

- The 9-positional-arg matvec protocol (X, ψ×4, eps×2, W_R, V_q0) is
  duplicated by every solver caller; W_R-vs-W_q is only distinguishable by
  reading the callers (violates the minimal-signatures preference; a bundle
  would help).
- `nk_local` naming in apply_V_ring (l.248) — it is the full nk.
- Smoke test reuses one PRNG key for re/im and for all arrays (l.867-874).
- `ring_matvec_correctness_check` l.975 rebuilds an identical matvec
  (duplicate compile).
