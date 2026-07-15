# src/bse/bse_lanczos.py (313 LOC) + src/solvers/lanczos.py (582 LOC) — deep-read notes

Audit date: 2026-07-15, lorrax_D checkout. Stated base `agent/slate-linalg-ffi` @ e18d0e5;
working HEAD is adc2197 (`agent/ppm-fit-conditioning`). Verified both target files are
**byte-identical between e18d0e5 and HEAD** (`git diff e18d0e5..HEAD --stat -- src/bse/bse_lanczos.py
src/solvers/lanczos.py` → empty), so all line numbers hold at both commits.

## Purpose

Lanczos-family eigensolvers for the TDA BSE. `solvers/lanczos.py` is physics-blind
infrastructure: four Lanczos variants (Python-loop, fori_loop-jit, block-jit,
convergence-driven block while_loop) that find the lowest `n_eig` eigenvalues of a
Hermitian operator given only a matvec. `bse/bse_lanczos.py` wraps them with the BSE
physics: it builds the matvec from ISDF-basis wavefunctions/interactions and dispatches
single-device (`solve_bse`) or 2-D-mesh sharded (`solve_bse_sharded`, which also hosts
the Davidson dispatch) solves. Eigenvalues are exciton energies in **Ry** throughout
(LORRAX-internal convention; the Ry→eV ×13.6056980659 conversion happens only at the
`bse_io.write_eigenvectors_stream` writer and at CLI print, per STATUS.md "Index ordering").

Physics of the matvec (TDA Hamiltonian H = D + V − W, signs as written; the kernels live
in `bse_serial.py:56-80` for `solve_bse` and `bse_ring_comm.py`/`bse_simple.py` for
`solve_bse_sharded`, but this family owns the wiring). Per-element, with b=trial-vector
batch, c/v=conduction/valence band, k=k-point (flat, nk=nkx·nky·nkz), s,t=spinor,
μ,ν=ISDF centroid:

```
D:  (HX)_D[b,c,v,k] = (ε_c[k,c] − ε_v[k,v]) · X[b,c,v,k]

V (q=0 exchange, rank-structured in the centroid basis):
    M[k,c,v,μ]   = Σ_s conj(ψ_c[k,c,s,μ]) ψ_v[k,v,s,μ]        (compute_pair_amplitude)
    S[b,μ,k]     = Σ_{c,v} M[k,c,v,μ] X[b,c,v,k] / √Nk
    U[b,μ,k]     = Σ_ν V_q0[μ,ν] S[b,ν,k]
    (HX)_V[b,c,v,k] = Σ_μ conj(M[k,c,v,μ]) U[b,μ,k] / √Nk

W (direct term, k−k′ convolution done as FFT over the k-grid):
    R[b,c,k,s,ν]     = Σ_v conj(ψ_v[k,v,s,ν]) X[b,c,v,k]       ("kvsN,bcvk->bcksN")
    T[b,μ,ν,t,s,k]   = Σ_c ψ_c[k,c,t,μ] R[b,c,k,s,ν]           ("kctM,bcksN->bMNtsk")
    T_R = ifftₖ(T, ortho);  W_R = ifft_q(W_q, ortho)
    U_R[b,μ,ν,t,s,R] = W_R[μ,ν,R] · T_R[b,μ,ν,t,s,R]           (elementwise; W spin-blind)
    U   = fftₖ(U_R, ortho)
    A[b,c,ν,s,k]     = Σ_{μ,t} conj(ψ_c[k,c,t,μ]) U[b,μ,ν,t,s,k]
    (HX)_W[b,c,v,k]  = Σ_{ν,s} ψ_v[k,v,s,ν] A[b,c,ν,s,k] / √Nk

    HX = D + V − W          (include_W=False ⇒ RPA: HX = D + V)
```

The generic solvers (as written, `solvers/lanczos.py`):

```
lanczos_eig_jit, step j (lines 256-283):
    z   = H q_j
    α_j = Re⟨q_j, z⟩                       (jnp.vdot conjugates the first arg)
    z  ← z − α_j q_j − β_{j−1} q_{j−1}      (β_{−1} := 0 via jnp.where)
    for i ∈ [max(0, j−n_reorth), j):  z ← z − ⟨q_i, z⟩ q_i    (i=j masked out by valid=i<j)
    β_j = ‖z‖₂ ;  q_{j+1} = z / max(β_j, 1e−15)
    Q[:, min(j+1, M−1)] ← q_{j+1}           ← final-slot overwrite, see Suspects
    T = diag(α) + diag(β[:M−1], ±1);  eigh(T);  eigvec_i = Σ_j vecs_T[j,i] Q[:,j]

block_lanczos_eig_jit, step j (lines 392-421, column layout Q_j ∈ ℂ^{n×bs}):
    Z    = matvec(Q_jᵀ)ᵀ                                  (= H Q_j)
    α_j[a,b] = Σ_i conj(Q_j[i,a]) Z[i,b] = ⟨q_a, H q_b⟩
    Z   ← Z − Q_j α_j − Q_{j−1} β_{j−1}^H
    reorth window [max(0,j−n_reorth), j):  Z ← Z − Q_i (Q_i^H Z)
    QR:  Z = Q_{j+1} β_j    (β_j = R, upper-triangular)     — correct Galerkin blocks
    T[s+bs:s+2bs, s:s+bs] = β_j ;  T[s:s+bs, s+bs:s+2bs] = β_j^H   (_build_block_tridiag)
```

`block_lanczos_eig_jit_converged` = same step inside `lax.while_loop`; every
`check_every` iters it builds the partial T, pushes inactive diagonal slots out of the
spectrum with `+1e6` (lines 541-545), and exits when
`max_i |λ_i − λ_i^prev| / max(|λ_i|, atol) < rtol` (lines 546-550).

Category: **pipeline stage — BSE eigensolve (bse_lanczos) on top of generic solver infra
(solvers.lanczos)**.

## Entry points (grep over src/, tests/, tools/, scripts/, sandbox runs/skills/scripts, docs)

| symbol | callers (grep evidence) |
|---|---|
| `solve_bse` | `src/bse/bse_jax.py:195` (`--debug-parallelism` random demo), `bse_jax.py:325` (`_preview_lanczos` single-device branch), `src/bse/test_bse.py:245`; re-exported `bse_jax.py:37/62` |
| `solve_bse_sharded` | `src/bse/bse_jax.py:228` (import) / `:283` (call, multi-device branch of `_preview_lanczos`). Production: `runs/Si/B_centroid_sweep_2026-04-27/run_bse_sweep.sh:46` — `lxrun python3 -u -m bse.bse_jax -i cohsex.in --bse --tda --lanczos …`; recommended command `src/bse/STATUS.md:100-105` |
| `lanczos_eig_jit` | `bse_lanczos.py:87` (serial), `:273` (sharded bs=1 fixed-iter); re-exports `solvers/__init__.py:3`, `bse_jax.py:35/57` |
| `block_lanczos_eig_jit` | `bse_lanczos.py:300` (sharded bs>1 fixed-iter) — only caller |
| `block_lanczos_eig_jit_converged` | `bse_lanczos.py:267` (bs=1, rtol>0), `:293` (bs>1, rtol>0) — only callers |
| `simple_lanczos_eig` | `bse_lanczos.py:92` (`use_jit_lanczos=False`, reachable via `test_bse.py --no-jit-lanczos`); `solvers/dos.py:85,90` (`estimate_spectrum` spectral bounds → `compute_dos:147` → `solvers/pseudobands.py` GW pseudobands + BSE KPM/FEAST window bounds) |
| `block_lanczos_eig` | `bse_lanczos.py:83`, gated on `solve_bse(use_block=True)` — **no caller passes `use_block=True`** (repo-wide grep for `use_block`: only `test_bse.py:257` with `use_block=False` and the default `bse_lanczos.py:42`). Re-exported `solvers/__init__.py:19`, `bse_jax.py:34/52` but never invoked |
| `_build_block_tridiag` | internal only: `solvers/lanczos.py:427, 537, 569` |

Module invocation: `python -m bse.bse_jax --lanczos` (argparse in `bse_jax.py:349-626`);
in-module manual tests `python -m bse.test_bse`, `python -m bse.test_davidson_bse`.
No pytest coverage: `grep -rn "lanczos\|solve_bse" tests/` → empty (incl.
`tests/multi_device`, `tests/regression`).

## Function table

### `src/bse/bse_lanczos.py`

#### `solve_bse(psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz, n_eig=20, max_iter=100, use_block=False, block_size=4, use_jit_lanczos=True, n_reorth=10, include_W=True)` — lines 30–97
- Single-device driver. `shape=(nc,nv,nk)` from `psi_c (nk,nc,nspinor,nμ)`; flat dim
  `n_flat = nc·nv·nk`. Trial vectors have **no spinor axis** — spinor is contracted
  inside the matvec.
- `_matvec_impl` (54–60): fresh `jax.jit(static_argnames=("nkx","nky","nkz","include_W"))`
  **per call** of `solve_bse` → recompile every invocation (observed in production
  profile: `runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/profile_bse/compile_summary.json:638`
  "never seen function: solve_bse.<locals>._matvec_impl").
- `matvec_block` (76–80): `partial(lambda X: apply_bse_hamiltonian_single_device(...))` —
  a `partial` with no bound arguments wrapping a lambda; the block path is **unjitted**
  (uses `apply_bse_hamiltonian_single_device`, not the `_jit` twin at `bse_serial.py:83`),
  so W_R = ifft(W_q) is recomputed per iteration.
- Dispatch: `use_block` → `block_lanczos_eig` (buggy T, see Suspects); else
  `use_jit_lanczos` (default) → `lanczos_eig_jit`; else `simple_lanczos_eig`.
  `use_block` ignores `use_jit_lanczos` and `n_reorth`.

#### `solve_bse_sharded(data, mesh_xy, *, n_eig=20, max_iter=200, n_reorth=10, include_W=True, block_size=1, rtol=0.0, atol=1e-8, check_every=4, solver_kind="lanczos", davidson_n_random_init=5, davidson_eps_shift_Ry=1e-3)` — lines 100–313
- Multi-device driver over the `(x,y)` 2-D mesh. `data` dict contract from
  `bse_io.load_bse_data_from_restart_sharded` (`psi_{c,v}_{X,Y}`, `eps_c/v`, `W_q`,
  `V_q0`, `n_cond_pad`, `n_val_pad`, `nkx/y/z`); returns
  `(eigenvalues (n_eig,), eigenvectors (n_eig,1,nc_pad,nv_pad,nk), n_iter_done)`.
- Matvec selection (150–165): `data.get("matvec_kind", "ring")` — the flag is smuggled
  through the **data dict** (set at `bse_jax.py:239`), not a parameter. `"simple"` →
  `bse_simple.build_bse_simple_matvec`; else `bse_ring_comm.build_bse_ring_matvec`
  with `low_mem=(kind=="ring")` (`"gather"` = all_gather variant). All three share
  the signature `matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)`
  and expect **W_R already IFFT'd** (verified: `bse_simple.py:57-58` docstring,
  `bse_ring_comm.py:444` `_matvec_impl(..., W_R, ...)`).
- W_R hoist (167–181): `make_sharded_ifftn_3d(mesh_xy, sh.W.spec, sh.W.spec, axes=(2,3,4),
  norm='ortho')` — one 3-D cuFFT of W_q **once outside the Lanczos loop** (the dominant
  cost of the serial path is 200× per-iter FFTs, per docstring 122-124). FFT axes must be
  replicated (helper contract, `common/fft_helpers.py:314`), satisfied by
  `sh.W = P("x","y",None,None,None)`.
- Davidson branch (188–231): builds `apply_H` (matvec + `with_sharding_constraint(V, sh.X)`),
  diagonal preconditioner `bse_davidson_helpers.bse_diagonal_precond` (1/(E_c−E_v−λ+shift)),
  seeded subspace `init_bse_subspace` (lowest-ΔE unit vectors + `davidson_n_random_init`
  Gaussians), pre-compiles Ritz solves via `warmup_davidson_jit(m_max=4·n_eig)`, then
  `solvers.davidson.davidson(..., tol=atol if atol>0 else 1e-8)`. **Ignores** `rtol`,
  `check_every`, `n_reorth`, `block_size`; returns `n_iter_done = jnp.int32(max_iter)`
  (line 231) — a placeholder, not the real count. Returns numpy (davidson's device_get),
  vs jax arrays from the Lanczos path.
- Lanczos branch (233–304): one end-to-end `jax.jit` (`_full_run`) with explicit
  `in_shardings=(sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y, sh.eps, sh.eps, sh.W, sh.V)`,
  `out_shardings=(P(), P(), P())` (everything replicated on output), and
  `donate_argnums=(6,)` (W_q consumed building W_R). Four sub-paths:
  bs=1 fixed → `lanczos_eig_jit`; bs=1 rtol>0 → `block_lanczos_eig_jit_converged`
  with `block_size=1`; bs>1 fixed → `block_lanczos_eig_jit`; bs>1 rtol>0 → converged.
  Each matvec closure reshapes the flat Krylov vector to `(bs, nc_pad, nv_pad, nk)` and
  re-pins `sh.X = P(None,"x","y",None)` via `with_sharding_constraint` (lines 255, 284)
  so collectives inside the matvec stay on the mesh.

### `src/solvers/lanczos.py`

#### `block_lanczos_eig(matvec, shape, n_eig=20, block_size=4, max_iter=50, tol=1e-8, seed=42)` — lines 28–135
- Python-loop block Lanczos, row-vector layout (`Q (bs, n)`), full reorth against all
  previous blocks (91–93). Early exit on `‖β_j‖ < tol·block_size` (99–102).
- α_j[a,b] = Σ_i conj(Q[a,i]) Z[b,i] = ⟨q_a, H q_b⟩ (line 83) — correct.
- β_j := Rᵀ from `qr(Z_flatᵀ) = Q̃R` (95–96); **T sub-blocks assembled transposed** —
  see Suspects/bug #1. Three-term subtraction (88–89) also misses a conj
  (subtracts Σ_a R_{j−1}[b,a] q_a^prev instead of Σ_a conj(R_{j−1}[b,a]) q_a^prev),
  masked in exact arithmetic by the full-reorth loop that follows.
- No overwrite issue: Q blocks live in a Python list; `Q_all = concat(Q_blocks[:n_blocks])`
  (128) takes exactly the blocks that T describes.

#### `simple_lanczos_eig(matvec, n, n_eig=20, max_iter=100, seed=42)` — lines 138–211
- Python-loop scalar Lanczos, full reorth (185–187), breakdown exit at β<1e−12 (190–192).
- Correct slot bookkeeping: `Q` allocated `(n, max_iter+1)` (172), q_{j+1} stored at
  column j+1 (195), Ritz vectors from `Q[:, :max_iter]` (206). The reference against
  which the jit variants' final-slot overwrite (bug #2) is judged an oversight.

#### `lanczos_eig_jit(matvec, n, n_eig=20, max_iter=100, seed=42, n_reorth=2)` — lines 214–300
- `lax.fori_loop` scalar Lanczos, partial reorth window `n_reorth` (268–275; `valid=i<j`
  masks self-projection). β guarded by `max(β,1e−15)` (280). Fixed iteration count; T
  uses all `max_iter` α/β with no breakdown masking (288–290) — if the Krylov space
  exhausts early, post-breakdown junk pollutes T (fixed-iter limitation, distinct from
  the STATUS.md "ghost eigenvalues without full reorth" note).
- **Final-slot overwrite** at 281 (bug #2 below). Eigenvectors 296–298.

#### `_build_block_tridiag(alpha_all, beta_all, max_iter, bs)` — lines 303–327
- In-jit `fori_loop` assembly of block-tridiagonal T from `(M,bs,bs)` stacks;
  sub-block = β_j, super = β_j^H, final `(T+T^H)/2` (327). O(1) trace-time HLO.

#### `block_lanczos_eig_jit(matvec, n, n_eig=20, block_size=4, max_iter=50, seed=42, n_reorth=2)` — lines 330–438
- fori_loop block Lanczos, column layout, β_j = R directly from `qr(Z)` (417) —
  **correct** Galerkin blocks (contrast with `block_lanczos_eig`). Pre-allocated
  `Q_all (M,n,bs)`, `alpha_all/beta_all (M,bs,bs)`. matvec contract: `(bs,n)→(bs,n)`,
  applied as `matvec(Q_jᵀ)ᵀ` (396). Same final-slot overwrite at 419–420 (bug #2).

#### `block_lanczos_eig_jit_converged(matvec, n, n_eig=20, block_size=4, max_iter=50, *, rtol=1e-6, atol=1e-8, check_every=4, min_iter=None, seed=42, n_reorth=2)` — lines 441–582
- `lax.while_loop` variant; carry = (j, Q_all, α, β, last_evals, converged).
  `min_iter = max(2·check_every, n_eig//bs + 1)` warmup (478–480). Convergence check
  under `lax.cond` every `check_every` iters (530–560): partial T with `+1e6` diagonal
  mask on slots `pos ≥ (j+1)·bs` (541–545), `eigvalsh`, exit when
  `max|Δλ|/max(|λ|,atol) < rtol`. Note `atol` here is an **eigenvalue-scale floor**, not
  an absolute tolerance. Returns `(evals, evecs, j_final)`.
- Final eigh re-masks the same way (569–574). Residual quirk: β_{j_final−1} (written at
  the last completed iteration, 515) couples the active window to the first masked
  block, biasing Ritz values by O(‖β‖²/1e6) ≈ 1e−6 Ry — negligible but systematic.
- Same final-slot overwrite (516) when the loop runs all M iterations unconverged.

## Flags / CLI args / config consumed

All plumbed from `python -m bse.bse_jax` argparse (`bse_jax.py:349-626`) into this family
via `_preview_lanczos`; no LorraxConfig/`cohsex.in` keys are read by these two files.

| flag | meaning | default |
|---|---|---|
| `--lanczos` | route to `_preview_lanczos` (else FEAST) | off |
| `--tda` | required gate: `bse_jax.py:606-607` raises `SystemExit` for the Lanczos path without it | off |
| `--bse` / `--rpa` | `include_W = not (args.rpa or not args.bse)` (`bse_jax.py:616`) — **default is RPA** (no W term) | RPA |
| `--n-eig` | eigenvalues to return | 5 |
| `--max-lanczos-iter` | total Krylov dimension cap (block path divides by block_size, `bse_jax.py:273`) | auto `max(30, min(200, dim//2))` |
| `--block-size` | block-Lanczos block size (1 = scalar) | 1 |
| `--lanczos-rtol` | >0 enables while_loop convergence-driven variant | 0.0 (fixed iters) |
| `--lanczos-check-every` | convergence-check cadence (block iters) | 4 |
| `--n-reorth` | reorth window; −1 → full reorth resolved to `block_max_iter` (`bse_jax.py:282`); STATUS.md says full is required for spinor BSE | −1 |
| `--matvec-kind` | ring / gather / simple — reaches `solve_bse_sharded` via `data["matvec_kind"]` (`bse_jax.py:239` → `bse_lanczos.py:155`) | ring |
| `--gather-t` | deprecated alias for `--matvec-kind=gather` (`bse_jax.py:622`) | off |
| `--solver` | lanczos / davidson → `solver_kind` | lanczos |
| `--n-val --n-cond --n-occ --eqp -i --write-eigs` | problem setup / eqp override / eigvec write, handled in `_preview_lanczos` | — |
| kwargs with no CLI exposure | `atol` (1e−8, doubles as Davidson residual tol), `davidson_n_random_init` (5), `davidson_eps_shift_Ry` (1e−3), `solve_bse`'s `use_block`/`use_jit_lanczos` | defaults only |
| hardcoded | RNG seeds 42 (solvers) / 0 (Davidson init); `LARGE=1e6` mask; 1e−15 β floor; `tol=1e-8` in `block_lanczos_eig` | — |

Env: `JAX_ENABLE_X64=1` set by `bse_jax` import side effect (`bse_jax.py:10`); everything
is complex128/float64.

## Sharding / PartitionSpec assumptions (`solve_bse_sharded`)

From `bse_ring_comm.make_bse_shardings` (`bse_ring_comm.py:46-63`):

| array | shape | spec |
|---|---|---|
| trial X | `(bs, nc_pad, nv_pad, nk)` | `P(None,"x","y",None)` — c on x, v on y |
| `psi_c_X`, `psi_v_X` | `(nk, nb, nspinor, nμ_pad)` | `P(None,None,None,"x")` (μ on x) |
| `psi_c_Y`, `psi_v_Y` | same | `P(None,None,None,"y")` (ν on y) |
| `W_q` | `(nμ, nμ, nkx, nky, nkz)` | `P("x","y",None,None,None)` — FFT axes 2-4 **must be replicated** (make_sharded_ifftn_3d contract) |
| `V_q0` | `(nμ, nμ)` | `P("x","y")` |
| `eps_c/v` | `(nk, nb_pad)` | `P(None,None)` replicated |
| eigenvalues / eigenvectors / n_iter | out | `P()` fully replicated (line 233, 243) |

Band counts are padded to mesh multiples (`n_cond_pad` multiple of px, `n_val_pad` of py)
by the loader; `nc_pad·nv_pad·nk = n_flat` is the Krylov vector length. The flat Krylov
basis `Q (n_flat, max_iter)` / `Q_all (M, n_flat, bs)` inside `_full_run` carries **no
sharding constraint** — only the per-matvec reshape re-pins `sh.X`; XLA chooses the Q
layout, typically replicated per device (16·n_flat·max_iter bytes each) — fine at Si
4×4×4 8×8 (~13 MB) but a memory wall for larger runs (violates the zero-replicated-
intermediates principle).

## Host vs device residency

Everything is device-resident: the `data` dict arrays arrive device_put by
`load_bse_data_from_restart_sharded`; no io_callback/host staging in either file. `W_q`
is donated into `_full_run` (freed after W_R). Davidson returns host numpy
(`davidson()` does the device_get); the Lanczos paths return jax arrays — a return-type
asymmetry the caller absorbs.

## TDA vs full BSE

This family is **TDA-only**: the matvec is the Hermitian H = D + V − W and all solvers
assume Hermiticity (`α_j = jnp.vdot(q,z).real`, `eigh`). The CLI enforces it —
`bse_jax.py:606-607` exits with "Lanczos preview currently supports TDA only" unless
`--tda`. The full non-TDA operator S = [[A,B],[−B^H,−A^H]] exists only as
`bse_ring_comm.build_bse_ring_matvec_full` (`bse_ring_comm.py:490+`), consumed by the
FEAST path, never by this family. `include_W=False` degrades H to RPA (D+V).

## Spin / nspinor

The Krylov/trial vector is `(b, c, v, k)` — no spinor axis. Spinor lives in the ψ arrays
`(nk, nb, nspinor, nμ)` and is contracted inside the matvec: the V pair-amplitude sums
Σ_s conj(ψ_c)ψ_v; the W term keeps two spinor indices (t,s) through the T/U tensors and
contracts both on the back-projection (W is spin-blind, applied diagonally). Band counts
nc/nv already count spinor bands. The solvers are spin-blind; the only spin-aware policy
is `--n-reorth -1` (full reorth) being the documented default *because of* the highly
degenerate spinor BSE spectra (`bse_jax.py:439-445`, STATUS.md:15).

## Coupling to gw/ and isdf/

- No direct `gw.*` or `isdf.*` imports in either file. The physics arrives via
  `bse_io.load_bse_data_from_restart_sharded`, which reads `isdf_tensors_*.h5` — the
  GW/ISDF pipeline's restart bundle: `W0_qmunu`/`V_qmunu` in the ISDF centroid (μ,ν)
  basis and ψ(μ) interpolation vectors. So BSE quality is bounded by the upstream ISDF
  fit (STATUS.md attributes the ~3 meV eigenvalue floor and ~80% per-state cosine
  similarity to ISDF compression, with the no-ISDF-rank-excuse caveat applying to
  plateau-shaped disagreements).
- `common.fft_helpers.make_sharded_ifftn_3d` (bse_lanczos.py:17,178) — same shard_map FFT
  helper family used by gw flat-k kernels.
- `solvers.davidson` + `bse_davidson_helpers` for the Davidson branch.
- Reverse coupling: `solvers/lanczos.py:simple_lanczos_eig` serves `solvers/dos.py`
  spectral-bound estimation used by GW pseudobands (`solvers/pseudobands.py:38`) and the
  BSE KPM/FEAST window machinery — changes to it reach beyond BSE.

## Suspects

### Bugs

1. **`block_lanczos_eig` assembles transposed β blocks in T** (`solvers/lanczos.py:96,
   117-120`). Per-element: with `Z_flatᵀ = Q̃R` (95), the Galerkin matrix requires
   `T[(i+1)bs+a, i·bs+b] = ⟨q̃_a, z_b⟩ = R[a,b]`; the code stores `beta_j = R.T` (96) and
   sets `T[end:end+bs, start:end] = beta_j` → entry `R[b,a]`. Symmetrization (122) keeps
   the transposed value. Per-block transposition is not spectrum-preserving: with
   zero α and consecutive sub-blocks B=[[1,2],[0,3]], C=[[4,5],[0,6]],
   tr(T⁴) contains 4·tr(BBᵀCᵀC)=4·869 for the true T vs 4·tr(BᵀBCCᵀ)=4·629 for the
   code's T → different Ritz values for block_size ≥ 2 once ≥ 3 block iterations
   (and wrong eigenvector coefficients even at 2 blocks). The jit twin stores β = R
   directly (417) and is correct — evidence of oversight, not convention. Reachable
   only via `solve_bse(use_block=True)`, which nothing at HEAD invokes.
2. **Final Krylov slot overwritten in all three jit solvers.** `lanczos_eig_jit:281`
   `Q.at[:, min(j+1, M−1)].set(q_next)`: at j=M−1 this clobbers q_{M−1} with q_M, then
   `eigenvectors = (Q @ vecs_T[:, idx]).T` (296) multiplies the coefficient
   `vecs_T[M−1,i]` (which belongs to q_{M−1}) onto q_M — per-eigenvector error
   ≈ √2·|vecs_T[M−1,i]| in an off-basis direction; eigenvalues (from T) unaffected.
   Same in `block_lanczos_eig_jit:419-420` and, when unconverged at M,
   `block_lanczos_eig_jit_converged:516`. The non-jit twins allocate M+1 slots
   (`simple_lanczos_eig:172,195`) or a list (`block_lanczos_eig:105,128`) and are clean.
   Hits exactly the unconverged-Ritz regime STATUS.md flags for per-state oscillator
   strengths (eigvec route needs n→4096 at Si 8×8).
3. **Zero-padded band slots inject spurious low eigenvalues in `solve_bse_sharded`.**
   The loader pads ψ and ε with **zeros** to mesh multiples
   (`bse_io.py:449-453` via `_pad_axis_to_multiple`, `jnp.pad mode="constant"`;
   `bse_jax.py:260-261` repeats it for the eqp override). For a padded conduction slot
   c′: ψ_c[k,c′,·,·]=0 ⇒ M[k,c′,v,μ]=0 and T gets no c′ contribution ⇒ V and W terms
   vanish identically on that slot; D gives H X̂ = (0 − ε_v[k,v])·X̂ exactly, so every
   unit vector on (c′,v,k) is an exact eigenvector with eigenvalue −ε_v[k,v] (≈ 0 at the
   VBM for VBM-referenced enk, i.e. **below the physical exciton gap**; padded valence
   slots give +ε_c[k,c] inside the spectrum). Concrete failure: 4 GPUs (2×2 mesh),
   `--n-cond 7 --n-val 8` → nc_pad=8, and the lowest returned "excitons" are
   nk·n_val spurious modes at −ε_v. Davidson is worse: `init_bse_subspace` seeds the
   lowest-ΔE transitions, i.e. the padded ones. Latent because validated runs used
   mesh-divisible band counts (8×8 on 2×2). No mask anywhere in the matvec or in
   `solve_bse_sharded` (grep "pad" in bse_lanczos.py → only nc_pad/nv_pad dims).
4. **`--write-eigs` crashes on padded shapes.** `_preview_lanczos` passes the unpadded
   CLI `n_val, n_cond` to `write_eigenvectors_stream` (`bse_jax.py:333-345`), which
   creates `eigenvectors` dataset `(1, n_write, nk, n_cond, n_val, ns, 2)`
   (`bse_io.py:81-86`) and assigns per-state slabs of shape `(nk, nc_pad, nv_pad, 1)`
   (`bse_io.py:88-103`) — h5py shape-mismatch error whenever padding occurred (same
   trigger as #3), after the full solve has completed. The BGW valence-axis flip
   `vec[:, :, ::-1]` (bse_io.py:100) would additionally shift real bands under padding
   if shapes ever matched.

### Dead / near-dead
- `solve_bse(use_block=True)` path and therefore `block_lanczos_eig`'s only in-repo call
  site: no caller sets `use_block=True` at HEAD (grep above). The function is still
  re-exported from `solvers/__init__.py:19` and `bse_jax.py:52`. Given bug #1 it is
  also *wrong* — delete or fix; the block-jit twin supersedes it.
- `solve_bse`'s `simple_lanczos_eig` branch (91–95) reachable only through
  `test_bse.py --no-jit-lanczos`.

### Redundancy / cruft
- `matvec_block = partial(lambda X: ...)` (bse_lanczos.py:76-80): pointless `partial`
  around a no-arg-bound lambda; the block path also skips the available jitted serial
  kernel (`apply_bse_hamiltonian_single_device_jit`, bse_serial.py:83).
- `_matvec_impl` re-jitted per `solve_bse` call (54): production compile logs show the
  cache miss (`profile_bse/compile_summary.json:638`).
- Five overlapping solver variants for one job (simple / jit / block / block-jit /
  block-jit-converged) with three distinct T-assembly implementations; the two
  non-jit variants exist only as references/manual-test paths — classic parallel
  old/new paths per the sandbox cruft pattern.

### Conventions / weird
- `matvec_kind` plumbed through the **data dict** (`bse_lanczos.py:155` reads
  `data["matvec_kind"]`; `bse_jax.py:239` writes it) instead of a keyword —
  the only flag in the family that bypasses the signature.
- STATUS.md:16 declares `--matvec-kind=simple` "fastest, default", but the actual
  default is `"ring"` both in argparse (`bse_jax.py:449`) and in the dict fallback
  (`bse_lanczos.py:155`) — stale doc vs code.
- CLI default kernel is **RPA**: without `--bse`, `include_W=False`
  (`bse_jax.py:542,616`) — a silent no-W run for anyone omitting the flag; STATUS's
  recommended command always passes `--bse --tda`.
- Davidson branch ignores `rtol/check_every/n_reorth/block_size` and reports
  `n_iter_done = max_iter` (bse_lanczos.py:231) regardless of actual iterations;
  `atol` doubles as Davidson's residual tol but as an eigenvalue-scale floor in the
  converged Lanczos — one name, two semantics.
- `LARGE = 1e6` diagonal masking (solvers/lanczos.py:541, 570) leaks O(‖β‖²/1e6) Ry
  (~1e−6 Ry ≈ 14 μeV for ‖β‖~1) into checked and final Ritz values on early exit.
- No guard for `n_eig > Krylov dim`: `argsort(...)[:n_eig]` silently returns fewer
  eigenpairs and the downstream `reshape(n_eig, ...)` (bse_lanczos.py:90,312) raises.
- Eigenvalues Ry everywhere (LORRAX-internal); Ry→eV and the BGW valence-axis flip are
  writer-side (`bse_io.py:40-41,100`) — do not "fix" units or v-ordering here
  (BGW-compat lives at the file boundary, per STATUS.md "Index ordering").

### Test coverage
- Zero automated coverage: nothing under `tests/` imports either file. Only manual
  `__main__` harnesses (`src/bse/test_bse.py`, `src/bse/test_davidson_bse.py`) needing a
  restart h5. All four numbered bugs above would fit a 1-GPU (or CPU) synthetic-operator
  pytest — the block-T transpose and slot-overwrite need only a random Hermitian matrix.
