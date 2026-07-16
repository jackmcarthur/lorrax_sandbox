# Batched trial-stack BSE matvec — design

BSE refactor-map, phase 2, 2026-07-16. Branch `agent/bse-phase2` off `sources/lorrax_A`
@ 6bd4dc9 (post-cleanup main). Owner spec: a jitted `matvec(trials[n_trials,c,v,k]) →
out[n_trials,c,v,k]` that materialises **one** trial's T-tensor `(μ,ν,t,s,k)` at a time
(donated), never the `n_trials` axis on T; RPA (D+V) | BSE (D+V−W) toggle; a single W-tile
seam for future W(ω)/ladders. Incorporates the settled B1 exchange fix (VERDICT.md: dense
in (k,k′), encode k-summed).

## Current state (file:line)

Four TDA matvecs (`bse_ring_comm.build_bse_ring_matvec` ring/gather, `bse_simple`, serial)
all carry the trial axis `b` on the direct-term tensor
`T[b,μ,ν,t,s,k]` (`sh.T = P(None,'x','y',None,None,None)`, ring_comm.py:56). Per device
`T` is `n_trials · μ_loc · ν_loc · ns² · nk` — the memory hog, linear in `n_trials`
(kernel_dataflow_trace.md boundary table, "the memory hog"). Consumers feed it a stack:
block-Lanczos `matvec_block(V_block.reshape(bs,c,v,k))` (bse_lanczos.py:282-289) and FEAST
`_rayleigh_ritz` which loops `for v in vectors: matvec(v,…)` one vector at a time
(bse_feast.py:337-350) — either replicating T over `b`, or paying a host dispatch per
vector. The exchange V-term keeps `k` as a batch index in `S[b,ν,k]` (ring_comm.py:239-252,
simple.py:101-106) — the B1 k-block-diagonal bug.

## Target math (per-element)

Stack `X[b,c,v,k]`, `b=0…n_trials−1`. `H X = D X + V X − W X` (BSE) or `D X + V X` (RPA).

**D** (batched, elementwise, local): `(DX)[b,c,v,k] = (ε_c[k,c] − ε_v[k,v]) X[b,c,v,k]`.

**V — dense exchange, B1-fixed** (batched, outside the scan; `S,U` are k-free ⇒ tiny):
```
M[k,c,v,ν] = Σ_s conj(ψ_c[k,c,s,ν]) ψ_v[k,v,s,ν]
S[b,ν]     = (1/√Nk) Σ_{k',c',v'} M[k',c',v',ν] X[b,c',v',k']      # k SUMMED (was batched)
U[b,μ]     = Σ_ν V_q0[μ,ν] S[b,ν]
(VX)[b,c,v,k] = (1/√Nk) Σ_μ conj(M[k,c,v,μ]) U[b,μ]                # broadcast over k
             = (1/Nk) Σ_{μν} M*_cvk(μ) V_0(μν) Σ_{k'c'v'} M_c'v'k'(ν) X[b,c'v'k']
```
The two `1/√Nk` compose to the `1/Nk` the dense formula requires (VERDICT.md Eq); the only
change vs current code is deleting `k` from `S`'s index list (transition-Hartree: build one
q=0 density summed over k′, one V_q0 solve, project back at every k).

**W — per-trial scan** (`b` fixed inside the scan body; T has NO `b` axis):
```
encode: T_b[μ,ν,t,s,k] = Σ_c ψ_c[k,c,t,μ] Σ_v conj(ψ_v[k,v,s,ν]) X_b[c,v,k]
conv:   U_b[μ,ν,t,s,k] = (1/√Nk) Σ_q W_q[μ,ν,q] T_b[μ,ν,t,s,k−q]   # ortho ifftₖ·W_R·fftₖ
decode: (WX)_b[c,v,k]  = (1/√Nk) Σ_{μ,ν,t,s} conj(ψ_c[k,c,t,μ]) ψ_v[k,v,s,ν] U_b[μ,ν,t,s,k]
```
Identical algebra to today's `apply_W_from_T` (ring_comm.py:335-350) with the leading `b`
removed — one trial at a time.

## Scan-inside-shard_map structure

W-term = ONE `shard_map` over the mesh `('x','y')`, body = `lax.scan` over the trial axis.
```
def _w_stack(X, psi_c_X, psi_v_Y, W_R):        # local shards; X:(n_trials,c_loc,v_loc,k)
    def body(carry, X_b):                       # carry = () ; X_b:(c_loc,v_loc,k)
        Xv   = all_gather(X_b, 'y', axis=1, tiled=True)        # (c_loc, v_full, k)
        R    = einsum('kvsN,cvk->cksN', conj(psi_v_Y), Xv)     # (c_loc, k, s, ν_loc)
        Rc   = all_gather(R, 'x', axis=0, tiled=True)          # (c_full, k, s, ν_loc)
        T_b  = einsum('kctM,cksN->MNtsk', psi_c_X, Rc)         # (μ_loc, ν_loc, t, s, k) ── the ONE intermediate
        T_k  = T_b.reshape(μ_loc, ν_loc, ns, ns, nkx, nky, nkz)
        U_b  = _local_fftn3(W_R * _local_ifftn3(T_k)).reshape(μ_loc, ν_loc, ns, ns, nk)   # donated over T_b
        A    = psum(einsum('kctM,MNtsk->cNsk', conj(psi_c_X), U_b), 'x')   # Σ_μ complete; (c_full,ν_loc,s,k)
        WXcv = psum(einsum('kvsN,cNsk->cvk',  psi_v_Y, A),        'y')      # Σ_ν complete; (c_full,v_full,k)
        WX_b = slice_local(WXcv, 'x','y') / √Nk                 # (c_loc, v_loc, k)
        return carry, WX_b
    _, WX = lax.scan(body, (), X)               # WX:(n_trials, c_loc, v_loc, k)
    return WX
```
`lax.scan` is a real loop, so XLA reuses the body's scratch (`T_b`, `T_k`, `U_b`) across
iterations — **one** T-family alive regardless of `n_trials`. A Python `for`/unrolled loop
or a naive outer `fori_loop` over trials inside `jit` would pile up `n_trials` live T slots
(the known slot-pile-up failure mode, `feedback_path_d_scaffolding_pattern`,
zeta_rchunk_memory_model lineage) — that is exactly what the scan avoids. Collectives
(`all_gather`/`psum`/`psum_scatter`) run per trial inside the scan body; that is the
memory-for-comm trade the owner asked for.

The final `psum + slice_local` pair is a `psum_scatter(scatter_dimension=…, tiled=True)` —
the pattern already in `apply_V_ring` (ring_comm.py:261) — reducing μ→scatter-c-on-x and
ν→scatter-v-on-y so the body returns the local `(c_loc,v_loc,k)` tile directly under
`out_spec P(None,'x','y',None)`. No replicated `(c_full,v_full)` buffer survives the step.

**FFT reuse (one FFT path).** `_local_ifftn3`/`_local_fftn3` are the *inner* of
`fft_helpers.make_sharded_{i,}fftn_3d` — those helpers are `shard_map(jnp.fft.ifftn on the
local shard)` (fft_helpers.py:304-345), and shard_map cannot nest. Consolidate: factor the
one-line local kernel (`jnp.fft.ifftn(x, axes, norm='ortho')`) into
`fft_helpers._local_ifftn3`/`_local_fftn3`; the existing `make_sharded_*` become the
shard_map wrappers around it (auto-partitioned callers) and the scan body calls the same
kernel directly (already inside a shard_map, k-axes device-local). One source, no raw
`jnp.fft` introduced in `bse/`.

## Memory proof sketch (per device, W-term shard_map)

| Buffer | Bytes (cplx128) | Scales with n_trials? |
|---|---|---|
| `X` stack (xs) | `n_trials · c_loc · v_loc · nk · 16` | yes — the input, unavoidable, = the trial stack |
| `WX` stack (ys) | same | yes — the output, unavoidable |
| `psi_c_X` / `psi_v_Y` | `nk·nc·ns·μ_loc·16` / `nk·nv·ns·ν_loc·16` | no |
| `W_R` | `μ_loc·ν_loc·nk·16` | no |
| **`T_b` + `T_k`/`U_b` FFT temps** | `≈ 3 · μ_loc·ν_loc·ns²·nk · 16` | **no — ONE, donated/reused by scan** |

Peak intermediate = the single `T`-family = `μ_loc·ν_loc·ns²·nk` (= `n_rmu²·n_spinor²·nk /
(px·py)`), matching the owner's bound (k included, per the clarification). Today's peak is
`n_trials ×` that. For Si 8v8c 4×4×4 (`nk=64, ns=2, μ_pad≈2048` on a 2×2 mesh → `μ_loc=ν_loc=1024`):
one `T` ≈ `1024·1024·4·64·16 B ≈ 4.3 GB`; at `n_trials=block_size=8` today's T is ≈ 34 GB
(OOM on 40 GB with the ψ caches). The scan holds 4.3 GB flat. `A`/`R` decode/encode
scratch are `≤ c_full·ν_loc·ns·nk` — same order, also one-at-a-time.

## Sharding constraints (every one)

Outer `jax.jit(_matvec, in_shardings=(sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y, sh.eps,
sh.eps, sh.W, sh.V), out_shardings=sh.X)` — `sh.X = P(None,'x','y',None)` already carries a
leading axis; `n_trials` occupies it (replaces block `b`). No new sharding vocabulary.
- input `X`, `D_term`, `VX`, `WX`, output → `sh.X` (`wsc` on `D_term`, `VX+WX` sum).
- exchange `S[b,ν]` → `P(None,'y')` (k-free; a `wsc` inline — one dropped axis vs
  `sh.S=P(None,'y',None)`; add `sh.S_k0 = P(None,'y')` accessor to `make_bse_shardings` to
  keep it single-sourced, no new class).
- exchange `U[b,μ]` → `P(None,'x')` (`sh.d_mu`-shaped without k → `P(None,'x')`).
- W-term shard_map `in_specs=(P(None,'x','y',None), P(None,None,None,'x'),
  P(None,None,None,'y'), sh.W.spec)`, `out_specs=P(None,'x','y',None)`. Inside shard_map:
  no `wsc` (operands already local); collectives carry the layout.

## Kernel toggle + the W-tile seam

`build_bse_stack_matvec(mesh, nkx,nky,nkz, *, kernel='bse')`. `kernel ∈ {'rpa','bse'}` is a
plain string: `'rpa'` → return `D+V` (W-term shard_map not built); `'bse'` → `D+V−W`. No
class, no config object.

**The seam is the single line `U_b = fftₖ(W_R * ifftₖ(T_b))`.** `W_R` is a plain argument
`(μ_pad,ν_pad,nkx,nky,nkz)` sharded `sh.W`, built **once outside** the matvec (and outside
the scan) via `make_sharded_ifftn_3d(W_q)`. Because the seam only reads a shape-stable tile
`W_R[μ_loc,ν_loc,q]`:
- **W(ω)**: the caller loops ω and calls the same matvec with `W_R(ω_i)`; nothing in
  encode/decode/scan changes. Signature of the provider: `get_W_R(ω) -> Array[μ_pad,ν_pad,
  nkx,nky,nkz] @ sh.W`.
- **Ladders / pseudopole-W**: a provider `ladder_kernel(mesh,…) -> Array[μ_pad,ν_pad,nkx,
  nky,nkz] @ sh.W` returns the T-matrix/ladder-dressed real-space kernel; caller passes
  `W_R_eff = W_R + K_ladder_R` (same shape/sharding). The scan body is byte-identical — a
  ladder is "a different `W_R`". This is the one hook the full-frequency-W + ladder
  buildout needs.

## File plan

New `src/bse/bse_stack_matvec.py` (~180 LOC): `build_bse_stack_matvec` + the per-trial
local `_encode_T_local`/`_decode_W_local` (plain-array helpers, no `b`) + the batched dense
exchange. Consolidations (single-source, no parallel paths):
- `fft_helpers._local_ifftn3/_local_fftn3` factored out; `make_sharded_*fftn_3d` wrap them.
- `_encode_T_local`/`_decode_W_local` are the ONE encode/decode; `build_bse_ring_matvec`'s
  `encode_T_ring`/`encode_T_gather`/`apply_W_from_T` are deleted when consumers repoint.
- exchange uses `compute_pair_amplitude` (bse_serial.py:12) — do not re-inline M.

## Consumers + retirement path

Supersedes the **memory-motivated** TDA matvecs — `ring` (ppermute) and `gather`
(all_gather) both existed only to bound T's peak; the scan bounds it strictly better (no
`b` on T). `simple` (auto-SPMD) is kept transiently as the dense-exchange reference for the
gate, then folded in.
1. Land `bse_stack_matvec.py` + gates (this branch).
2. Repoint `solve_bse_sharded`: block path (bs>1) and bs=1 path both call the stack matvec
   (bse_lanczos.py:253-303); `matvec_kind` collapses to a single path.
3. Repoint FEAST `_rayleigh_ritz` (bse_feast.py:337-353): replace the `for v in vectors`
   loop with one stack call on `jnp.stack(vectors)` — one dispatch, one compiled program;
   the Davidson `apply_H` (bse_lanczos.py:198) already passes a stack, drop-in.
4. Delete `build_bse_ring_matvec` + `bse_simple` + `matvec_kind` selector in the same
   change. Non-TDA (`build_bse_ring_matvec_full`) is out of scope (B2 broken); it will
   reuse `_encode_T_local`/`_decode_W_local` once B2 lands — leave it, note the shared
   helper. (solver_program.md P1: matvec-closure contract is the seam to the solver zoo.)

## Gate plan (1 GPU, MoS2/Si-scale, pytest-collected)

Phase the series so each commit has a clean equality gate:
- **Commit 1 — memory refactor, exchange kept k-diagonal.** Stack matvec with the OLD
  k-batched `S[b,ν,k]`. Gate: **byte-equal to `bse_simple`** on `D+V+W` for a random
  `n_trials=4` stack (MoS2 3×3 restart from the session fixture's `tmp/isdf_tensors_*.h5` —
  piggyback `gnppm_session`/`bispinor_session` in `tests/conftest.py`; no second GW run).
  `rtol<1e-12`. This isolates the structural change.
- **Commit 2 — B1 dense exchange.** Flip `S` to k-summed. Gates: (a) exchange-only vs a
  **dense 2-k toy reference** (explicit `⟨cvk|K^x|c'v'k'⟩` quadrature, VERDICT.md
  §Evidence); (b) `D+W` still byte-equal to simple (exchange isolated); (c) optional
  BGW `bsemat.h5` `/mats/exchange` off-diagonal k-blocks (empirical_bsemat.md).
- **memory_analysis() evidence.** AOT-compile the stack matvec at `n_trials ∈ {1,4,8}` and
  record `compiled.memory_analysis().temp_size_in_bytes` per rank (the same
  `AOT-compile → memory_analysis()` recipe fft_helpers.py:44-108 already uses). Assert the
  peak-temp is **flat in `n_trials`** (T off the batch axis) vs `build_bse_ring_matvec`'s
  linear growth — the machine-checked proof of the owner's bound. Record the table in this
  report on execution.
- **Golden gates unaffected.** BSE is not in the pytest GW gate set (kernel_dataflow_trace
  §Entry points); `test_gw_jax_regression` (cohsex/si_cohsex_3d/gnppm) +
  `test_ibz_equals_full_bz` stay green (no GW path touched). Run the full plain 1-GPU suite
  before the final commit of each series (copy `liblorrax_ffi.so` into the worktree first —
  KNOWN_SANDBOX_ERRORS 2026-07-15).

## Open items for Jack

1. Keep `bse_simple` as a permanent 1-device reference, or delete after Commit 2? (design
   assumes delete — single source.)
2. Commit-1/Commit-2 split OK, or land the B1 dense exchange in one shot? (split gives the
   clean byte-equal gate; B1 is greenlit as its own gated commit per VERDICT.md.)
3. W(ω)/ladder buildout: is the `get_W_R(ω)` provider closure the desired seam, or should
   the matvec take `W_q` and IFFT internally per-ω? (design keeps IFFT outside, once.)
