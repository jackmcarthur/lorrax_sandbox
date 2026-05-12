# Plan: pad the band axis once, propagate sentinels, trust the zero contract

**Author:** agent A, 2026-05-04
**Status:** proposal — not implemented in this branch

## What's actually broken

Confirmed by direct test: running the Si COHSEX driver on **nband=57 (or 60 on world=8, or any nband not divisible by world)** crashes here:

```
File "src/common/phdf5_wfn_reader.py", line 110, in coeffs_gspace
    raise ValueError(f"band count {nb} not divisible by world={self._world_size}")
```

The trip is on the **per-band-chunk size**, not the total. With `nband=57`, `band_chunk_size=16`, `world=4`: chunks are 16, 16, 16, 9; the last (9) fails `9 % 4 ≠ 0`. So the surface is "any nband whose tail chunk doesn't divide world" — basically any non-aligned input.

Where `nb_padded` already exists in the codebase (`gvec_fft_box.py:73, 96`, `load_wfns.py:370, 716`, `phdf5_wfn_reader.py:93, 110, 395`), the term is used as a precondition ("caller must already have padded"), not as an in-function pad. No caller pads. That's the gap.

## Where padding is and isn't safe — the operations audit

The band axis flows through these operations. For each, "is ψ_padded=0 safe?":

| operation | reads | safe with ψ=0? | what it needs |
|---|---|---|---|
| iFFT to real space | ψ(G) → ψ(r) | yes (FFT of zero is zero) | nothing |
| pair density `P(μ,ν) = Σ_n ψ*(μ)ψ(ν)` | ψ(r_μ), ψ(r_ν) | yes (zero rows contribute 0) | nothing |
| CCT k→q FFT | P_k | yes | nothing |
| Cholesky / solve | C_q | n/a (no band axis) | n/a |
| Σ_X bare = `−Σ_n f_n M_n V M_n` | ψ, e_n,k, V_q | yes if M_padded=0 | M_padded=0 follows from ψ_padded=0 |
| Σ_C(ω) PPM = `Σ_n ω_n / (ω - e_n,k ± iη) ⋅ M_n V M_n` | ψ, e_n,k, V, ω₀ | yes if M_padded=0 | M_padded=0 + e_padded any finite-ish value (resolvent×0=0) |
| W solve / dyson | χ₀(q,ω) | n/a (no explicit band axis after χ₀ build) | n/a |
| pseudoband norm divide `ψ / max(1, w_n)` | ψ, band_norms | yes (existing `max(1, 0)=1` floor) | band_norms_padded set to 0 (or 1) |
| occupation `f_n = step(E_F - e_n,k)` | e_n,k | yes if e_padded > E_F | e_padded = ∞ (or any large finite, e.g. `wfn.energies.max() + 1` Ry) |
| `E_min`/`E_max` for weight heuristic in `get_enk_bandrange:121` | `wfn.energies[:, sigma_lo:sigma_hi]` | yes | uses raw `wfn.energies`, not padded `enk` — already safe |
| eqp.dat / sigma.h5 output writer | enk, sigma | n/a | iterate user's original nband, slice off pad |

**Conclusions for the safety contract:**

1. **`ψ_padded(G) = 0`** is the "carrier" of the pad-safety guarantee. Zero ψ propagates through FFTs, gathers, einsums, divisions (with the existing `max(1, w)` floor) without ever creating a NaN or non-zero contribution.
2. **`e_padded` needs a finite sentinel above E_F**, not `∞` — sigma evaluation does arithmetic on `e_n,k` (`ω - e + iη`) where `inf - finite = inf` is fine but a downstream `1/(inf + iη)` could trigger fp-warnings. Use `max(wfn.energies) + 1 Ry`. Big enough for `f=0`, finite enough for arithmetic safety.
3. **`band_norms_padded`** can be left at zero — the existing `_band_norms_slice` clamps to `max(1, w)` so divides become `ψ/1 = 0`. No code change needed if we leave the default.
4. **No band-axis operation requires non-zero contribution from the pad.** Every reduction is sum/mean (zero-safe), every divide has the existing floor, every sub-Hamiltonian assembly uses ψ as a coefficient.

The padding is therefore **uniformly safe under one rule: ψ_padded(G) = 0, e_padded = E_max + δ.**

## The minimal-touch design

Two reasonable places to pad: at the *reader* (per band chunk), or at the *driver* (once, on `nband` itself). The driver-level pad is much simpler because it ensures the chunked loop's last chunk size is also clean — no special-casing inside the read path.

### Pad at one place: `gw_init` (driver-level)

```
# pseudocode in gw_init.py, after meta is built
world = jax.device_count()
nb_user = b_end - b_start                           # what the user asked for
nb_padded = ((nb_user + world - 1) // world) * world
b_end_padded = b_start + nb_padded                  # may exceed wfn.nbnd
meta.b_id_4 = b_end_padded                          # internal
meta.b_id_4_user = b_end                            # for output writers
```

Then propagate `b_id_4_user` to the eqp/sigma writers (one parameter, plumbed through one call chain). Everywhere else uses `b_end_padded` and inherits zero-band safety.

### Three (and only three) places need explicit pad-aware code

**A. `phdf5_wfn_reader.coeffs_gspace`** (the file-short case)

Replace the `raise ValueError` with: read up to `min(b_hi, self.nbnd)`, zero-fill the rest. The shape returned stays `(nk, b_hi - b_lo, …)`. With `b_hi - b_lo = nb_padded`, divisibility holds by construction.

```python
# was:
if nb % self._world_size:
    raise ValueError(...)
bands_per_rank = nb // self._world_size

# is:
assert nb % self._world_size == 0, (
    f"caller must pre-pad band_range; got nb={nb}, world={self._world_size}")
bands_per_rank = nb // self._world_size
# inside the read loop, if b_hi > self.nbnd:
#   read [b_lo, self.nbnd) into the leading rows; jnp.zeros for the trailing
```

(The assert stays as a sanity check; the contract is now "caller pre-pads", which `gw_init` honors.)

**B. `load_wfns.get_enk_bandrange`** (the energy-sentinel case)

```python
# detect overrun
b_hi_eff = min(band_hi, wfn.energies.shape[2])
en_irk = np.asarray(wfn.energies[0, :, band_lo:b_hi_eff], dtype=np.float64)
if b_hi_eff < band_hi:
    sentinel = float(wfn.energies.max()) + 1.0  # Ry; > any physical band, finite
    pad = np.full((en_irk.shape[0], band_hi - b_hi_eff), sentinel)
    en_irk = np.concatenate([en_irk, pad], axis=1)
enk = en_irk[irk_to_k, :]
```

Weights for padded bands: 0 (so heuristic-driven weight contributions vanish). Already handled by the `np.where(enk <= efermi, …)` since sentinel > E_F → cond branch with weight `1/sqrt(sentinel - E_min)` ≈ 0; tighten by setting `weights[:, b_hi_eff - band_lo:] = 0.0` to make it exact.

**C. Output writers** (`gw/sigma_output.py` / `gw/eqp_bgw.py` / wherever eqp0.dat is built)

Iterate over `[b_id_0, b_id_4_user)` instead of `[b_id_0, b_id_4)`. Drops the padded rows from the output table. One slice per writer.

That's it.

### What does NOT need to change

- `compute_pair_density_spin_traced`, `compute_CCT_from_left_right`, `compute_L_q_from_CCT`, `solve_zeta_from_L_q`, the V_q kernel, Σ_X kernel, Σ_C(ω) PPM kernel, `cholesky_2d_batched` — all are pure reductions/contractions with no per-band-index special-casing. Zero-band → zero contribution everywhere.
- `_band_norms_slice` already returns 1.0 for zero-weight bands via `max(1, w)`. No change.
- The mesh / sharding code. The pad keeps `nb_padded % world == 0`, which is what every sharded path already wants.
- `pivoted_cholesky` (operates on centroids, not bands).

## Edge cases

1. **WFN.h5 is short (b_end_padded > nbnd_in_file).** Common case for high `nband` near the file's limit. Reader zero-pads; energies use sentinel; physics correct (those bands' contributions are zero by construction).
2. **`nband` already aligned (e.g. 60 on world=4).** `nb_padded == nb_user`; no pad rows; bit-identical to today's behavior.
3. **band_chunk_size not aligned.** Doesn't matter — the per-chunk size is determined by the chunker, and as long as `nband_total` is aligned, the *last* chunk is `nb_total - (n_full_chunks * chunk_size)`. We'd need to also align `band_chunk_size` to `world` so per-chunk sizes are clean. That's a one-line fix in the chunk-size chooser.
4. **Pseudobands `band_norms` array shorter than `nb_padded`.** Extend `_band_norms_slice` to fill the tail with 0 (which the existing `max(1, 0)=1` floor turns into "divide by 1"). One extra line in that helper.
5. **Bispinor mode (nspinor=2).** `weights` is `np.repeat(weights, repeats=nspinor, axis=1)` in get_enk_bandrange. Zero-weight tail repeats correctly; safe.

## What I'm asking for before I implement

- Does this match your intent for the pad design? Specifically:
  - Padding once at `gw_init` (driver-level), not per-call inside the reader.
  - Sentinel of `wfn.energies.max() + 1 Ry` rather than `∞` for `e_padded`.
  - Output writers (eqp.dat, sigma.h5) truncate to user `nband`; everything else operates on padded.
- Are there other operations that read `e_n,k` arithmetically that I haven't enumerated? (PPM ω₀ extraction comes to mind — would need to confirm it doesn't choke on a sentinel energy. I'd verify this in implementation.)
- Should the pad helper live in `common/sharding_utils.py` (new file) or just be inline in `gw_init`? My read: inline for now (1 caller), promote to a helper if `kin_ion_io` / `psp/get_dipole_mtxels` adopt the same pattern.

## Estimated change size

| file | lines changed |
|---|---|
| `gw/gw_init.py` | +5 (compute padded nb, store user nb in meta) |
| `common/load_wfns.py:get_enk_bandrange` | +5 (sentinel-fill for short WFN) |
| `common/phdf5_wfn_reader.py:coeffs_gspace` | +3 (zero-fill for short WFN; assert stays) |
| `common/isdf_fitting.py:_band_norms_slice` | +2 (pad with zeros to nb_padded) |
| `gw/eqp_bgw.py` (or wherever eqp0.dat is written) | +1 (slice to user nband) |
| `gw/sigma_output.py` (sigma.h5 writer) | +1 (slice to user nband) |
| `gw/gw_config.py` (or `Meta`) | +1 (`b_id_4_user` field) |

Total: ~18 lines across 7 files. The actual padding logic is ~3 lines; everything else is plumbing the user's original nband through to the output writers.

## Test plan

1. **Single-mesh sanity**: Si nband=57 on (2,2). Should produce eqp0.dat with 57 entries per k. Bare Σ_X should match Si nband=56 to within numeric noise (the 57th band contributes ~zero to Σ_X anyway since it's high-energy unoccupied).
2. **Cross-mesh determinism**: Si nband=57 on (2,2) vs (2,4) vs (4,4). Bare Σ_X bit-equivalent to ~meV.
3. **Edge case**: nband > file's nbnd. Si has 62 bands in WFN.h5; run with nband=70 on (2,4). Should pad with zeros, produce sensible Σ_X (matches nband=62 within noise).
4. **No-pad regression**: Si nband=64 on (2,4). nb_padded == nb; should be bit-identical to current main.
