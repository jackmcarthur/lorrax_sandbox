"""
Benchmark: does axis ordering matter for jnp.fft.fftn / ifftn?

Scenario from GWJAX:
  V(qx, qy, qz, mu, nu)  or equivalently  V(mu, nu, kx, ky, kz)

We want to FFT over the 3 k/q-grid dimensions.  Three layouts tested:

  A) "k-last":          physically contiguous as (mu, nu, kx, ky, kz),
                         FFT on axes=(-3,-2,-1)
  B) "k-first":         physically contiguous as (kx, ky, kz, mu, nu),
                         FFT on axes=(0,1,2)
  C) "xpose+fft+xpose": input is k-last contiguous, but inside the jit we
                         transpose to k-first, FFT, transpose back.

Key subtlety: JAX's .transpose() returns a lazy view (non-contiguous strides).
So naively doing `arr.transpose(2,3,4,0,1)` and passing that as input gives
XLA a strided buffer — NOT the same as a physically k-first array.
We use jnp.array(arr.transpose(...), copy=True) to force a physical copy.
"""

import jax
import jax.numpy as jnp
import time


def _make_contiguous_copy(arr):
    """Force a contiguous physical copy (not a strided view)."""
    return jnp.array(arr, copy=True)


def _mem_ok(n_basis, nk, n_copies=3):
    """Check if we can fit n_copies of the array in ~6 GB."""
    return n_basis * n_basis * nk * 16 * n_copies < 6e9


def bench(kgrid, n_basis, n_warmup=5, n_iter=30):
    kx, ky, kz = kgrid
    nk = kx * ky * kz

    if not _mem_ok(n_basis, nk):
        return None

    # --- Build physically contiguous arrays for each layout ---
    key = jax.random.key(42)
    arr_klast = jax.random.normal(key, (n_basis, n_basis, kx, ky, kz), dtype=jnp.float64)
    arr_klast = arr_klast + 1j * jax.random.normal(
        jax.random.key(43), arr_klast.shape, dtype=jnp.float64)

    # Physical copy in k-first order (not a view!)
    arr_kfirst = _make_contiguous_copy(arr_klast.transpose(2, 3, 4, 0, 1))

    # Verify they're actually different physical layouts
    assert arr_klast.shape == (n_basis, n_basis, kx, ky, kz)
    assert arr_kfirst.shape == (kx, ky, kz, n_basis, n_basis)

    # --- Define the three strategies ---
    @jax.jit
    def fft_klast(x):
        return jnp.fft.fftn(x, axes=(-3, -2, -1), norm='ortho')

    @jax.jit
    def ifft_klast(x):
        return jnp.fft.ifftn(x, axes=(-3, -2, -1), norm='ortho')

    @jax.jit
    def fft_kfirst(x):
        return jnp.fft.fftn(x, axes=(0, 1, 2), norm='ortho')

    @jax.jit
    def ifft_kfirst(x):
        return jnp.fft.ifftn(x, axes=(0, 1, 2), norm='ortho')

    @jax.jit
    def fft_xpose(x):
        """k-last in → transpose → FFT k-first → transpose → k-last out."""
        y = jnp.fft.fftn(x.transpose(2, 3, 4, 0, 1), axes=(0, 1, 2), norm='ortho')
        return y.transpose(3, 4, 0, 1, 2)

    @jax.jit
    def ifft_xpose(x):
        y = jnp.fft.ifftn(x.transpose(2, 3, 4, 0, 1), axes=(0, 1, 2), norm='ortho')
        return y.transpose(3, 4, 0, 1, 2)

    # Also time bare transpose cost
    @jax.jit
    def just_transpose(x):
        return _make_contiguous_copy(x.transpose(2, 3, 4, 0, 1))

    results = {}
    cases = [
        ("fft_klast",   fft_klast,   arr_klast),
        ("ifft_klast",  ifft_klast,  arr_klast),
        ("fft_kfirst",  fft_kfirst,  arr_kfirst),
        ("ifft_kfirst", ifft_kfirst, arr_kfirst),
        ("fft_xpose",   fft_xpose,   arr_klast),
        ("ifft_xpose",  ifft_xpose,  arr_klast),
        ("transpose",   just_transpose, arr_klast),
    ]

    try:
        for label, fn, arr in cases:
            # Warmup (compilation + cache warming)
            for _ in range(n_warmup):
                out = fn(arr)
                out.block_until_ready()

            times = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                out = fn(arr)
                out.block_until_ready()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            times.sort()
            results[label] = times[len(times) // 2]
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return None
        raise

    return results


def main():
    print(f"JAX {jax.__version__} | {jax.devices()} | x64={jnp.ones(1, dtype=jnp.float64).dtype}")
    print()

    kgrids = [
        (4, 4, 1),
        (4, 4, 4),
        (6, 6, 6),
        (8, 8, 8),
        (10, 10, 10),
    ]
    n_bases = [100, 200, 400]

    hdr = (f"{'kgrid':>12s}  {'nb':>4s}  {'nk':>5s}  {'size':>6s}  "
           f"{'fft_klast':>10s}  {'fft_kfirst':>10s}  {'fft_xpose':>10s}  "
           f"{'ifft_klast':>10s} {'ifft_kfirst':>10s}  {'ifft_xpose':>10s}  "
           f"{'transpose':>10s}  "
           f"{'kfirst/klast':>12s}")
    print(hdr)
    print("-" * len(hdr))

    for kgrid in kgrids:
        for nb in n_bases:
            nk = kgrid[0] * kgrid[1] * kgrid[2]
            size_mb = nb * nb * nk * 16 / 1e6

            res = bench(kgrid, nb)
            if res is None:
                print(f"{str(kgrid):>12s}  {nb:>4d}  {nk:>5d}  {size_mb:>5.0f}M  "
                      f"{'SKIP (OOM)':>60s}")
                continue

            ratio = res['fft_kfirst'] / res['fft_klast']

            def ms(key):
                return f"{res[key]*1e3:>8.3f}ms"

            print(f"{str(kgrid):>12s}  {nb:>4d}  {nk:>5d}  {size_mb:>5.0f}M  "
                  f"{ms('fft_klast')}  {ms('fft_kfirst')}  {ms('fft_xpose')}  "
                  f"{ms('ifft_klast')} {ms('ifft_kfirst')}  {ms('ifft_xpose')}  "
                  f"{ms('transpose')}  "
                  f"{ratio:>11.2f}x")

    print()
    print("kfirst/klast = fft_kfirst / fft_klast  (<1 means k-first is faster)")
    print("All arrays are physically contiguous in their respective layouts.")
    print("'fft_xpose' = transpose→fft→transpose inside one jit (input is k-last contiguous).")


if __name__ == "__main__":
    main()
