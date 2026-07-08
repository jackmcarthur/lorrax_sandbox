#!/usr/bin/env python3
"""Determine whether W0_ppm in w_copies_debug.h5 is W_c or full W.

Checks W0_screen - W0_ppm: if it is a large, positive, diagonal-dominant,
nearly-real matrix (i.e. v_munu), the two datasets differ by the bare Coulomb
and W0_ppm is the correlation part W_c. Also prints diag stats of each.
"""
import h5py
import numpy as np

P = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/"
     "01_lorrax_gn_ppm/w_copies_debug.h5")

def rc(ds):
    a = np.asarray(ds)
    if a.dtype.kind == "c":
        return a
    if a.dtype.names:
        return a[a.dtype.names[0]] + 1j * a[a.dtype.names[1]]
    return a[..., 0] + 1j * a[..., 1]

with h5py.File(P, "r") as f:
    for k in f:
        print(k, f[k].shape, f[k].dtype)
    Wp0 = rc(f["W0_ppm_q000_munu"])
    Ws0 = rc(f["W0_screen_q000_munu"])
    Wiw = rc(f["Wiwp_ppm_q000_munu"])

D = Ws0 - Wp0
for name, M in (("W0_ppm", Wp0), ("W0_screen", Ws0), ("Wiwp_ppm", Wiw),
                ("screen-ppm (v?)", D)):
    d = np.diag(M)
    off = M - np.diag(d)
    print(f"{name:16s} diag Re: med={np.median(np.real(d)):+.4e} "
          f"min={np.real(d).min():+.4e} max={np.real(d).max():+.4e} | "
          f"max|offdiag|={np.abs(off).max():.4e} "
          f"mean|offdiag|={np.abs(off).mean():.4e} "
          f"max|Im|={np.abs(np.imag(M)).max():.3e}")
print("\nrel |D| vs |W0_ppm|: ", np.abs(D).sum() / np.abs(Wp0).sum())
print("D diag all positive? ", bool((np.real(np.diag(D)) > 0).all()))
# Hermiticity checks (W_c should be ~Hermitian in an orthonormal-ish basis)
for name, M in (("W0_ppm", Wp0), ("Wiwp_ppm", Wiw)):
    h = np.abs(M - M.conj().T).max() / np.abs(M).max()
    print(f"{name} rel non-hermiticity: {h:.3e}")
