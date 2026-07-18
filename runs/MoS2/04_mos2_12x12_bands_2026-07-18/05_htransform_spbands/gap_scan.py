"""Adjacent-band minimum gaps over the 12x12 BZ from the stored restart
energies — window-boundary safety scan.  A htransform window boundary
between bands (b-1 | b) is UNSAFE if min_k [eps_b(k) - eps_{b-1}(k)] ~ 0
anywhere (window truncation of a degenerate multiplet breaks the k-
smoothness of fH and hence off-grid interpolation; Si root-cause 2026-07;
the w2331/w2533 ~2 eV off-grid failures)."""
import numpy as np
import h5py

RY2EV = 13.6056980659
with h5py.File("../00_lorrax_cohsex/tmp/isdf_tensors_640.h5", "r") as f:
    enk = f["enk_full"][()]          # (nk, 80) Ry
print(f"enk_full shape {enk.shape}")
print("boundary (b-1|b)   min_k gap (meV)   argmin k")
for b in range(18, 37):
    g = (enk[:, b] - enk[:, b - 1]) * RY2EV * 1e3
    print(f"  {b-1:3d}|{b:<3d}  {g.min():14.3f}   {int(g.argmin()):4d}")

print("\nband   min (eV)   max (eV)   BW (eV)   [4*BW = f-transform a]")
for b in range(17, 36):
    e = enk[:, b] * RY2EV
    print(f"  {b:3d}  {e.min():9.4f}  {e.max():9.4f}  {e.max()-e.min():8.4f}"
          f"   {4*(e.max()-e.min()):8.4f}")
