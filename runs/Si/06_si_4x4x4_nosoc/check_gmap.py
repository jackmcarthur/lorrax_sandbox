"""Check whether the rotated-IBZ G-sphere differs from the target G-sphere
at full-BZ k-points. BGW zeroes coefficients in the target sphere whose
inverse-rotated G falls outside the IBZ sphere; LORRAX SymMaps just rotates
the IBZ G-list. They differ near the sphere edge.

For each full-BZ k_full:
  1. rotated_IBZ = { S G - kg0 | G in IBZ sphere of k_irr }    (LORRAX)
  2. target = G-vectors of k_full in WFN.h5 (independently sorted by |k_full+G|²)
  3. report set differences
"""
import os; os.environ.setdefault("MPLBACKEND", "Agg")
import sys, numpy as np
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
from runtime import set_default_env; set_default_env()
from file_io import WFNReader
from common.symmetry_maps import SymMaps

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/nscf/WFN.h5"
ECUTWFC = 25.0   # Ry

wfn = WFNReader(WFN)
sym = SymMaps(wfn)
bvec_cart = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)  # 1/bohr
nx, ny, nz = (int(v) for v in wfn.fft_grid)

# Build full G-set within FFT box, then filter by kinetic energy
ix = np.concatenate([np.arange(0, nx//2 + 1), np.arange(-(nx//2 - 1), 0)])
iy = np.concatenate([np.arange(0, ny//2 + 1), np.arange(-(ny//2 - 1), 0)])
iz = np.concatenate([np.arange(0, nz//2 + 1), np.arange(-(nz//2 - 1), 0)])
GX, GY, GZ = np.meshgrid(ix, iy, iz, indexing='ij')
G_all = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=-1).astype(int)

def target_sphere(k_full):
    """Return sorted set of G's such that |k_full + G|² ≤ 2·ecutwfc."""
    qG_cart = (np.asarray(k_full)[None, :] + G_all) @ bvec_cart
    qG_sq = np.sum(qG_cart ** 2, axis=1)
    in_sphere = qG_sq <= ECUTWFC + 1e-10
    return G_all[in_sphere]

def to_set(arr):
    return set(map(tuple, arr.tolist()))

print(f"  k_full           ngk_irr_rot  ngk_target   |only-rot|   |only-tgt|")
for nk_full in range(sym.nk_tot):
    k_full = sym.unfolded_kpts[nk_full]
    g_rot = np.asarray(sym.get_gvecs_kfull(wfn, nk_full))   # LORRAX rotated IBZ
    g_tgt = target_sphere(k_full)                            # actual sphere at k_full

    s_rot = to_set(g_rot)
    s_tgt = to_set(g_tgt)
    only_rot = s_rot - s_tgt   # in LORRAX, not in BGW target
    only_tgt = s_tgt - s_rot   # in BGW target, not in LORRAX
    print(f"  {str(k_full):20s}  {len(g_rot):>5d}      {len(g_tgt):>5d}      "
          f"{len(only_rot):>5d}        {len(only_tgt):>5d}")
    if nk_full < 4 and (len(only_rot) > 0 or len(only_tgt) > 0):
        print(f"    only_rot examples: {list(only_rot)[:5]}")
        print(f"    only_tgt examples: {list(only_tgt)[:5]}")
