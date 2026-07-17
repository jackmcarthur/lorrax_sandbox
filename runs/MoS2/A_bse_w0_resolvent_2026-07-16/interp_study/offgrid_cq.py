"""Off-grid / midpoint interpolation of C_q from a COARSE SUBLATTICE, validated
against the fine-grid direct-fit truth (READ-ONLY numpy).

The arbitrary-Q use case interpolates ingredients to q's that are NOT on the
coarse grid.  A fine grid that contains a coarser sublattice lets us validate
this with ground truth: interpolate C_q from the coarse sublattice (factor f)
to the fine-only ("midpoint") q's, compare to the fine direct-fit C_q there.

MoS2 4x4  -> coarse 2x2, midpoints = odd-index q's (ground truth on 4x4).
Si 4x4x4  -> coarse 2x2x2, midpoints = the 56 fine-only q's.
"""
import sys, h5py, numpy as np

restart = sys.argv[1]; label = sys.argv[2]
f_coarse = int(sys.argv[3])                       # sublattice factor (2)
kgrid_override = sys.argv[4] if len(sys.argv) > 4 else ""

with h5py.File(restart, "r") as f:
    psi = f["psi_full_y"][()]
    if "kgrid" in f:
        kgrid = f["kgrid"][()].astype(int)
    elif kgrid_override:
        kgrid = np.array([int(x) for x in kgrid_override.split(",")])
    else:
        vs = f["V_qmunu"].shape; kgrid = np.array([vs[3], vs[4], vs[5]])
nk, nb, ns, n_mu = psi.shape
nkx, nky, nkz = [int(x) for x in kgrid]; nq = nkx*nky*nkz

# rebuild C_q (all bands, charge)
psiX = np.conj(psi).transpose(0, 3, 1, 2)
P = np.einsum('kmna,knbr->karmb', psiX, psi, optimize=True).reshape(nkx, nky, nkz, ns, n_mu, n_mu, ns)
P_R = np.fft.ifftn(P, axes=(0, 1, 2), norm='forward')
C_R = np.einsum('xyzavmb,xyzavmb->xyzvm', np.conj(P_R), P_R, optimize=True)
C_q = np.transpose(np.fft.fftn(C_R, axes=(0, 1, 2), norm='forward').reshape(nq, n_mu, n_mu), (0, 2, 1))
del P, P_R, C_R
Cq_flat = C_q.reshape(nq, -1)

# index tables
idx3 = np.array([(ix, iy, iz) for ix in range(nkx) for iy in range(nky) for iz in range(nkz)])
qflat = idx3 / np.array([nkx, nky, nkz])[None, :]

# coarse sublattice: indices divisible by f_coarse in every dim that HAS >1 point
def on_coarse(ix, iy, iz):
    ok = True
    for i, n in [(ix, nkx), (iy, nky), (iz, nkz)]:
        if n > 1 and (i % f_coarse) != 0:
            ok = False
    return ok
coarse = [q for q, (ix, iy, iz) in enumerate(idx3) if on_coarse(ix, iy, iz)]
fine_only = [q for q in range(nq) if q not in coarse]
# coarse R-set: the sublattice's own reciprocal cell (R in [0, nk/f) per active dim)
def wrap(n, N): return n - N*((2*n) > N)
# coarse q-grid has ncoarse = nk/f points per active axis at q = j/ncoarse;
# its Fourier dual R runs over {0..ncoarse-1} in ORIGINAL lattice units
# (wrapped), NOT scaled by f_coarse.
Rc = []
nax = [max(1, n//f_coarse) if n > 1 else 1 for n in (nkx, nky, nkz)]
for rx in range(nax[0]):
    for ry in range(nax[1]):
        for rz in range(nax[2]):
            Rc.append([wrap(rx, nkx) if nkx > 1 else 0,
                       wrap(ry, nky) if nky > 1 else 0,
                       wrap(rz, nkz) if nkz > 1 else 0])
Rc = np.array(Rc)
print(f"[{label}] nq={nq} coarse(f={f_coarse})={len(coarse)} pts, "
      f"fine-only(midpoints)={len(fine_only)} pts, coarse R-set={len(Rc)}")

# fit C_R on the coarse sublattice, predict the fine-only q's
F = np.exp(-2j*np.pi*(qflat[coarse] @ Rc.T))
CR, *_ = np.linalg.lstsq(F, Cq_flat[coarse], rcond=None)
errs = []
for q0 in fine_only:
    f0 = np.exp(-2j*np.pi*(qflat[q0] @ Rc.T))
    pred = f0 @ CR
    errs.append(np.linalg.norm(pred - Cq_flat[q0]) / np.linalg.norm(Cq_flat[q0]))
errs = np.array(errs)
print(f"[{label}] OFF-GRID C_q midpoint interp (coarse {len(coarse)}-pt -> "
      f"{len(fine_only)} midpoints):  med rel-Frob={np.median(errs):.4e}  "
      f"max={np.max(errs):.4e}  min={np.min(errs):.4e}")
