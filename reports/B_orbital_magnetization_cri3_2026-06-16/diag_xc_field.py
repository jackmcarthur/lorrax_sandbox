"""Quantitative confirmation: the missing xc exchange field B_xc near the Cr
atoms.  LORRAX's standalone V_xc is spin-UNPOLARIZED (charge only); QE used
spin-polarized noncollinear V_xc.  The omitted spin splitting V_x^up - V_x^dn
should be ~1 eV near the Cr nuclei, matching the band-selective H residual."""
import os, sys, numpy as np, h5py
QESAVE = sys.argv[1]; RY = 13.605693122994
f = h5py.File(os.path.join(QESAVE, "charge-density.hdf5"), "r")
mill = np.asarray(f["MillerIndices"]); nx, ny, nz = 45, 45, 120
def to_r(name):
    g = np.zeros((nx, ny, nz), np.complex128)
    v = np.asarray(f[name]).view(np.complex128)
    g[mill[:,0]%nx, mill[:,1]%ny, mill[:,2]%nz] = v
    return np.fft.ifftn(g).real * (nx*ny*nz)
n  = to_r("rhotot_g")                         # charge density e/bohr^3
mz = to_r("m_z"); mx = to_r("m_x"); my = to_r("m_y")
mmag = np.sqrt(mx**2 + my**2 + mz**2)         # |m|(r)
nup = np.clip((n + mmag)/2, 0, None); ndn = np.clip((n - mmag)/2, 0, None)
# LSDA exchange potential, Rydberg:  V_x^sigma = -2 (3/pi)^{1/3} (2 n_sigma)^{1/3}
c = 2.0 * (3.0/np.pi)**(1.0/3.0)
Vxu = -c * (2*nup)**(1.0/3.0); Vxd = -c * (2*ndn)**(1.0/3.0)
Bxc = 0.5*(Vxu - Vxd)                          # the omitted xc field (Ry), <0 where m>0
print(f"|m| integral (net moment, z) = {mz.sum()*4960.595/(nx*ny*nz):.3f} mu_B-ish (sum m_z)")
print(f"max |m|(r)               = {mmag.max():.4f} e/bohr^3   at charge peak n={n.max():.3f}")
print(f"missing exchange field |V_x^up - V_x^dn|:")
print(f"   max  = {np.abs(Vxu-Vxd).max()*RY:.0f} meV   ({np.abs(Vxu-Vxd).max():.4f} Ry)")
print(f"   |B_xc| max = {np.abs(Bxc).max()*RY:.0f} meV")
# value AT the charge-density peak (a Cr nucleus region)
pk = np.unravel_index(np.argmax(n), n.shape)
print(f"   at charge peak {pk}: |m|={mmag[pk]:.3f}  V_x^up-V_x^dn={(Vxu-Vxd)[pk]*RY:.0f} meV")
# the residual was worst on the deep Cr 3s (1037 meV) -- compare the local
# exchange-field where the Cr semicore density lives (high-|m| voxels):
hi = mmag > 0.5*mmag.max()
print(f"   over high-|m| region (Cr atoms): mean |V_x^up-V_x^dn| = "
      f"{np.abs(Vxu-Vxd)[hi].mean()*RY:.0f} meV  max = {np.abs(Vxu-Vxd)[hi].max()*RY:.0f} meV")
