"""Is the CrI3 V_loc error the Hankel-transform accuracy on the UPF log grid?
Compare vloc_sr_table for Cr computed on (a) the native UPF radial grid vs
(b) a spline-refined 8x-denser radial grid, at the q-values that matter
(0..12 bohr^-1).  If V_loc(q) differs materially -> radial-FT accuracy is the
bug.  Pure-numpy radial integral; no GPU needed beyond the import."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from scipy.special import spherical_jn, erf
from psp.pseudos import load_pseudopotentials
from psp.species import extract_species

PDIR = sys.argv[1]
pseudos = load_pseudopotentials(PDIR)
species = extract_species(pseudos, nspinor=2)
sp = next((s for s in species if str(s.element).startswith("Cr")), species[0])
print("species:", sp.element, " z_valence=", sp.z_valence, " n_r=", len(sp.r))
e2 = 2.0
r, rab, vloc_r = np.asarray(sp.r), np.asarray(sp.rab), np.asarray(sp.vloc_r)
v_sr = vloc_r + sp.z_valence * e2 * erf(r) / np.where(r > 0, r, 1.0)
print(f"v_sr(r->0) = {v_sr[1]:.4f} Ry   vloc_r(min)={vloc_r.min():.2f} Ry")

def hankel0_native(q):
    out = np.zeros_like(q)
    for i, qq in enumerate(q):
        integ = spherical_jn(0, qq * r) * r * r * v_sr * rab
        out[i] = 4.0 * np.pi * np.sum(integ)   # crude trapz-on-rab (= what hankel_l does w/ Simpson)
    return out

# refined: cubic-spline v_sr onto an 8x denser uniform-in-index log grid
from scipy.interpolate import CubicSpline
# build a denser radial grid by log-uniform refinement between r[1]..r[-1]
lr = np.log(r[1:]); lr_fine = np.linspace(lr[0], lr[-1], len(lr) * 8)
r_f = np.exp(lr_fine)
cs = CubicSpline(r[1:], v_sr[1:])
v_f = cs(r_f)
# rab on fine grid = dr/di; for log grid dr = r * dln r
dlnr = lr_fine[1] - lr_fine[0]
rab_f = r_f * dlnr

def hankel0_fine(q):
    out = np.zeros_like(q)
    for i, qq in enumerate(q):
        integ = spherical_jn(0, qq * r_f) * r_f * r_f * v_f * rab_f
        out[i] = 4.0 * np.pi * np.sum(integ)
    return out

q = np.linspace(0.0, 12.0, 25)
vn = hankel0_native(q); vf = hankel0_fine(q)
print(f"\n  q[bohr^-1]   V_loc_native     V_loc_8x       diff       reldiff")
for i in range(len(q)):
    d = vf[i] - vn[i]
    rd = d / (abs(vn[i]) + 1e-12)
    print(f"  {q[i]:7.3f}   {vn[i]:13.5f}  {vf[i]:13.5f}  {d:10.5f}  {rd:8.2%}")
print(f"\nmax |diff| over q = {np.max(np.abs(vf-vn)):.5f} (Ry*vol units)")
