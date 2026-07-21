"""Characterize the far-conduction V_H / Vxc defect.

Hypothesis: the band-diagonal Hartree <nk|V_H|nk> (gw/cohsex_sigma.py:hartree)
is a pure ISDF-CENTROID quadrature.  The centroids come from a charge-density
weighted k-means, so they have NO support in the vacuum of the MoS2 slab.
Far-conduction bands are vacuum / free-electron-like, so their |psi|^2 lives
where there are no centroids -> V_H is unrepresentable -> the whole error lands
on Vxc = E_dft - kin_ion - V_H -> the +100..+280 eV far-band scissors.

Test: for every (k,n) compute
  f_vac  = fraction of |psi_nk(r)|^2 in the vacuum slab region
  f_cent = centroid-set Riemann estimate of the norm,
           (V/N_r) * sum_mu |psi_nk(r_mu)|^2   (== 1 if the centroids sample the
           state the way they sample a slab-localized state)
and correlate with |dVxc| = |(kin_ion + V_H) - KIH_ref| from the QE kih.dat.
"""
import os
import numpy as np
import h5py

RUN = os.environ.get(
    "VAC_RUN",
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/03_lorrax_gw_postfix_2026-07-21")
QE = os.environ.get(
    "VAC_QE",
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/qe/nscf")
OUT = os.environ.get(
    "VAC_OUT",
    "/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_conduction_postfix_2026-07-21")
NB = 100
NVAL = 26


def parse_bgw_diag(path):
    data, ik = {}, -1
    for line in open(path):
        p = line.split()
        if len(p) == 5 and '.' in p[0]:
            ik += 1
            data[ik] = {}
        elif len(p) == 4:
            data[ik][int(p[1])] = float(p[2])
    return data


def parse_fd(path):
    cols, rows = None, {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#'):
            pp = s.lstrip('#').split()
            if len(pp) >= 3 and pp[0] == 'k' and pp[1] == 'n':
                cols = pp[2:]
            continue
        if not s or cols is None:
            continue
        p = s.split()
        if len(p) != len(cols) + 2:
            continue
        rows[(int(p[0]), int(p[1]))] = {c: float(v) for c, v in zip(cols, p[2:])}
    return rows


# ---------------- WFN ----------------
f = h5py.File(os.path.join(RUN, "WFN.h5"), "r")
ngk = f["mf_header/kpoints/ngk"][()]
nspin = int(f["mf_header/kpoints/nspin"][()])
nspinor = int(f["mf_header/kpoints/nspinor"][()])
fftg = f["mf_header/gspace/FFTgrid"][()]
gvecs = f["wfns/gvecs"][()]
nk = len(ngk)
print(f"nk={nk} nspin={nspin} nspinor={nspinor} FFTgrid={fftg} ngk[0]={ngk[0]}", flush=True)
off = np.concatenate([[0], np.cumsum(ngk)])
Nx, Ny, Nz = (int(v) for v in fftg)

# centroid grid indices (fractional -> nearest FFT grid point)
import glob
CENT = os.environ.get("VAC_CENT") or sorted(
    glob.glob(os.path.join(RUN, "centroids*.txt")))[0]
print(f"centroid file: {CENT}", flush=True)
cf = np.loadtxt(CENT)
if cf.ndim == 1:
    cf = cf.reshape(1, -1)
cf = cf[:, :3]
ci = np.rint(cf * np.array([Nx, Ny, Nz])).astype(int) % np.array([Nx, Ny, Nz])
ncent = ci.shape[0]
print(f"centroids: {ncent}; unique grid cells {len(set(map(tuple, ci)))}", flush=True)
Nr = Nx * Ny * Nz

f_vac = np.zeros((nk, NB))
pz_all = np.zeros((nk, NB, Nz))
f_cent = np.zeros((nk, NB))
rho_z_val = np.zeros(Nz)

for k in range(nk):
    lo, hi = off[k], off[k + 1]
    gk = gvecs[lo:hi]
    ix = gk[:, 0] % Nx
    iy = gk[:, 1] % Ny
    iz = gk[:, 2] % Nz
    co = f["wfns/coeffs"][:NB, :, lo:hi, :]          # (NB, ns, ngk, 2)
    cc = co[..., 0] + 1j * co[..., 1]
    for b in range(NB):
        rho = np.zeros((Nx, Ny, Nz))
        for s in range(cc.shape[1]):
            g = np.zeros((Nx, Ny, Nz), dtype=complex)
            np.add.at(g, (ix, iy, iz), cc[b, s])
            r = np.fft.ifftn(g) * Nr
            rho += np.abs(r) ** 2
        tot = rho.sum()
        rho /= tot                                   # normalized weight per grid pt
        pz = rho.sum(axis=(0, 1))
        if b < NVAL:
            rho_z_val += pz
        f_cent[k, b] = rho[ci[:, 0], ci[:, 1], ci[:, 2]].sum() * (Nr / ncent)
        pz_all[k, b] = pz
    if k % 6 == 0:
        print(f"  k={k} done", flush=True)
f.close()

# vacuum region = z-planes carrying < 0.2% of the (k-summed) valence density max
rho_z_val /= rho_z_val.sum()
vac = rho_z_val < 0.002 * rho_z_val.max()
print(f"vacuum planes: {vac.sum()}/{Nz}  (valence z-profile max {rho_z_val.max():.4f})", flush=True)
f_vac = pz_all[:, :, vac].sum(axis=2)

# ---------------- errors ----------------
kih = parse_bgw_diag(os.path.join(QE, "kih.dat"))
fd = parse_fd(os.path.join(RUN, "sigma_freq_debug.dat"))
Ed = np.full((nk, NB), np.nan); ki = np.full((nk, NB), np.nan); VH = np.full((nk, NB), np.nan)
for (k, n), d in fd.items():
    if n < NB:
        Ed[k, n] = d['E_dft']; ki[k, n] = d['kin_ion']; VH[k, n] = d['V_H']
KIHr = np.array([[kih[k][b + 1] for b in range(NB)] for k in range(nk)])
err = np.abs((ki + VH) - KIHr)

TAG = os.environ.get("VAC_TAG", "")
np.savez(os.path.join(OUT, f"vacuum_weight{TAG}.npz"),
         f_vac=f_vac, f_cent=f_cent, err=err, Edft=Ed, kin_ion=ki, V_H=VH,
         KIH_ref=KIHr, rho_z_val=rho_z_val, vac=vac)

print("\n=== correlation ===")
print(f"  OVERALL |dVxc|: mean {err.mean():8.2f}  max {err.max():8.2f} eV "
      f"(N={err.size}); bands<{NVAL}: mean {err[:, :NVAL].mean():.3f} "
      f"max {err[:, :NVAL].max():.3f} eV")
print(f"  corr(|dVxc|, f_vac ) = {np.corrcoef(err.ravel(), f_vac.ravel())[0,1]:+.3f}")
print(f"  corr(|dVxc|, 1-f_cent) = {np.corrcoef(err.ravel(), 1-f_cent.ravel())[0,1]:+.3f}")
print(f"  corr(|dVxc|, E_dft ) = {np.corrcoef(err.ravel(), Ed.ravel())[0,1]:+.3f}")

print("\n=== binned by vacuum weight ===")
edges = [0, .05, .15, .30, .50, .70, 1.01]
print(f"{'f_vac bin':>14} {'N(k,n)':>8} {'|dVxc| mean':>12} {'max':>9} {'f_cent mean':>12} {'bands':>34}")
for a, bnd in zip(edges[:-1], edges[1:]):
    m = (f_vac >= a) & (f_vac < bnd)
    if not m.sum():
        continue
    bs = sorted(set(np.where(m)[1].tolist()))
    bstr = f"{bs[0]}..{bs[-1]} ({len(bs)})" if len(bs) > 6 else str(bs)
    print(f"  [{a:.2f},{bnd:.2f}) {m.sum():>8} {err[m].mean():>12.2f} {err[m].max():>9.2f} "
          f"{f_cent[m].mean():>12.3f} {bstr:>34}")

print("\n=== worst 15 (k,n) by |dVxc| ===")
print(f"{'k':>3} {'n':>4} {'Edft':>8} {'f_vac':>7} {'f_cent':>7} {'kin_ion':>9} {'V_H':>9} {'KIH_ref':>9} {'|dVxc|':>8}")
for k, b in zip(*np.unravel_index(np.argsort(-err.ravel())[:15], err.shape)):
    print(f"{k:>3} {b:>4} {Ed[k,b]:>8.2f} {f_vac[k,b]:>7.3f} {f_cent[k,b]:>7.3f} "
          f"{ki[k,b]:>9.2f} {VH[k,b]:>9.2f} {KIHr[k,b]:>9.2f} {err[k,b]:>8.2f}")

print("\n=== best 10 (k,n) by |dVxc| (control) ===")
for k, b in zip(*np.unravel_index(np.argsort(err.ravel())[:10], err.shape)):
    print(f"{k:>3} {b:>4} {Ed[k,b]:>8.2f} {f_vac[k,b]:>7.3f} {f_cent[k,b]:>7.3f} "
          f"{ki[k,b]:>9.2f} {VH[k,b]:>9.2f} {KIHr[k,b]:>9.2f} {err[k,b]:>8.2f}")

print("\n=== per-band mean over k ===")
print(f"{'b':>4} {'Edft':>8} {'f_vac':>7} {'f_cent':>7} {'|dVxc|':>8}")
for b in range(0, NB, 2):
    print(f"{b:>4} {np.nanmean(Ed[:,b]):>8.2f} {f_vac[:,b].mean():>7.3f} "
          f"{f_cent[:,b].mean():>7.3f} {err[:,b].mean():>8.2f}")
print("\nDONE vacuum_weight")
