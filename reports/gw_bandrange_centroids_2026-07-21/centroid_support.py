"""Vacuum support of a centroid SET (no GW run needed).

Same |psi|^2 / vacuum-plane machinery as
reports/gw_conduction_postfix_2026-07-21/vacuum_weight.py, but evaluated for
several centroid files at once so the k-means-weight sweep can be screened
before spending GW cycles:

  f_vac[k,n]  = fraction of |psi_nk|^2 in the vacuum planes   (set-independent)
  f_cent[k,n] = (Nr/Ncent) * sum_mu |psi_nk(r_mu)|^2          (per centroid set)
                -> 1.0 means the centroid set samples this state's norm
                   correctly; ~0 means the state is invisible to the ISDF
                   quadrature, >>1 means it is over-counted.
  n_vac_cent  = how many centroids sit in the vacuum planes.
"""
import os
import glob
import numpy as np
import h5py

RUN = os.environ.get(
    "CS_RUN",
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/"
    "04_lorrax_gw_bandrange_2026-07-21/kmeans_wd")
OUT = os.environ.get(
    "CS_OUT",
    "/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_bandrange_centroids_2026-07-21")
NB = int(os.environ.get("CS_NB", "100"))
NVAL = 26

files = sorted(glob.glob(os.path.join(RUN, "cent_*.txt")))
tags = [os.path.basename(f)[5:-4] for f in files]
print("centroid sets:", tags, flush=True)

f = h5py.File(os.path.join(RUN, "WFN.h5"), "r")
ngk = f["mf_header/kpoints/ngk"][()]
fftg = f["mf_header/gspace/FFTgrid"][()]
gvecs = f["wfns/gvecs"][()]
nk = len(ngk)
off = np.concatenate([[0], np.cumsum(ngk)])
Nx, Ny, Nz = (int(v) for v in fftg)
Nr = Nx * Ny * Nz
print(f"nk={nk} FFTgrid={fftg}", flush=True)

ci = {}
for tag, path in zip(tags, files):
    c = np.loadtxt(path)
    c = c.reshape(1, -1) if c.ndim == 1 else c
    ci[tag] = np.rint(c[:, :3] * np.array([Nx, Ny, Nz])).astype(int) % np.array(
        [Nx, Ny, Nz])
    print(f"  {tag}: {ci[tag].shape[0]} centroids, "
          f"{len(set(map(tuple, ci[tag])))} unique cells", flush=True)

pz_all = np.zeros((nk, NB, Nz))
f_cent = {t: np.zeros((nk, NB)) for t in tags}
rho_z_val = np.zeros(Nz)

for k in range(nk):
    lo, hi = off[k], off[k + 1]
    gk = gvecs[lo:hi]
    ix, iy, iz = gk[:, 0] % Nx, gk[:, 1] % Ny, gk[:, 2] % Nz
    co = f["wfns/coeffs"][:NB, :, lo:hi, :]
    cc = co[..., 0] + 1j * co[..., 1]
    for b in range(NB):
        rho = np.zeros((Nx, Ny, Nz))
        for s in range(cc.shape[1]):
            g = np.zeros((Nx, Ny, Nz), dtype=complex)
            np.add.at(g, (ix, iy, iz), cc[b, s])
            r = np.fft.ifftn(g) * Nr
            rho += np.abs(r) ** 2
        rho /= rho.sum()
        pz_all[k, b] = rho.sum(axis=(0, 1))
        if b < NVAL:
            rho_z_val += pz_all[k, b]
        for t in tags:
            f_cent[t][k, b] = rho[ci[t][:, 0], ci[t][:, 1], ci[t][:, 2]].sum() * (
                Nr / ci[t].shape[0])
    if k % 9 == 0:
        print(f"  k={k} done", flush=True)
f.close()

rho_z_val /= rho_z_val.sum()
vac = rho_z_val < 0.002 * rho_z_val.max()
print(f"\nvacuum planes: {vac.sum()}/{Nz}", flush=True)
f_vac = pz_all[:, :, vac].sum(axis=2)

np.savez(os.path.join(OUT, "centroid_support.npz"),
         f_vac=f_vac, vac=vac, rho_z_val=rho_z_val,
         **{f"f_cent_{t}": f_cent[t] for t in tags})

print("\n=== centroids in the vacuum planes ===")
print(f"{'set':>8} {'N_cent':>7} {'N_vac':>7} {'%vac':>7} {'z-span(frac)':>16}")
for t in tags:
    inv = vac[ci[t][:, 2]]
    zs = ci[t][:, 2] / Nz
    print(f"{t:>8} {ci[t].shape[0]:>7} {inv.sum():>7} "
          f"{100.0*inv.mean():>6.1f}% {zs.min():>7.3f}..{zs.max():<7.3f}")

print("\n=== f_cent (centroid Riemann estimate of the state norm; 1.0 = ideal) ===")
edges = [0, .05, .15, .30, .50, 1.01]
hdr = f"{'f_vac bin':>14} {'N(k,n)':>8}" + "".join(f"{t:>10}" for t in tags)
print(hdr)
for a, b in zip(edges[:-1], edges[1:]):
    m = (f_vac >= a) & (f_vac < b)
    if not m.sum():
        continue
    row = f"  [{a:.2f},{b:.2f}) {m.sum():>8}"
    for t in tags:
        row += f"{f_cent[t][m].mean():>10.3f}"
    print(row)

print("\n=== per-band mean f_cent over k ===")
print(f"{'b':>4} {'f_vac':>7}" + "".join(f"{t:>10}" for t in tags))
for b in range(0, NB, 5):
    print(f"{b:>4} {f_vac[:, b].mean():>7.3f}"
          + "".join(f"{f_cent[t][:, b].mean():>10.3f}" for t in tags))
print("\nDONE centroid_support")
