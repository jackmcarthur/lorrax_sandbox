"""Close a centroid set under the crystal spatial point group.

Standalone experiment for the V_H C3-symmetry test.  The WFN only stores
ntran=2 = {E, sigma_h}; the crystal's C3/C3v is NOT in the WFN sym list.
Here we regenerate the crystal spatial point group from the lattice metric
+ atomic basis (symmorphic, tau=0 for MoS2) and close the centroid set
under it, so the ISDF centroid quadrature becomes point-group symmetric.

Usage:
  python close_centroids.py IN_CENTROIDS OUT_CENTROIDS MODE
    MODE = 'c3'   -> close under {E, C3, C3^2} only (in-plane; NOT sigma_h;
                     keeps the IBZ cascade OFF so V_H test is isolated)
    MODE = 'full' -> close under the full crystal spatial point group
"""
import sys
import numpy as np

IN, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]

# --- lattice (angstrom) and atomic basis (crystal), from nscf.in ---
avec = np.array([[3.164292, 0.0, 0.0],
                 [-1.582142, 2.740357, 0.0],
                 [0.0, 0.0, 12.0]])                      # rows = a1,a2,a3
atoms = [("Mo", (2.0/3.0, 1.0/3.0, 0.0)),
         ("S", (1.0/3.0, 2.0/3.0, 0.131928)),
         ("S", (1.0/3.0, 2.0/3.0, -0.131928))]
FFT = np.array([24, 24, 80])

G = avec @ avec.T          # metric = C^T C with C = avec^T (cols = lattice vecs)

# --- enumerate integer point ops M with M^T G M = G, entries in {-1,0,1} ---
vals = [-1, 0, 1]
cand = []
import itertools
# in-plane 2x2 block over {-1,0,1}, z-block = diag(+/-1), no xy-z mixing
for a, b, c, d in itertools.product(vals, repeat=4):
    for zz in (1, -1):
        M = np.array([[a, b, 0], [c, d, 0], [0, 0, zz]], dtype=np.int64)
        if np.array_equal(np.rint(M.T @ G @ M).astype(np.int64), np.rint(G).astype(np.int64)):
            cand.append(M)
cand = np.array(cand)
print(f"holohedry candidates (metric-preserving): {len(cand)}")

# --- filter to crystal ops: M must map atoms->atoms (same species), tau=0 ---
def maps_atoms(M):
    for sp, p in atoms:
        img = (M @ np.array(p)) % 1.0
        ok = False
        for sp2, p2 in atoms:
            if sp2 != sp:
                continue
            d = (img - np.array(p2)) % 1.0
            d = np.minimum(d, 1.0 - d)
            if np.all(d < 1e-4):
                ok = True
                break
        if not ok:
            return False
    return True

crystal = [M for M in cand if maps_atoms(M)]
print(f"crystal spatial point group order: {len(crystal)}")
for M in crystal:
    print("  ", M[0], M[1], M[2], " det=", int(round(np.linalg.det(M))))

if MODE == "c3":
    C3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=np.int64)
    grp = [np.eye(3, dtype=np.int64), C3, C3 @ C3]
    assert maps_atoms(C3), "C3 not a crystal sym!"
    print("Using C3-only group (order 3), sigma_h EXCLUDED (cascade stays off).")
else:
    grp = crystal
    print(f"Using full crystal point group (order {len(grp)}).")

# --- load centroids, close under group, snap to FFT grid, dedupe ---
cent = np.loadtxt(IN)  # (N,3) fractional
print(f"input centroids: {cent.shape[0]}")
imgs = []
for M in grp:
    im = (cent @ M.T) % 1.0        # frac row-vector action r' = M r  -> row: r @ M^T
    imgs.append(im)
allc = np.concatenate(imgs, axis=0)
idx = np.rint(allc * FFT).astype(np.int64) % FFT
key = idx[:, 0] * (FFT[1] * FFT[2]) + idx[:, 1] * FFT[2] + idx[:, 2]
_, first = np.unique(key, return_index=True)
uidx = idx[np.sort(first)]
uc = uidx.astype(float) / FFT
print(f"closed+unique centroids: {uc.shape[0]}")

np.savetxt(OUT, uc, fmt="%.6f", delimiter=" ",
           header=f"x y z (closed under {MODE} crystal point group, snapped {tuple(FFT)}, {uc.shape[0]} unique)",
           comments="# ")
print(f"wrote {OUT}")
