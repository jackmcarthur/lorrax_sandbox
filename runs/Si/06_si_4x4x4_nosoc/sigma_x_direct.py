"""Direct G-space Σ_X(n, k=Γ) using LORRAX's BGW vcoul reader.

  Σ^X_n(Γ) = -(1/N_k) Σ_q Σ_{m∈occ} Σ_G |M_{nm}(Γ,q,G)|² · v_BGW(q,G)

with v_BGW from BGW's MC-averaged vcoul.dat (the same file LORRAX consumes
when use_bgw_vcoul=true). Mirrors the structure of
`misc/archived_tests/cohsex_noisdf.py:get_sigma_x_exact`, but for 3D and
using `file_io.read_bgw_vcoul`.

Pair-density convention follows the reference:
    psi_rtot[r] = N_grid * ifftn(c_box)[r]   (i.e. unnormalized u(r))
    psi_l = ψ_v(k-q, r)*  (conjugated for left)
    psi_r = ψ_n(k, r)
    M_{vn}(k, q, G) = (1/N_grid) * fftn(ψ_l × ψ_r)[G]
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys, numpy as np
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")

from runtime import set_default_env; set_default_env()
from file_io import WFNReader
from file_io.read_bgw_vcoul import read_bgw_vcoul, fill_v_grid_for_q
from common.symmetry_maps import SymMaps

WFN_PATH = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/nscf/WFN.h5"
VCOUL_PATH = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/D_bgw_cohsex/vcoul"
N_VAL = 8
N_SIGMA = 16

# ─── wfns ───────────────────────────────────────────────────────────
wfn = WFNReader(WFN_PATH)
sym = SymMaps(wfn)
nx, ny, nz = (int(v) for v in wfn.fft_grid)
N_grid = nx * ny * nz
N_k = sym.nk_tot
nspinor = wfn.nspinor
V_cell = float(wfn.cell_volume)
print(f"FFT grid ({nx},{ny},{nz})  N_k={N_k}  V_cell={V_cell:.4f}  nspinor={nspinor}")

nb = N_VAL + N_SIGMA
print(f"Loading ψ on FFT box (G-space) for {N_k} k × {nb} bands...")
# Store ψ in r-space directly using the cohsex_noisdf convention:
#   psi_l[k] = conj(N * ifftn(c_box))   for left side
#   psi_r[k] = N * ifftn(c_box)          for right side
psi_l = np.zeros((N_k, nb, nspinor, nx, ny, nz), dtype=np.complex128)
psi_r = np.zeros((N_k, nb, nspinor, nx, ny, nz), dtype=np.complex128)
for ik in range(N_k):
    gv = np.asarray(sym.get_gvecs_kfull(wfn, ik))
    cnk = sym.get_cnk_fullzone_batch(wfn, np.arange(nb), ik)   # (nb, ns, ngk)
    box = np.zeros((nb, nspinor, nx, ny, nz), dtype=np.complex128)
    box[:, :, gv[:, 0], gv[:, 1], gv[:, 2]] = cnk
    # ifftn on FFT-box axes; multiply by N_grid → unnormalized u(r).
    u_r = np.fft.ifftn(box, axes=(2, 3, 4)) * N_grid
    psi_r[ik] = u_r
    psi_l[ik] = np.conj(u_r)
kpts_full = np.asarray(sym.unfolded_kpts)

# ─── BGW vcoul ──────────────────────────────────────────────────────
table = read_bgw_vcoul(VCOUL_PATH)
print(f"BGW vcoul: {len(table.q_fracs)} unique q's stored")

# ─── Σ_X loop at k = Γ ──────────────────────────────────────────────
ik_gamma = int(np.argmin(np.linalg.norm(kpts_full, axis=1)))
assert np.linalg.norm(kpts_full[ik_gamma]) < 1e-8

ry2ev = 13.605693122994
sig_x_diag = np.zeros(N_SIGMA, dtype=np.float64)

# Per-q breakdown for vbm
sig_x_per_q_vbm = np.zeros(N_k, dtype=np.float64)

for iq in range(N_k):
    qfrac = kpts_full[iq]
    # k-q for k = Γ: -q in ext zone, find first-BZ representative k'
    target = (-qfrac) % 1.0
    diffs = np.linalg.norm(((kpts_full - target[None, :] + 0.5) % 1.0) - 0.5, axis=1)
    ik_kmq = int(np.argmin(diffs))
    assert diffs[ik_kmq] < 1e-6
    # G_unwrap: k' = k - q + G_unwrap. We stored ψ at k', so the pair density
    # gives M at G + G_unwrap relative to the physical convention.
    G_unwrap = np.rint(kpts_full[ik_kmq] - (-qfrac)).astype(int)

    # v(q+G) on the FFT box (Miller-indexed, signed, wrapped to [0,N))
    # use the q at k_outer - k_inner = Γ - k_inner = -k_inner = qfrac (since iq is k_inner)
    # NB: convention question — qfrac IS the q-vector going from outer Γ to inner k_inner = -qfrac
    # so we look up vcoul at qfrac itself.
    v_box = fill_v_grid_for_q(
        table, qfrac, fft_grid=(nx, ny, nz),
        cell_volume=V_cell, sym_mats_k=sym.sym_mats_k,
    )
    # Re-instate the head (fill_v_grid_for_q zeros [0,0,0] at q=0).
    if np.linalg.norm(qfrac) < 1e-8:
        iq_table, _, _ = table.find_q_index(qfrac, sym_mats_k=sym.sym_mats_k)
        G_miller = table.G_miller_per_q[iq_table]
        vcoul_q = table.vcoul_per_q[iq_table]
        zero_idx = np.where((G_miller == 0).all(axis=1))[0]
        if len(zero_idx):
            v_box[0, 0, 0] = vcoul_q[zero_idx[0]] / V_cell

    # Sanity: check whether v_box is "smooth" or has MC-averaged values vs naive
    if iq == 0:
        # Compare v_box at G=(1,0,0) etc. to naive 8π/|q+G|²
        bvec_cart = float(wfn.blat) * np.asarray(wfn.bvec)
        for Gtest in [(1,0,0), (0,0,1), (-1,0,0), (1,1,1)]:
            G_cart = (qfrac + np.array(Gtest)) @ bvec_cart
            naive = 8.0 * np.pi / np.sum(G_cart**2) / V_cell
            stored = v_box[Gtest[0]%nx, Gtest[1]%ny, Gtest[2]%nz]
            print(f"  q=Γ G={Gtest}: stored={stored:.6e}  naive_geom/V={naive:.6e}  "
                  f"ratio={stored/naive:.6f}")

    # Pair density u_v*(k-q, r) × u_n(Γ, r) summed over spinor
    pair = np.einsum('vsxyz,nsxyz->vnxyz',
                     psi_l[ik_kmq, :N_VAL], psi_r[ik_gamma, :N_SIGMA],
                     optimize=True)
    # M_{vn}(q, G) = fftn(pair) / N_grid
    M_box = np.fft.fftn(pair, axes=(2, 3, 4)) / N_grid     # (N_VAL, N_SIGMA, nx, ny, nz)

    # Roll v_box by +G_unwrap so v_box[G_eff] aligns with M_box[G_eff = G_phys + G_unwrap].
    v_box_rolled = np.roll(v_box,
                           shift=(G_unwrap[0], G_unwrap[1], G_unwrap[2]),
                           axis=(0, 1, 2))
    # Contraction: Σ_G |M|² · v_box_rolled(G).  v_box has 1/V_cell already;
    # the 1/N_k is the only outer factor.
    contrib_n = -np.einsum('vnxyz,vnxyz,xyz->n',
                            M_box.conj(), M_box, v_box_rolled,
                            optimize=True).real / N_k
    sig_x_diag += contrib_n
    sig_x_per_q_vbm[iq] = contrib_n[0] * ry2ev

    if iq == 0:
        # Sanity at q=0
        M00 = M_box[:, :, 0, 0, 0]
        # M_{vn}(0, 0) should be δ_vn × spinor-trace = 1 along v=n diagonal,
        # 0 otherwise. (Pair convention: ψ_v_conj(Γ) × ψ_n(Γ); their G=0 component
        # is ⟨v|n⟩ = δ_vn.)
        print(f"  q=Γ check: |M_{{vv}}(0,0)| diag (v=n=1..{N_VAL}) ≈ {np.abs(np.diag(M00))[:N_VAL]}")
        print(f"  q=Γ check: vbm contrib = {contrib_n[0]*ry2ev:+.5f} eV")

sig_x_eV = sig_x_diag * ry2ev
print(f"\nDirect G-space Σ_X (eV) at k=Γ — using BGW vcoul (head AND body MC-averaged):")
for n in range(N_SIGMA):
    print(f"  n={n+1:2d}  Σ_X = {sig_x_eV[n]:+.5f}")

# Per-q breakdown for vbm
print(f"\nPer-q Σ_X(vbm=1) breakdown (eV):")
print(f"  q=Γ:     {sig_x_per_q_vbm[0]:+.5f}")
print(f"  q≠Γ sum: {np.sum(sig_x_per_q_vbm[1:]):+.5f}")
print(f"  total:   {sig_x_per_q_vbm.sum():+.5f}")
print(f"\n  by IBZ orbit (q-magnitude):")
qmags = np.linalg.norm(((kpts_full + 0.5) % 1.0) - 0.5, axis=1)
unique_mags, idx = np.unique(np.round(qmags, 6), return_inverse=True)
for um, i_um in zip(unique_mags, range(len(unique_mags))):
    mask = (idx == i_um)
    n_in_orbit = int(mask.sum())
    sum_orbit = float(sig_x_per_q_vbm[mask].sum())
    print(f"    |q|={um:.4f}  multiplicity={n_in_orbit}  sum={sum_orbit:+.5f}  "
          f"avg={sum_orbit/n_in_orbit:+.5f}")
