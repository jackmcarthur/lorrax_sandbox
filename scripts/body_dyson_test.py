"""Test: body-only Dyson on BGW eps0mat vs LORRAX W_qmunu.

Hypothesis: LORRAX zeros V(G=0)=0 BEFORE Dyson → effectively body-Dyson.
BGW eps0mat has full Dyson done, then we zero head/wings of result.

This reconstructs chi₀ from BGW's eps⁻¹, zeros its G=0 row/col, re-does the
Dyson on the body subspace, projects to (μ,ν), and compares to LORRAX.
"""
import sys
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

import numpy as np
import h5py

P = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band'

# Load BGW eps⁻¹ at both frequencies (q=0, full G-space)
with h5py.File(f'{P}/03_bgw_full_freq_2pt/eps0mat.h5', 'r') as f:
    nmtx = int(f['eps_header/gspace/nmtx'][0])
    print(f'eps0mat nmtx = {nmtx}')
    # Read eps⁻¹ (real/imag) for q=0, both frequencies
    M = np.asarray(f['mats/matrix'])  # (nq=1, nmat=1, nfreq=2, nmtx, nmtx, 2)
    ek0_raw = np.asarray(f['eps_header/gspace/ekin']).ravel()  # |q+G|² for q=0
    ek0 = np.asarray(ek0_raw[:nmtx], dtype=np.float64)
    g0_idx = int(np.argmin(ek0))
    print(f'g0_idx = {g0_idx}, ekin[g0] = {ek0[g0_idx]:.6e}')
    epsinv0 = M[0,0,0,:nmtx,:nmtx,0] + 1j*M[0,0,0,:nmtx,:nmtx,1]
    epsinvp = M[0,0,1,:nmtx,:nmtx,0] + 1j*M[0,0,1,:nmtx,:nmtx,1]
    vcoul = np.asarray(f['eps_header/gspace/vcoul'])[:nmtx]  # 8π/|q+G|² in Ry·a.u.

# v(q+G) for q=0.00025 sample point. v(G=0) is the small-q value.
v_g_raw = np.asarray(vcoul, dtype=np.float64).ravel()[:nmtx]
v_g = np.asarray(v_g_raw, dtype=np.complex128)
print(f'v(G=0) = {float(v_g[g0_idx].real):.4e}  (BGW q=0.00025 limit)')
mask_off = np.arange(nmtx) != g0_idx
print(f'v(G!=0) typical = {float(np.median(np.abs(v_g[mask_off]))):.4e}')

# Recover chi₀_GG'(ω) = (δ_GG' - eps_GG'(ω))/v(G)  ...  (V is diagonal, so eps = δ - V·χ)
def recover_chi0(epsinv, v):
    eps = np.linalg.inv(epsinv)
    chi0 = (np.eye(epsinv.shape[0]) - eps) / v[:, None]   # (1-eps)_GG' / v(G)
    return chi0

chi0_0 = recover_chi0(epsinv0, v_g)
chi0_p = recover_chi0(epsinvp, v_g)
print(f'fro(chi0_0)  = {np.linalg.norm(chi0_0,"fro"):.4e}')
print(f'fro(chi0_p)  = {np.linalg.norm(chi0_p,"fro"):.4e}')

# Method A (FULL DYSON, zero W's head/wings AFTER): equivalent to current script behaviour
def method_full_then_zero(epsinv, v):
    W = epsinv * v[None, :]  # eps⁻¹·diag(v)
    W = W.copy()
    W[g0_idx, :] = 0; W[:, g0_idx] = 0
    return W

# Method B (BODY DYSON: zero V(G=0) and chi₀'s G=0 row/col BEFORE inverting): mimic LORRAX
def method_body_dyson(chi0, v):
    v_body = v.copy(); v_body[g0_idx] = 0   # V(G=0) = 0, à la LORRAX
    chi_body = chi0.copy()
    chi_body[g0_idx, :] = 0; chi_body[:, g0_idx] = 0
    # W = (I - V·χ)⁻¹·V with V diagonal: V·χ_GG' = v(G)·chi_GG'
    Vchi = v_body[:, None] * chi_body
    W = np.linalg.inv(np.eye(chi_body.shape[0]) - Vchi) @ np.diag(v_body)
    return W

W0_full = method_full_then_zero(epsinv0, v_g)
Wp_full = method_full_then_zero(epsinvp, v_g)
W0_body = method_body_dyson(chi0_0, v_g)
Wp_body = method_body_dyson(chi0_p, v_g)

print()
print(f'Method A (current script: full Dyson, zero W head/wings AFTER):')
print(f'  fro(W0)  = {np.linalg.norm(W0_full,"fro"):.4e}')
print(f'  fro(Wp)  = {np.linalg.norm(Wp_full,"fro"):.4e}')
print(f'Method B (body Dyson: zero V(G=0) + chi₀ wings BEFORE invert):')
print(f'  fro(W0)  = {np.linalg.norm(W0_body,"fro"):.4e}')
print(f'  fro(Wp)  = {np.linalg.norm(Wp_body,"fro"):.4e}')

print()
print(f'Method-A vs Method-B (rel-fro of difference, body part only):')
mask_body = np.ones(W0_full.shape, dtype=bool)
mask_body[g0_idx, :] = False; mask_body[:, g0_idx] = False
def rel_fro_body(A, B):
    return np.linalg.norm((A-B)[mask_body])/np.linalg.norm(A[mask_body])
print(f'  W(0):  rel-fro(A-B) / fro(A) = {rel_fro_body(W0_full, W0_body):.4e}')
print(f'  W(p):  rel-fro(A-B) / fro(A) = {rel_fro_body(Wp_full, Wp_body):.4e}')

# Now project both to (μ,ν) and compare to LORRAX W_qmunu
# Need ζ_μ(G) in eps-order. Reuse projection logic from script.
print()
print(f'Projecting both methods to (μ,ν) basis and comparing to LORRAX W_qmunu...')

# Find centroid file and load zeta
zeta_path = f'{P}/1A_hl_bare25/tmp/zeta_q.h5'
import os
if not os.path.exists(zeta_path):
    print(f'  zeta_q.h5 not found at {zeta_path}, looking elsewhere...')
    import glob
    cands = glob.glob(f'{P}/1A_hl_bare25/tmp/zeta*.h5')
    print(f'  found: {cands}')
    if cands: zeta_path = cands[0]

with h5py.File(zeta_path, 'r') as f:
    zeta_q = np.asarray(f['zeta_q'][...], dtype=np.complex128)  # (nk_flat=64, nrtot=13824, nmu=480)
print(f'  zeta_q shape: {zeta_q.shape}')

# Take q=Γ (ik=0) and FFT real-space ζ to G-space
# zeta_q[0] has shape (nrtot, nmu) → reshape to (Nx, Ny, Nz, nmu) and ifft
zeta_r = zeta_q[0]  # (nrtot, nmu)
nrtot, nmu = zeta_r.shape
# Find FFT shape
with h5py.File(f'{P}/1A_hl_bare25/WFN.h5', 'r') as f:
    fft_grid = tuple(int(x) for x in np.asarray(f['mf_header/gspace/FFTgrid'][...]).ravel()[:3])
print(f'  FFT grid: {fft_grid}, nrtot match: {np.prod(fft_grid)} == {nrtot}')

zeta_box = zeta_r.reshape(*fft_grid, nmu)
zeta_G = np.fft.fftn(zeta_box, axes=(0,1,2), norm='ortho')  # (Nx,Ny,Nz,nmu)
zeta_G = zeta_G.reshape(-1, nmu)  # (Nfft, nmu)

# Map eps-order G to FFT indices
with h5py.File(f'{P}/03_bgw_full_freq_2pt/eps0mat.h5', 'r') as f:
    g_eps = np.asarray(f['eps_header/gspace/gind_eps2rho'][...]).ravel()[:nmtx] - 1  # 1-indexed
    # gind_eps2rho gives index into the rho FFT order
    # We need actual G-vectors. Get them from rho gvecs
with h5py.File(f'{P}/1A_hl_bare25/WFN.h5', 'r') as f:
    g_rho = np.asarray(f['mf_header/gspace/components'][...])  # (3, ng_rho) maybe
print(f'  g_rho shape: {g_rho.shape}')
if g_rho.shape[0] == 3:
    g_rho = g_rho.T  # now (ng_rho, 3)
g_eps_vec = g_rho[g_eps]  # (nmtx, 3) - eps-order G vectors

# Map eps-order G to FFT-grid linear index
def g_to_fft_lin(gvec, fft_shape):
    # standard FFT layout: G can be negative; FFT index = G mod N
    Nx, Ny, Nz = fft_shape
    ix = gvec[:, 0] % Nx
    iy = gvec[:, 1] % Ny
    iz = gvec[:, 2] % Nz
    return ix * (Ny * Nz) + iy * Nz + iz

eps_fft_idx = g_to_fft_lin(g_eps_vec, fft_grid)
zeta_eps = zeta_G[eps_fft_idx]  # (nmtx, nmu)
print(f'  zeta_eps shape: {zeta_eps.shape}')

# Project Method A and Method B to (μ,ν) basis, body-only mask
def project(W_gg, zeta_eps_):
    return np.einsum('Gm,GH,Hn->mn', np.conj(zeta_eps_), W_gg, zeta_eps_)

# For body-only projection, zero G=g0_idx row/col first
def proj_body(W_gg, zeta_eps_):
    Wb = W_gg.copy()
    Wb[g0_idx, :] = 0
    Wb[:, g0_idx] = 0
    return project(Wb, zeta_eps_)

W0_full_munu = proj_body(W0_full, zeta_eps)
W0_body_munu = proj_body(W0_body, zeta_eps)
Wp_full_munu = proj_body(Wp_full, zeta_eps)
Wp_body_munu = proj_body(Wp_body, zeta_eps)

# Load LORRAX W_qmunu at q=Γ (HL run)
with h5py.File(f'{P}/1A_hl_bare25/tmp/isdf_tensors_480.h5', 'r') as f:
    W0_lor = np.asarray(f['W0_qmunu'][0,0,0,0,0,0,:,:], dtype=np.complex128)
with h5py.File(f'{P}/1A_hl_bare25/tmp/wprobe_qmunu.h5', 'r') as f:
    Wp_lor = np.asarray(f['Wprobe_qmunu'][0,0,0,0,0,0,:,:], dtype=np.complex128)

print()
print(f'(μ,ν)-basis comparison (q=Γ, body-only):')
print(f'  fro(W0_lor) = {np.linalg.norm(W0_lor,"fro"):.4e}')
print(f'  fro(Wp_lor) = {np.linalg.norm(Wp_lor,"fro"):.4e}')
print(f'  fro(W0_full_munu) = {np.linalg.norm(W0_full_munu,"fro"):.4e}')
print(f'  fro(W0_body_munu) = {np.linalg.norm(W0_body_munu,"fro"):.4e}')
print()
print(f'LORRAX vs Method-A (full Dyson then zero) at ω=0:  rel-fro = {np.linalg.norm(W0_lor - W0_full_munu)/np.linalg.norm(W0_lor):.4e}')
print(f'LORRAX vs Method-B (body Dyson)         at ω=0:  rel-fro = {np.linalg.norm(W0_lor - W0_body_munu)/np.linalg.norm(W0_lor):.4e}')
print(f'LORRAX vs Method-A                      at ω=p:  rel-fro = {np.linalg.norm(Wp_lor - Wp_full_munu)/np.linalg.norm(Wp_lor):.4e}')
print(f'LORRAX vs Method-B                      at ω=p:  rel-fro = {np.linalg.norm(Wp_lor - Wp_body_munu)/np.linalg.norm(Wp_lor):.4e}')
