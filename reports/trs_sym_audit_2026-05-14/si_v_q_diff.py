#!/usr/bin/env python3
"""Element-wise diff of Si V_q,μν between IBZ-cascade unfold and
direct-at-full-BZ construction.

Both runs share the SAME 432-centroid basis. Both share the SAME WFN
(sym, ntran=48, nrk=8). Difference: ζ-on-disk is at 8 IBZ q's
(cascade) vs all 64 q's (force-full-BZ). V_q,μν reconstructed from
ζ̃(G) · v(q+G) · ζ̃(G) per the LORRAX V_q kernel definition.

For the cascade path, we additionally apply the user-spec unfold:
    V_full[q1, μ, ν] = exp(2π i q_irr · (L_μ − L_ν)) · V_ibz[parent, α(μ), α(ν)]

Then diff against the direct V_q at the same q_full index.

This tests whether the discrepancy is in:
  - Cascade construction of V_q_ibz at 8 IBZ q's (compare to direct
    V_q at those same q's),
  - The unfold formula application (apply formula in numpy, compare
    to LORRAX's jit'd unfold_v_q).
  - The direct V_q construction at 64 full-BZ q's.
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'
import sys
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src')
import numpy as np
import h5py
from pathlib import Path

from file_io import WfnLoader
from centroid.orbit_syms import compute_centroid_sym_perm

PARENT = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14')
CASCADE = PARENT / 'run_sym_bgw_conv_2026-05-15'   # 8 IBZ q's on disk
DIRECT  = PARENT / 'run_sym_force_full_bz_2026-05-15'  # 64 full-BZ q's on disk

# Load WFN to get sym + kgrid + centroid info
wfn = WfnLoader(CASCADE / 'WFN.h5')
ntran = int(wfn.ntran)
mtrx = np.asarray(wfn.sym_matrices[:ntran], dtype=np.int64)
tau_raw = np.asarray(wfn.translations[:ntran], dtype=np.float64)
tau_frac = tau_raw / (2 * np.pi)
kgrid = np.asarray(wfn.kgrid, dtype=np.int64)
fft_grid = np.array([24, 24, 24])  # Si FFT grid

# Centroids
r_mu_frac = np.loadtxt(CASCADE / 'centroids_frac_432.txt')
r_mu_idx = np.rint(r_mu_frac * fft_grid).astype(np.int32)
n_mu = r_mu_frac.shape[0]
print(f'n_mu = {n_mu}, ntran = {ntran}, kgrid = {kgrid.tolist()}')

# Build (α, L) per BGW convention r' = inv(mtrx)·r + τ; user-spec inverse
# form: y_μ = mtrx · (r_μ − τ), decompose y_μ = x_{α(μ)} + L_μ.
sym_perm, L_table = compute_centroid_sym_perm(
    r_mu_idx, mtrx, tau_raw, fft_grid,
    validate=True, extend_trs=True)
print(f'sym_perm.shape = {sym_perm.shape}, L_table.shape = {L_table.shape}')

# Load q-mapping: full-BZ q_idx → (irr_parent, sym_idx)
# These are stored on the SymMaps object; rebuild
from common.symmetry_maps import SymMaps
sym = SymMaps(wfn)
irr_idx_q = np.asarray(sym.irr_idx_q)   # (n_q_full,)
sym_idx_q = np.asarray(sym.sym_idx_q)   # (n_q_full,)
q_irr_kgrid_int = np.asarray(sym.q_irr_kgrid_int)  # (n_q_ibz, 3)
n_q_full = int(np.prod(kgrid))
n_q_ibz = q_irr_kgrid_int.shape[0]
print(f'n_q_full = {n_q_full}, n_q_ibz = {n_q_ibz}')

# BGW q-wrap then divide by kgrid
kg_arr = kgrid.astype(np.float64)
q_irr_int = q_irr_kgrid_int.astype(np.float64)
q_irr_wrap = np.where(q_irr_int > kg_arr/2, q_irr_int - kg_arr, q_irr_int)
q_irr_frac = q_irr_wrap / kg_arr
print(f'q_irr_frac (first 3):\n{q_irr_frac[:3]}')

# Load ζ̃ for both paths
def load_zeta_and_gvecs(rundir, n_q_expected):
    zeta_path = rundir / 'tmp' / f'zeta_q.h5'
    if not zeta_path.exists():
        # search for any zeta file
        cand = list((rundir / 'tmp').glob('zeta*.h5'))
        if cand:
            zeta_path = cand[0]
    if not zeta_path.exists():
        raise FileNotFoundError(f"No zeta_q.h5 in {rundir / 'tmp'}")
    with h5py.File(zeta_path, 'r') as f:
        zeta = f['zeta_q_G'][:]   # (n_q, n_mu, ngkmax) complex
        gvec_comp = f['isdf_header/gvec_components'][:]  # (n_q, 3, ngkmax)
        ngk = f['isdf_header/ngk'][:]         # (n_q,) actual G-count per q
    assert zeta.shape[0] == n_q_expected, f'{zeta_path} expected {n_q_expected} q, got {zeta.shape[0]}'
    return zeta, gvec_comp, ngk, zeta_path

zeta_ibz, gvec_ibz, ngk_ibz, p1 = load_zeta_and_gvecs(CASCADE, n_q_ibz)
zeta_full, gvec_full, ngk_full, p2 = load_zeta_and_gvecs(DIRECT, n_q_full)
print(f'cascade ζ from {p1}: shape {zeta_ibz.shape}, ngk[0]={ngk_ibz[0]}')
print(f'direct  ζ from {p2}: shape {zeta_full.shape}, ngk[0]={ngk_full[0]}')

# Compute V_q,μν from ζ̃ at one q:
# V_q[μ,ν] = Σ_G conj(ζ̃[μ,G]) · v(q+G) · ζ̃[ν,G]
# v(q+G) for bare Coulomb 3D: 4π/|q+G|² in atomic units. For Si 3D bare-Coulomb cutoff = 25 Ry.
# But the Coulomb factor here is what LORRAX's kernel uses: v_q(q+G) factory.
# To avoid reimplementing, use the V_q dumps from prior compare scripts if present,
# OR just build a simple v factory locally with the same cutoff.

# Get bvec for Cartesian |q+G|²
avec = np.asarray(wfn.avec)
bvec = np.asarray(wfn.bvec)
print(f'avec:\n{avec}\nbvec:\n{bvec}')

def v_factor(q_frac, gvecs_int_q):
    """v(q+G) for 3D bare Coulomb, |q+G|^2 in cartesian.
    gvecs_int_q shape (3, ngk); q_frac shape (3,)."""
    qG_frac = q_frac[:, None] + gvecs_int_q   # (3, ngk)
    qG_cart = bvec.T @ qG_frac                  # (3, ngk) cartesian
    qG_sq = (qG_cart**2).sum(axis=0)            # (ngk,)
    # Bare 3D: 4π / |q+G|^2 in atomic units; we want consistency with LORRAX's V_q value.
    # 1/Ω comes from FT convention. Use the same constants as LORRAX:
    # 4π/|q+G|² · 1/Ω (Hartree per electron pair).
    cell_volume = float(np.abs(np.linalg.det(avec)))
    v = np.zeros_like(qG_sq)
    nz = qG_sq > 1e-14
    v[nz] = 4 * np.pi / qG_sq[nz] / cell_volume
    return v

def compute_V_at_q(zeta_q, gvec_q, ngk_q, q_frac):
    """V[μ,ν] from ζ̃ at q. Use ngk_q for the actual sphere; padded
    sentinel G's have zero zeta so they don't contribute."""
    ngk = int(ngk_q)
    z = zeta_q[:, :ngk]              # (n_mu, ngk)
    g = gvec_q[:, :ngk]              # (3, ngk)
    v = v_factor(q_frac, g)          # (ngk,)
    # V[μ,ν] = Σ_G conj(z[μ,G]) · v[G] · z[ν,G]
    return np.einsum('mg,g,ng->mn', np.conj(z), v, z, optimize=True)

# Build full-BZ q list (LORRAX canonical order)
kx, ky, kz = np.meshgrid(np.arange(kgrid[0]), np.arange(kgrid[1]),
                          np.arange(kgrid[2]), indexing='ij')
q_full_kg = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1).astype(np.int64)
q_full_int_wrap = np.where(q_full_kg.astype(float) > kg_arr/2,
                            q_full_kg.astype(float) - kg_arr, q_full_kg.astype(float))
q_full_frac = q_full_int_wrap / kg_arr

# Pick a few full-BZ q's that map to non-trivial sym (sym_idx > 0) AND test
# at sym=1 (a non-symmorphic op).
def has_sym_op(target_s):
    idx = np.where(sym_idx_q == target_s)[0]
    return int(idx[0]) if len(idx) else None

# Find a q_full with sym_idx=1 (non-symmorphic) and a non-Γ parent
print(f'\nsym_idx_q unique: {np.unique(sym_idx_q)}')
print(f'irr_idx_q sample: {irr_idx_q[:20]}')

target_q_full = None
for q in range(n_q_full):
    s = int(sym_idx_q[q])
    p = int(irr_idx_q[q])
    if s == 1 and p > 0:
        target_q_full = q
        break
if target_q_full is None:
    target_q_full = int(np.where(sym_idx_q > 0)[0][0])

s = int(sym_idx_q[target_q_full])
parent = int(irr_idx_q[target_q_full])
print(f'\nTesting at q_full={target_q_full}, parent_ibz={parent}, sym={s}')
print(f'  q_full_kgrid_int = {q_full_kg[target_q_full]}, frac (BGW-wrap) = {q_full_frac[target_q_full]}')
print(f'  q_irr_kgrid_int = {q_irr_kgrid_int[parent]}, frac (BGW-wrap) = {q_irr_frac[parent]}')
print(f'  mtrx[{s}] = {mtrx[s].tolist()}')
print(f'  tau_frac[{s}] = {tau_frac[s]}')

# Direct V_q at target_q_full
V_direct = compute_V_at_q(zeta_full[target_q_full],
                          gvec_full[target_q_full],
                          ngk_full[target_q_full],
                          q_full_frac[target_q_full])
print(f'\nV_direct shape {V_direct.shape}, |V|max = {np.abs(V_direct).max():.4e}')

# V_ibz at parent
V_ibz_p = compute_V_at_q(zeta_ibz[parent],
                         gvec_ibz[parent],
                         ngk_ibz[parent],
                         q_irr_frac[parent])
print(f'V_ibz_parent: |V|max = {np.abs(V_ibz_p).max():.4e}')

# Apply user-spec unfold: V_full[μ,ν] = exp(+2π i q · (L_μ−L_ν)) · V_ibz[α(μ), α(ν)]
# (inverse form, my LORRAX convention)
alpha = sym_perm[s]
L = L_table[s].astype(np.float64)
q_p = q_irr_frac[parent]
qL = L @ q_p
phase = np.exp(2j * np.pi * qL)

V_unfolded = phase[:, None] * V_ibz_p[np.ix_(alpha, alpha)] * phase.conj()[None, :]
print(f'V_unfolded: |V|max = {np.abs(V_unfolded).max():.4e}')

# Diff
err = np.abs(V_unfolded - V_direct)
print(f'\nmax |V_unfolded − V_direct| = {err.max():.4e}')
print(f'rel max = {err.max() / np.abs(V_direct).max():.3e}')
print(f'top 5 |err| (μ, ν, err):')
idx_flat = np.argsort(err.ravel())[-5:][::-1]
for k in idx_flat:
    mu, nu = np.unravel_index(k, err.shape)
    print(f'  μ={mu:3d} ν={nu:3d}: |err|={err[mu,nu]:.3e}, '
          f'unfolded={V_unfolded[mu,nu]:.4e}, direct={V_direct[mu,nu]:.4e}')

# Also test the FORWARD-form formula to compare:
# V_full[μ(α), μ(β)] = exp(-2π i q_full · (T_α − T_β)) · V_ibz[α, β]
# For each source centroid α, image μ(α) = (mtrx_forward_action(x_α + τ_q_irr_action_etc...)
# Forward direction: y_α = U·x_α + τ where U = inv(mtrx)
# (BGW r-action: r' = inv(mtrx)·r + τ.  So forward U = inv(mtrx).)
U_fwd = np.linalg.inv(mtrx[s].astype(np.float64))
y_fwd_alpha = (U_fwd @ r_mu_frac.T).T + tau_frac[s][None, :]   # (n_mu, 3)
x_target = y_fwd_alpha - np.floor(y_fwd_alpha)
T_alpha = np.rint(y_fwd_alpha - x_target).astype(np.int64)
# image centroid index μ(α)
INV = int(round(1e6))
keys = {tuple((np.rint(r * INV).astype(np.int64) % INV).tolist()): i for i, r in enumerate(r_mu_frac)}
mu_target = np.array([keys.get(tuple((np.rint(x * INV).astype(np.int64) % INV).tolist()), -1)
                       for x in x_target])
n_unmatched = int((mu_target < 0).sum())
print(f'\nForward form image lookup: {n_mu - n_unmatched}/{n_mu} hit')

if n_unmatched == 0:
    q1_frac = q_full_frac[target_q_full]
    qT = T_alpha.astype(np.float64) @ q1_frac
    phase_fwd = np.exp(-2j * np.pi * qT)   # exp(-2π i q1 · T_α)
    # V_q1[μ(α), μ(β)] = phase_fwd[α] · V_ibz[α, β] · conj(phase_fwd[β])
    # Build via target indexing: V_unfolded2[μ(α), μ(β)] = ...
    V_unfolded2 = np.zeros_like(V_direct)
    for a in range(n_mu):
        for b in range(n_mu):
            V_unfolded2[mu_target[a], mu_target[b]] = (
                phase_fwd[a] * V_ibz_p[a, b] * np.conj(phase_fwd[b]))
    err2 = np.abs(V_unfolded2 - V_direct)
    print(f'Forward-form: max |V − V_direct| = {err2.max():.4e}, rel = {err2.max()/np.abs(V_direct).max():.3e}')

# Sweep all 64 q's; report per-q rel error
print('\n=== Per-q sweep (inverse formula) ===')
print(f'{"q":>3} {"par":>4} {"s":>3}  q_full_frac          rel_err')
max_rel = 0.0; max_q = -1
for q_full in range(n_q_full):
    p = int(irr_idx_q[q_full])
    s = int(sym_idx_q[q_full])
    V_d = compute_V_at_q(zeta_full[q_full], gvec_full[q_full],
                          ngk_full[q_full], q_full_frac[q_full])
    V_i = compute_V_at_q(zeta_ibz[p], gvec_ibz[p], ngk_ibz[p],
                         q_irr_frac[p])
    alpha = sym_perm[s]
    L = L_table[s].astype(np.float64)
    qL = L @ q_irr_frac[p]
    phase = np.exp(2j * np.pi * qL)
    V_u = phase[:, None] * V_i[np.ix_(alpha, alpha)] * phase.conj()[None, :]
    err = np.abs(V_u - V_d).max() / max(np.abs(V_d).max(), 1.0)
    if err > max_rel:
        max_rel = err; max_q = q_full
    print(f'{q_full:>3} {p:>4} {s:>3}  {np.round(q_full_frac[q_full],3).tolist()!s:>20}  {err:.3e}')

print(f'\nWorst q: {max_q}, rel err {max_rel:.3e}')
