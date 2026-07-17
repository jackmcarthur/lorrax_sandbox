"""Stage 1+2 of the ingredient-interpolation falloff study (READ-ONLY numpy).

Owner hypothesis (arbitrary_q_bse.md counter-argument): LORRAX's ISDF fit
solves  C_q ζ_q = Z_q  per momentum q.  Even though ζ_q = C_q^{-1} Z_q is NOT
interpolable across q (inverse of a band-limited object; and cond(C)~3.6e9),
the *ingredients* C_q and Z_q ARE interpolable because their q->R Fourier
images C_R, Z_R have Green's-function-like (superalgebraic) spatial falloff.
The 3x3 master-ζ test failed only because the grid was too coarse to resolve
that falloff.

This stage measures the falloff of C_R (the CCT Gram) — the harder-conditioned
ingredient — and does a leave-one-out Fourier interpolation of C_q on the
existing grids.  No ζ needed here (Stage 3 does the full V_q test).

C_q is rebuilt from psi_full_y EXACTLY as gw.isdf_fitting / isdf.core.
c_q_from_psi_sm builds it (verified per-element against the source):

    P[k, a, ν, μ, b] = Σ_n conj(ψ_{n,k,a}(r_μ)) ψ_{n,k,b}(r_ν)        # 'kmna,knbr->karmb'
    P_R  = ifftn_k(P, norm='forward')                                  # lattice-R image
    C_R[ν,μ] = Σ_{a,b} conj(P_R)[a,ν,μ,b] · P_R[a,ν,μ,b]               # charge γ̃^0 spin trace
    C_q  = fftn_k(C_R, norm='forward');  transpose last two -> C_q[μ,ν]

For every MoS2/Si config here band_range_left == band_range_right == (0, nband)
(b1=nelec-nval=0 and b3=nelec+ncond=nband), so P_l == P_r == the all-band pair
density and C_q is Hermitian PSD.
"""
import sys, h5py, numpy as np

np.set_printoptions(linewidth=160, suppress=False)

restart = sys.argv[1]                      # isdf_tensors_*.h5
label   = sys.argv[2]                      # e.g. MoS2_3x3
zeta_for_meta = sys.argv[3] if len(sys.argv) > 3 else ""   # optional zeta_q.h5 for adot

kgrid_override = sys.argv[4] if len(sys.argv) > 4 else ""   # "nx,ny,nz" for legacy restarts
with h5py.File(restart, "r") as f:
    psi   = f["psi_full_y"][()]            # (nk, nb, ns, n_mu) un-conjugated ψ
    if "kgrid" in f:
        kgrid = f["kgrid"][()].astype(int)                 # (3,)
    elif kgrid_override:
        kgrid = np.array([int(x) for x in kgrid_override.split(",")])
    else:
        # legacy 8-D V_qmunu (1,npol,npol,nkx,nky,nkz,μ,ν) carries the kgrid
        vs = f["V_qmunu"].shape
        kgrid = np.array([vs[3], vs[4], vs[5]])
nk, nb, ns, n_mu = psi.shape
nkx, nky, nkz = [int(x) for x in kgrid]
nq = nkx*nky*nkz
assert nk == nq, f"nk={nk} != nq={nq}"
print(f"[{label}] nk={nk} nb={nb} ns={ns} n_mu={n_mu} kgrid={nkx}x{nky}x{nkz} nq={nq}")

# real-space metric adot (Bohr^2) for |R|; reuse from a same-cell zeta if given
adot = None
if zeta_for_meta:
    with h5py.File(zeta_for_meta, "r") as f:
        adot = f["mf_header/crystal/adot"][()]
if adot is None:
    adot = np.eye(3)   # fall back to fractional-lattice distance
adot = np.asarray(adot, dtype=np.float64)

# ---------------------------------------------------------------------------
# Rebuild C_q (code-exact) — left==right==all bands.
# ---------------------------------------------------------------------------
psiY = psi                                  # (k, n, s, ν)  un-conj  -> col leg
psiX = np.conj(psi).transpose(0, 3, 1, 2)   # (k, μ, n, s)  conj     -> μ leg
# P[k,a,ν,μ,b] = Σ_n psiX[k,μ,n,a] psiY[k,n,b,ν]
P = np.einsum('kmna,knbr->karmb', psiX, psiY, optimize=True)   # (k,a,ν,μ,b)
P = P.reshape(nkx, nky, nkz, ns, n_mu, n_mu, ns)
P_R = np.fft.ifftn(P, axes=(0, 1, 2), norm='forward')
# C_R[kx,ky,kz, ν, μ] = Σ_{a,b} conj(P_R)[...,a,ν,μ,b] P_R[...,a,ν,μ,b]
C_R = np.einsum('xyzavmb,xyzavmb->xyzvm', np.conj(P_R), P_R, optimize=True)
C_q3 = np.fft.fftn(C_R, axes=(0, 1, 2), norm='forward')        # (kx,ky,kz, ν, μ)
C_q = np.transpose(C_q3.reshape(nq, n_mu, n_mu), (0, 2, 1))    # (q, μ, ν)
del P, P_R

# sanity: Hermiticity + PSD conditioning of the q=0 (Γ) block
C0 = C_q[0]
herm = np.linalg.norm(C0 - C0.conj().T) / np.linalg.norm(C0)
ev = np.linalg.eigvalsh(0.5*(C0 + C0.conj().T))
cond = ev[-1] / max(ev[0], 1e-300)
print(f"[{label}] C_q rebuilt.  Γ: ||C-C^H||/||C||={herm:.2e}  eig[min,max]="
      f"[{ev[0]:.3e},{ev[-1]:.3e}]  cond={cond:.3e}")

# ---------------------------------------------------------------------------
# STAGE 1 — falloff of C_R vs |R|
# ---------------------------------------------------------------------------
def wrap(n, N):        # integer coord -> (-N/2, N/2]
    return n - N*((2*n) > N)
Rvecs = np.array([[wrap(ix, nkx), wrap(iy, nky), wrap(iz, nkz)]
                  for ix in range(nkx) for iy in range(nky) for iz in range(nkz)])
# C_R in the same C-order flat as Rvecs
C_R_flat = C_R.reshape(nq, n_mu, n_mu)     # index i -> (ix,iy,iz) C-order
Rdist = np.sqrt(np.einsum('ri,ij,rj->r', Rvecs, adot, Rvecs))    # Bohr (or frac if adot=I)
maxabs = np.array([np.max(np.abs(C_R_flat[i])) for i in range(nq)])
frob   = np.array([np.linalg.norm(C_R_flat[i]) for i in range(nq)])
# group into |R| shells
order = np.argsort(Rdist)
print(f"\n[{label}] STAGE 1 — C_R falloff (per-shell max & Frobenius, "
      f"normalised to R=0):")
print(f"  {'|R|(Bohr)':>10} {'nR':>3} {'max|C_R|/max0':>14} {'||C_R||F/F0':>12}")
seen = {}
for i in order:
    key = round(Rdist[i], 3)
    seen.setdefault(key, []).append(i)
m0 = maxabs[order[0]]; f0 = frob[order[0]]
shell_rows = []
for key in sorted(seen):
    idxs = seen[key]
    sm = max(maxabs[j] for j in idxs)
    sf = max(frob[j]   for j in idxs)
    shell_rows.append((key, len(idxs), sm/m0, sf/f0))
    print(f"  {key:>10.3f} {len(idxs):>3d} {sm/m0:>14.4e} {sf/f0:>12.4e}")

# ---------------------------------------------------------------------------
# STAGE 2 — leave-one-out Fourier interpolation of C_q (truncated-R)
# ---------------------------------------------------------------------------
# q fractional (C-order flat), R integer lattice coords (wrapped, sorted by |R|)
qfrac = np.array([[ix/nkx, iy/nky, iz/nkz]
                  for ix in range(nkx) for iy in range(nky) for iz in range(nkz)])
Rsort = Rvecs[np.argsort(Rdist)]           # R shells, nearest first
Rdist_sorted = Rdist[np.argsort(Rdist)]
Cq_flat = C_q.reshape(nq, -1)              # (q, μ·ν)

def loo_interp(nR):
    """Leave-one-out: for each held-out q, fit C_R from the other q's using
    the nearest nR R-vectors, predict C at the held-out q.  Returns per-q rel
    Frobenius error array."""
    Rset = Rsort[:nR]
    errs = np.zeros(nq)
    for q0 in range(nq):
        train = [q for q in range(nq) if q != q0]
        F = np.exp(-2j*np.pi*(qfrac[train] @ Rset.T))        # (n_train, nR)
        # least-squares C_R = argmin ||F C_R - Cq_train||
        CR, *_ = np.linalg.lstsq(F, Cq_flat[train], rcond=None)   # (nR, μ·ν)
        f0 = np.exp(-2j*np.pi*(qfrac[q0] @ Rset.T))          # (nR,)
        pred = f0 @ CR
        errs[q0] = np.linalg.norm(pred - Cq_flat[q0]) / np.linalg.norm(Cq_flat[q0])
    return errs

print(f"\n[{label}] STAGE 2 — leave-one-out Fourier interp of C_q vs #R-vectors:")
print(f"  {'nR':>4} {'|R|max(Bohr)':>13} {'med rel-Frob':>13} {'max rel-Frob':>13}")
maxnR = min(nq-1, nq)      # can't fit more R than training points
for nR in sorted(set([r for r in [1, 4, 7, 9, 13, 19, 27, 33, 45, 57, nq-2] if 1 <= r <= maxnR])):
    e = loo_interp(nR)
    print(f"  {nR:>4d} {Rdist_sorted[nR-1]:>13.3f} {np.median(e):>13.4e} {np.max(e):>13.4e}")

# save shell data for the report
np.savez(f"/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/falloff_{label}.npz",
         shell_R=np.array([r[0] for r in shell_rows]),
         shell_maxabs_rel=np.array([r[2] for r in shell_rows]),
         shell_frob_rel=np.array([r[3] for r in shell_rows]),
         cond=cond, kgrid=kgrid, n_mu=n_mu)
print(f"\n[{label}] done.")
