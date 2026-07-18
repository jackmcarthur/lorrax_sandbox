"""lr_pin — the Poisson-DF / physical-tensor arbitration rung (literature
synthesis: Manby-Knowles Poisson DF & PySCF MDF pin the aux multipoles
exactly and fit only the multipole-free remainder).  In the poly basis
every non-constant term vanishes at K_par = 0, so pinning the monopole is
just: FREEZE the constant term of each Gz channel to the literal monopole
m0_mu(Gz) = sum_xy FFT_z[zeta](x, y, Gz) (tile_prep T2 machinery, Tik-
cleaned zeta), and LSQ only the a+b >= 1 terms.  Questions:
  P1 is the literal m0 a physical-tensor-like (q-independent) object in
     the Tikhonov gauge?  (hard-gauge adjacent-q roughness was 0.357)
  P2 does pinning it beat / match / hurt the free-constant fit (b26p) in
     the 6x6 LOO B metric?  (hard-gauge literal-moment history: literal
     o0 E-scheme 1.18e-2 vs FITTED constant D 4.98e-3 — fits won)

Run: JID=<jid> ./proto1_run.sh python3 -u lr_pin.py MoS2_6x6
"""
import sys
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import fix_sphere_wrap, run_gates, sorted_stencil
from tile_prep import TileStudy, B_tile, check_slab_axes
from lr_prep import LRSamples, ChannelFit, spec_poly, eval_basis

FIXNAME = sys.argv[1] if len(sys.argv) > 1 else "MoS2_6x6"
ALPHA = 0.30
EPS_TIK = 1e-4
t00 = time.time()

fx = Fixture(FIXNAME)
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
assert check_slab_axes(fx) < 1e-12
Stik = []
for q in range(fx.nq):
    lam, R = ts.eig[q]
    g = lam ** 2 / (lam ** 2 + (EPS_TIK * lam[0]) ** 2)
    Stik.append((R * g[None, :]) @ R.conj().T)
lr = LRSamples(ts, None, ALPHA)
for q in range(fx.nq):
    zt = Stik[q] @ fx.ZG[q]
    idx = ts.sphere_slot(q, lr.GS)
    zt_ext = np.concatenate([zt, np.zeros((fx.n_mu, 1), np.complex128)], 1)
    qG = fx.qfr[q][None, :] + lr.GS.T.astype(np.float64)
    ph = np.exp(2j * np.pi * (fx.rmu_frac @ qG.T))
    lr.Fch[q] = ph * zt_ext[:, idx]
Vc = np.stack([np.conj(Stik[q]) @ ts.V_ref[q] @ np.conj(Stik[q])
               for q in range(fx.nq)])
VLRc = np.stack([np.conj(Stik[q])
                 @ fx.make_Vq(fx.ZG[q], q, kind="slab_lr", alpha=ALPHA)
                 @ np.conj(Stik[q]) for q in range(fx.nq)])

# literal per-(q,Gz) monopoles of the Tik-cleaned zeta (T2 machinery)
BSGZ = [g for g in lr.gz_vals if abs(g) <= 3]
sel = [int(g) % fx.nz for g in BSGZ]
m0 = np.zeros((fx.nq, fx.n_mu, len(BSGZ)), dtype=np.complex128)
for q in range(fx.nq):
    zlab = ts._recon_rows(q, Stik[q] @ fx.ZG[q]).reshape(
        fx.n_mu, fx.nx, fx.ny, fx.nz)
    Fz = np.fft.fft(zlab, axis=3)[:, :, :, sel]
    zph = np.exp(2j * np.pi * np.outer(
        fx.rmu_frac[:, 2], np.array(BSGZ, dtype=float)))
    m0[q] = zph * np.einsum("mxyg->mg", Fz)

# P1: adjacent-q roughness of the literal m0 (Tik gauge)
kg = fx.kgrid
pairs = [(q, fx.k_lookup[tuple((fx.k_int[q] + np.array([1, 0, 0])) % kg)])
         for q in range(fx.nq)]
d = [np.linalg.norm(m0[a] - m0[b]) / np.linalg.norm(m0[a])
     for a, b in pairs]
print(f"\n[P1] literal m0 (Tik) adjacent-q rel diff: med "
      f"{np.median(d):.3f} max {np.max(d):.3f}   "
      f"(hard-gauge cleaned m0 was 0.357 med — grep tile_t1t2_6x6.log)")

# P2: pinned vs free-constant b26p, 6x6 LOO B (D-style)
BS = {0: spec_poly(3), 1: spec_poly(2), 2: spec_poly(0), 3: spec_poly(0)}
specs = {g: BS[abs(g)] for g in lr.gz_vals if abs(g) in BS}
cf = ChannelFit(lr, specs, tag="b26p")
gzpos = {g: i for i, g in enumerate(BSGZ)}


def coeffs_pinned(exclude):
    """Freeze constant term to the TRAIN-mean literal m0; LSQ the rest.
    Solve on the reduced (non-constant) block of the same normal eqs:
    C1 = A11^{-1} (AtY_1 - A10 c0)."""
    sel_q = [q for q in range(fx.nq) if q != exclude]
    out = {}
    for g, (AtA, AtY) in cf.blocks.items():
        A = AtA[sel_q].sum(0)
        Y = AtY[sel_q].sum(0)
        c0 = m0[sel_q][:, :, gzpos[g]].mean(0)          # (nmu,) pinned
        nb = A.shape[0]
        if nb == 1:
            out[g] = c0[None, :].astype(np.complex128)
            continue
        A11 = A[1:, 1:] + cf.RIDGE * (np.trace(A) / nb) * np.eye(nb - 1)
        rhs = Y[1:] - A[1:, :1] @ c0[None, :]
        C = np.zeros((nb, fx.n_mu), dtype=np.complex128)
        C[0] = c0
        C[1:] = np.linalg.solve(A11, rhs)
        out[g] = C
    return out


R7 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                         for j in range(-2, 4)])[:7]
Bfree, Bpin, fid_f, fid_p = [], [], [], []
for q0 in range(fx.nq):
    train = [q for q in range(fx.nq) if q != q0]
    w = truncR_weights(fx.qfr[train], fx.qfr[q0], R7)
    x = fx.gap_window_pairs(q0, 3, 3)
    B_true = B_tile(x, ts.V_ref[q0])
    SRi = np.tensordot(w, Vc[train] - VLRc[train], axes=(0, 0))
    for Cl, acc, fac in ((cf.coeffs(exclude=q0), Bfree, fid_f),
                         (coeffs_pinned(q0), Bpin, fid_p)):
        Vm = ts.V_from_F(cf.model_F(Cl, fx.qfr[q0]), fx.qfr[q0], lr.GS,
                         ALPHA)
        acc.append(relF(B_tile(x, SRi + Vm), B_true))
        fac.append(relF(Vm, VLRc[q0]))
print(f"\n[P2] 6x6 LOO (Tik, D-style): free-constant b26p B med "
      f"{np.median(Bfree):.3e} max {np.max(Bfree):.3e} "
      f"(fid med {np.median(fid_f):.3e})")
print(f"[P2] 6x6 LOO (Tik, D-style): PINNED-m0 b26p   B med "
      f"{np.median(Bpin):.3e} max {np.max(Bpin):.3e} "
      f"(fid med {np.median(fid_p):.3e})")
np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         f"lr_pin_{FIXNAME}_results.npz",
         Bfree=np.array(Bfree), Bpin=np.array(Bpin), m0adj=np.array(d),
         fid_f=np.array(fid_f), fid_p=np.array(fid_p))
print(f"\n[lr_pin {FIXNAME}] ALL DONE in {time.time()-t00:.0f}s")
