"""lr_fiber_source — WHERE does the 6x6 q-fiber live?  The lr_singlevalued
S3 test found ~65% of the rich-fit residual coherent per-(q,Gz) at 6x6
(vs tiny at 3x3).  Candidate source: the HARD rank-cut cleaning projector
P_q (sec 12.3 measured a persistent ~22% Davis-Kahan cut-edge rotation
between adjacent q — modes that are physically inert under B but jump in
the cleaned zeta).  Test: run the same rich fit + fiber decomposition on
  (a) raw channels        (rc=None — no projector at all)
  (b) hard-cut channels   (rc=1e-4 — the study default)
  (c) Tikhonov channels   (S_eps = R g_eps(L) R^H, g = l^2/(l^2+eps^2),
                           eps = 1e-4 * lam0 — the sec-12.3 smooth filter)
If the fiber shrinks in (a)/(c) vs (b), the fiber is the cut edge, i.e.
physically-inert junk the B-metric forgives (the ladder's D_rich B row
tests the same claim downstream).

Run: JID=<jid> ./proto1_run.sh python3 -u lr_fiber_source.py MoS2_6x6
"""
import sys
import time
import numpy as np

from proto1_prep import Fixture
from offgrid_prep import fix_sphere_wrap, run_gates
from tile_prep import TileStudy, check_slab_axes
from lr_prep import LRSamples, ChannelFit, spec_gto

FIXNAME = sys.argv[1] if len(sys.argv) > 1 else "MoS2_6x6"
ALPHA = 0.30
SIG = [ALPHA / np.sqrt(2.0), ALPHA, ALPHA * np.sqrt(2.0)]
t00 = time.time()

fx = Fixture(FIXNAME)
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
assert check_slab_axes(fx) < 1e-12


def tik_zt(q, eps_rel):
    lam, R = ts.eig[q]
    g = lam ** 2 / (lam ** 2 + (eps_rel * lam[0]) ** 2)
    return (R * g[None, :]) @ (R.conj().T @ fx.ZG[q])


def fiber_report(tag, lr):
    specs = {g: (spec_gto(4, SIG) if abs(g) <= 2 else spec_gto(2, SIG))
             for g in lr.gz_vals}
    cf = ChannelFit(lr, specs)
    st = cf.resid_stats(cf.coeffs())
    ngroups = len(st["rows"])
    nsamp = sum(len(lr.cols[g]) * fx.nq for g in cf.specs)
    r0 = st["rows"][st["rows"][:, 1] == 0]
    fr = np.sqrt(r0[:, 4] / np.maximum(r0[:, 5], 1e-300))
    print(f"  [{tag:>8s}] wres {st['rel']:.4f}  fiber_frac "
          f"{st['fiber_frac']:.4f} (dof {ngroups/nsamp:.4f})  "
          f"per-q coh/|Y| Gz=0 med {np.median(fr):.4f} max {np.max(fr):.4f}")


print(f"\n[lr_fiber_source {FIXNAME}] rich-fit fiber by cleaning variant")
fiber_report("hard1e-4", LRSamples(ts, 1e-4, ALPHA))
fiber_report("raw", LRSamples(ts, None, ALPHA))
lr_t = LRSamples(ts, None, ALPHA)
for q in range(fx.nq):
    zt = tik_zt(q, 1e-4)
    idx = ts.sphere_slot(q, lr_t.GS)
    zt_ext = np.concatenate([zt, np.zeros((fx.n_mu, 1), np.complex128)], 1)
    qG = fx.qfr[q][None, :] + lr_t.GS.T.astype(np.float64)
    ph = np.exp(2j * np.pi * (fx.rmu_frac @ qG.T))
    lr_t.Fch[q] = ph * zt_ext[:, idx]
fiber_report("tik1e-4", lr_t)
print(f"[lr_fiber_source {FIXNAME}] ALL DONE in {time.time()-t00:.0f}s")
