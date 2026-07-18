"""lr_singlevalued — EXPERIMENT 1 of the LR compact-representation study:
is the phase-factored form factor M_mu(K) (sampled as F_mu(q;G) at K = q+G)
approximately a function of K ALONE, i.e. free of q-dependence beyond K?

On-grid the samples tile a regular fine lattice with exactly ONE sample per
K point (stage G proves this from the data), so single-valuedness cannot be
a coincidence test; it decomposes into (see lr_prep header):
  S1 seam parity  — adjacent fine-lattice pairs that cross a BZ boundary
     (Miller label changes between the two samples' sheets) vs pairs that
     do not.  Both classes compare DIFFERENT source q's at the same |dK|;
     they differ only in the G-relabel, so any excess in the cross class is
     a residual seam beyond the analytically-carried winding phase.
  S2 plateau      — weighted fit residual vs polynomial degree at Gz=0/1:
     a q-fiber obstruction would floor the residual independently of basis
     richness; a smooth single-valued M keeps dropping to the fiber floor.
  S3 fiber        — after a rich smooth fit, fraction of weighted residual
     power in coherent per-(q,Gz) means, vs the white-residual dof
     expectation #groups/#samples.  ~dof-level => no per-q fiber; O(1) =>
     M has real q-dependence beyond K and the global-fit program caps out.

Run: JID=<jid> ./proto1_run.sh python3 -u lr_singlevalued.py MoS2_6x6
"""
import sys
import time
import numpy as np

from proto1_prep import Fixture, relF
from offgrid_prep import fix_sphere_wrap, run_gates
from tile_prep import TileStudy, check_slab_axes
from lr_prep import LRSamples, ChannelFit, spec_poly, spec_gto, eval_basis

FIXNAME = sys.argv[1] if len(sys.argv) > 1 else "MoS2_6x6"
ALPHA = 0.30
RC = 1e-4
t00 = time.time()
NPZ = {}

print(f"[lr_sv] fixture {FIXNAME}; alpha={ALPHA}, rc={RC}")
fx = Fixture(FIXNAME)
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
assert check_slab_axes(fx) < 1e-12
lr = LRSamples(ts, RC, ALPHA)

# ===========================================================================
# Stage G — structure gates
# ===========================================================================
print("\n[stageG] sample-structure gates")
print(f"  [gate] fine-lattice integrality dev {lr.fine_dev:.3e}"
      + ("  OK" if lr.fine_dev < 1e-6 else "  ** FAIL **"))
assert lr.fine_dev < 1e-6   # stored rk carries ~3e-10 decimal rounding
dup = 0
for g in lr.gz_vals:
    seen = set()
    for q in range(fx.nq):
        for j in lr.cols[g]:
            key = (int(lr.fine[q, j, 0]), int(lr.fine[q, j, 1]))
            if key in seen:
                dup += 1
            seen.add(key)
print(f"  [gate] duplicate fine-lattice points across (q,G): {dup}"
      + ("  OK (one sample per K — coincidence test impossible; "
         "smoothness/fiber tests below)" if dup == 0 else "  ** FAIL **"))
assert dup == 0
# synthetic recovery: plant known coefficients, fit must recover exactly
spec0 = spec_poly(3)
Ctrue = (np.random.default_rng(0).standard_normal((len(spec0), fx.n_mu))
         + 1j * np.random.default_rng(1).standard_normal((len(spec0),
                                                          fx.n_mu)))
lr_fake = LRSamples(ts, RC, ALPHA)
for q in range(fx.nq):
    c = lr_fake.cols[0]
    Phi = eval_basis(lr_fake.K[q][:2][:, c], spec0, ALPHA)
    lr_fake.Fch[q][:, c] = (Phi @ Ctrue).T
cf0 = ChannelFit(lr_fake, {0: spec0})
C0 = cf0.coeffs()
err = relF(C0[0], Ctrue)
print(f"  [gate] synthetic poly3 recovery relF {err:.3e}"
      + ("  OK" if err < 1e-8 else "  ** FAIL **"))
assert err < 1e-8
ws = lr.wshare_gz()
print("  [info] v_LR weight share per Gz: "
      + " ".join(f"{g}:{ws[g]:.3f}" for g in sorted(ws, key=abs)
                 if abs(ws[g]) > 5e-4))
NPZ["wshare_gz"] = np.array([[g, ws[g]] for g in lr.gz_vals])

# ===========================================================================
# S1 — seam parity: adjacent fine-lattice pairs, same-G vs cross-G
# ===========================================================================
print("\n[S1] seam parity (adjacent fine-lattice pairs, |dK| = |b|/N)")
WMAXG = {g: max(lr.W[:, lr.cols[g]].max(), 1e-300) for g in lr.gz_vals}
rows = []          # (gz, cross, reldiff, wmin)
for g in lr.gz_vals:
    loc = {}       # fine-int -> (q, j)
    for q in range(fx.nq):
        for j in lr.cols[g]:
            loc[(int(lr.fine[q, j, 0]), int(lr.fine[q, j, 1]))] = (q, j)
    for (fxi, fyi), (q1, j1) in loc.items():
        for dx, dy in ((1, 0), (0, 1)):
            nb = loc.get((fxi + dx, fyi + dy))
            if nb is None:
                continue
            q2, j2 = nb
            F1, F2 = lr.Fch[q1][:, j1], lr.Fch[q2][:, j2]
            n1, n2 = np.linalg.norm(F1), np.linalg.norm(F2)
            if n1 + n2 < 1e-300:
                continue
            d = np.linalg.norm(F1 - F2) / (0.5 * (n1 + n2))
            cross = not np.array_equal(lr.GS[:, j1], lr.GS[:, j2])
            wmin = min(lr.W[q1, j1], lr.W[q2, j2])
            rows.append((g, float(cross), d, wmin))
rows = np.array(rows)
NPZ["seam_rows"] = rows
for gzsel, lab in ((None, "all Gz"), (0, "Gz=0")):
    m = np.ones(len(rows), bool) if gzsel is None else rows[:, 0] == gzsel
    for wcut, wlab in ((1e-4, "w>=1e-4*wmax"), (0.0, "all")):
        sel = m & np.array([r[3] >= wcut * WMAXG[int(r[0])] for r in rows])
        for cross in (0.0, 1.0):
            s = sel & (rows[:, 1] == cross)
            if not np.any(s):
                continue
            d = rows[s, 2]
            w = rows[s, 3]
            wm = float(np.sum(w * d) / np.sum(w))
            print(f"  [S1] {lab:>6s} {wlab:<14s} "
                  f"{'cross-G' if cross else 'same-G '} pairs {s.sum():>5d}: "
                  f"med {np.median(d):.3f} p90 "
                  f"{np.percentile(d, 90):.3f} w-mean {wm:.3f}")
# headline: ratio of medians on the weight-relevant subset, all Gz
selw = np.array([r[3] >= 1e-4 * WMAXG[int(r[0])] for r in rows])
med_same = np.median(rows[selw & (rows[:, 1] == 0), 2])
med_cross = np.median(rows[selw & (rows[:, 1] == 1), 2])
print(f"  [S1] HEADLINE seam ratio cross/same = "
      f"{med_cross / med_same:.3f} (1.0 = no seam beyond analytic phase)")
NPZ["seam_ratio"] = np.array([med_same, med_cross])

# ===========================================================================
# S2 — plateau: weighted rel residual vs poly degree, Gz = 0 and |Gz| = 1
# ===========================================================================
print("\n[S2] fit-residual plateau (weighted rel residual, own fit)")
plateau = {}
for g in (0, 1, -1):
    rels = []
    for d in range(9):
        cf = ChannelFit(lr, {g: spec_poly(d)})
        st = cf.resid_stats(cf.coeffs())
        rels.append(st["rel"])
    plateau[g] = rels
    print(f"  [S2] Gz={g:+d} poly d=0..8: "
          + " ".join(f"{r:.3f}" for r in rels))
    NPZ[f"plateau_gz{g}"] = np.array(rels)
# gto ladder at Gz=0 for the radial-freedom comparison
sig = [ALPHA / np.sqrt(2.0), ALPHA, ALPHA * np.sqrt(2.0)]
for d in (2, 3, 4):
    cf = ChannelFit(lr, {0: spec_gto(d, sig)})
    st = cf.resid_stats(cf.coeffs())
    print(f"  [S2] Gz=+0 gto3xpoly{d} (nb={cf.n_coeff()}): rel "
          f"{st['rel']:.3f}")
    NPZ[f"plateau_gz0_gto{d}"] = np.array([cf.n_coeff(), st["rel"]])

# ===========================================================================
# S3 — fiber test: per-(q,Gz) coherent residual fraction after a rich fit
# ===========================================================================
print("\n[S3] q-fiber test (rich fit: gto3 x poly4 on |Gz|<=2, "
      "gto3 x poly2 beyond)")
specs = {}
for g in lr.gz_vals:
    specs[g] = spec_gto(4, sig) if abs(g) <= 2 else spec_gto(2, sig)
cf = ChannelFit(lr, specs)
C = cf.coeffs()
st = cf.resid_stats(C)
ngroups = len(st["rows"])
nsamp = sum(len(lr.cols[g]) * fx.nq for g in cf.specs)
print(f"  [S3] global weighted rel residual (rich): {st['rel']:.4f}")
print(f"  [S3] fiber fraction (coherent per-(q,Gz) share of residual "
      f"power): {st['fiber_frac']:.4f}")
print(f"  [S3] white-residual dof expectation #groups/#samples: "
      f"{ngroups}/{nsamp} = {ngroups / nsamp:.4f}")
print(f"  [S3] HEADLINE fiber excess = "
      f"{st['fiber_frac'] / (ngroups / nsamp):.2f}x dof expectation")
NPZ["fiber"] = np.array([st["rel"], st["fiber_frac"], ngroups / nsamp])
# per-q spread of the fiber means at Gz=0 (is any single q an outlier?)
r0 = st["rows"][st["rows"][:, 1] == 0]
fr = np.sqrt(r0[:, 4] / np.maximum(r0[:, 5], 1e-300))
print("  [S3] per-q coherent-residual/|Y| at Gz=0: med "
      f"{np.median(fr):.4f} max {np.max(fr):.4f} (q_max="
      f"{int(r0[np.argmax(fr), 0])})")
NPZ["fiber_perq_gz0"] = r0

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         f"lr_singlevalued_{FIXNAME}_results.npz", **NPZ)
print(f"\n[lr_singlevalued {FIXNAME}] ALL DONE in {time.time()-t00:.0f}s")
