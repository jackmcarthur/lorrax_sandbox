#!/usr/bin/env python3
"""Analyze a PPM fit dump (omega_ry=ω̃, absB_ry, absWc0_ry, absWci_ry, valid).
Characterizes BOTH the low-Ω (near-invalid, ω̃²≈0⁺) and high-Ω/huge-|B| ends."""
import sys, numpy as np
RY = 13.605693122994
f = np.load(sys.argv[1])
Om = f["omega_ry"].astype(np.float64)      # ω̃ (Ry), 0 for invalid/pad
aB = f["absB_ry"].astype(np.float64)       # |B| (Ry)
aW0 = f["absWc0_ry"].astype(np.float64)    # |Wc0| (Ry)
valid = f["valid"].astype(bool)
nmu = int(f["n_mu_logical"]); probe = float(f["probe_ry"])
print(f"# dump {sys.argv[1]}  n_mu_logical={nmu}  probe ω_p={probe} Ry ({probe*RY:.1f} eV)")
tot = valid.size
nv = int(valid.sum())
print(f"# total elements={tot}  valid(dispersive-pole)={nv} ({100*nv/tot:.2f}%)  "
      f"invalid+dead+pad={tot-nv}")
Ov = Om[valid]; Bv = aB[valid]; W0v = aW0[valid]
Ov_eV = Ov*RY; Bv_eV = Bv*RY; W0v_eV = W0v*RY
def q(a,p): return float(np.percentile(a,p)) if a.size else 0.0
print("\n== valid ω̃ (pole freq) distribution, eV ==")
print(f"  min={Ov_eV.min():.4e}  p0.01={q(Ov_eV,0.01):.4e}  p0.1={q(Ov_eV,0.1):.4f}  "
      f"p1={q(Ov_eV,1):.3f}  p50={q(Ov_eV,50):.3f}  p99={q(Ov_eV,99):.3f}  max={Ov_eV.max():.3f}")
print(f"  probe ω_p = {probe*RY:.2f} eV;  median pole/probe = {q(Ov,50)/probe:.3f}")
print("\n== LOW-Ω (near-invalid ω̃²≈0⁺) census — the owner hypothesis ==")
for thr in (0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5):  # Ry
    sel = Ov < thr
    n = int(sel.sum())
    if n:
        print(f"  ω̃<{thr:5.3f}Ry ({thr*RY:5.2f}eV): n={n:>8d} ({100*n/nv:7.4f}% of valid)  "
              f"max|B|={Bv_eV[sel].max():.3e}eV  med|B|={np.median(Bv_eV[sel]):.4f}eV  "
              f"max|Wc0|={W0v_eV[sel].max():.3e}eV")
    else:
        print(f"  ω̃<{thr:5.3f}Ry ({thr*RY:5.2f}eV): n=0")
print("\n== HIGH-Ω / huge-|B| census — the fix-agent population ==")
for thr in (2.0,3.0,5.0,6.0):  # Ry
    sel = Ov > thr
    n = int(sel.sum())
    if n:
        print(f"  ω̃>{thr:.1f}Ry ({thr*RY:5.1f}eV): n={n:>8d} ({100*n/nv:7.4f}%)  "
              f"max|B|={Bv_eV[sel].max():.3e}eV  max|Wc0|={W0v_eV[sel].max():.3e}eV")
print("\n== |B| distribution (valid), eV ==")
print(f"  p50={q(Bv_eV,50):.1f}  p99={q(Bv_eV,99):.3e}  p99.9={q(Bv_eV,99.9):.3e}  "
      f"p99.99={q(Bv_eV,99.99):.3e}  max={Bv_eV.max():.3e}")
# where do the huge-|B| live in ω̃?
topB = np.argsort(Bv_eV)[-2000:]
print(f"  top-2000 |B|: ω̃ range=[{Ov_eV[topB].min():.2f},{Ov_eV[topB].max():.2f}]eV  "
      f"median ω̃={np.median(Ov_eV[topB]):.2f}eV  (are the huge-|B| at LOW or HIGH ω̃?)")
print(f"  |B|(eV) among near-invalid ω̃<0.05Ry: "
      f"max={Bv_eV[Ov<0.05].max() if (Ov<0.05).any() else 0:.3e}  "
      f"vs global max={Bv_eV.max():.3e}")
