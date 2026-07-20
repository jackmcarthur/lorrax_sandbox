"""Exciton bandstructure with head_minibz_average OFF vs ON (validation d).
Runs the driver twice in one process and quantifies the per-Q eigenvalue shift.
Fixture: MoS2 3x3 640-centroid, Γ→M→K→Γ path.  [RESULT]-tagged."""
import os, sys
import numpy as np

RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/minibz_head_averaging_2026-07-20/exb_cmp"
os.chdir(RUN)
from bse.exciton_bands import main

def run(prefix, extra):
    argv = ["-i", "exb.in", "--n-val", "2", "--n-cond", "2", "--n-eig", "4",
            "--block-size", "4", "--max-iter", "30", "--vq-mode", "interp",
            "--out-prefix", prefix] + extra
    rc = main(argv)
    assert rc == 0, f"{prefix} rc={rc}"

def parse(prefix):
    rows = [l.split() for l in open(f"{prefix}.dat")
            if l.strip() and not l.startswith("#")]
    interp = [r for r in rows if r[5] == "interp"]
    # columns: iQ qx qy qz |Q| mode ev0 ev1 ...
    iQ = np.array([int(r[0]) for r in interp])
    Qn = np.array([float(r[4]) for r in interp])
    ev = np.array([[float(x) for x in r[6:]] for r in interp])  # eV
    return iQ, Qn, ev

print(">>> run OFF (point value, default)")
run("off", [])
print(">>> run ON (mini-BZ cell average)")
run("on", ["--head-minibz-average"])

iQo, Qno, evo = parse("off")
iQn, Qnn, evn = parse("on")
assert np.array_equal(iQo, iQn)
dshift = (evn - evo) * 1000.0   # meV
print("\n[RESULT] (d) exciton eigenvalue shift ON-OFF (meV), lowest 4 states per Q:")
print(f"[RESULT] (d) {'iQ':>3} {'|Q|':>7}  {'ev0_off(eV)':>11} {'Δev0':>8} {'Δev1':>8} "
      f"{'Δev2':>8} {'Δev3':>8}")
for k in range(len(iQo)):
    print(f"[RESULT] (d) {iQo[k]:>3d} {Qno[k]:>7.4f}  {evo[k,0]:>11.4f} "
          + " ".join(f"{dshift[k,j]:>8.2f}" for j in range(min(4, dshift.shape[1]))))
# near-Γ (smallest nonzero |Q|) and zone-boundary (M ~ |Q| max on first leg)
nz = np.where(Qno > 1e-6)[0]
i_nearG = nz[np.argmin(Qno[nz])]
i_maxQ = int(np.argmax(Qno))
print(f"\n[RESULT] (d) near-Γ  iQ={iQo[i_nearG]} |Q|={Qno[i_nearG]:.4f}: "
      f"lowest-state shift = {dshift[i_nearG,0]:+.2f} meV, "
      f"max|shift| over 4 states = {np.max(np.abs(dshift[i_nearG])):.2f} meV")
print(f"[RESULT] (d) zone-bd iQ={iQo[i_maxQ]} |Q|={Qno[i_maxQ]:.4f}: "
      f"lowest-state shift = {dshift[i_maxQ,0]:+.2f} meV, "
      f"max|shift| over 4 states = {np.max(np.abs(dshift[i_maxQ])):.2f} meV")
gam = np.where(Qno < 1e-6)[0]
if len(gam):
    print(f"[RESULT] (d) Γ (Q=0, untouched) max|shift| = "
          f"{np.max(np.abs(dshift[gam])):.4f} meV (expect ~0: q=0 tile is production)")
print(f"[RESULT] (d) global median|shift|={np.median(np.abs(dshift)):.2f} meV "
      f"max|shift|={np.max(np.abs(dshift)):.2f} meV")
np.savez(f"{RUN}/exciton_flag_shift.npz", iQ=iQo, Qn=Qno, ev_off=evo, ev_on=evn)
print("[INFO] DONE")
