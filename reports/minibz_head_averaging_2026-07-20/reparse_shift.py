import numpy as np
D="/pscratch/sd/j/jackm/lorrax_sandbox/reports/minibz_head_averaging_2026-07-20/exb_cmp"
def parse(p):
    rows=[l.split() for l in open(f"{D}/{p}.dat") if l.strip() and not l.startswith("#")]
    r=[x for x in rows if x[5]=="interp"]
    iQ=np.array([int(x[0]) for x in r]); Qfr=np.array([[float(x[2]),float(x[3]),float(x[4])] for x in r])
    ev=np.array([[float(z) for z in x[6:]] for x in r]); return iQ,Qfr,ev
iQ,Qfr,evo=parse("off"); _,_,evn=parse("on")
# |Q| via reciprocal metric from the fixture bvec (crystal_b fractional)
import h5py
with h5py.File(f"{D}/tmp/zeta_q.h5") as f:
    blat=float(np.real(f["mf_header/crystal/blat"][()])); bvec=f["mf_header/crystal/bvec"][()]*blat
Qc=Qfr@bvec; Qn=np.linalg.norm(Qc,axis=1)
d=(evn-evo)*1000.0
labels={0:"Gamma",1:"near-Gamma",4:"M(zone-bd)",8:"K",9:"Gamma"}
print("[RESULT] (d) per-Q exciton shift ON-OFF (meV):")
print(f"[RESULT] (d) {'iQ':>3} {'|Q|(1/bohr)':>11} {'label':>11}  {'Δev0':>7} {'Δev1':>7} {'Δev2':>7} {'Δev3':>7}")
for k in range(len(iQ)):
    print(f"[RESULT] (d) {iQ[k]:>3d} {Qn[k]:>11.4f} {labels.get(iQ[k],''):>11}  "
          +" ".join(f"{d[k,j]:>7.2f}" for j in range(4)))
nz=np.where(Qn>1e-6)[0]; iG=nz[np.argmin(Qn[nz])]; iM=4
print(f"[RESULT] (d) near-Γ iQ={iQ[iG]} |Q|={Qn[iG]:.4f}: Δev0={d[iG,0]:+.2f} meV, max|Δ|={np.max(np.abs(d[iG])):.2f} meV")
print(f"[RESULT] (d) zone-bd(M) iQ={iQ[iM]} |Q|={Qn[iM]:.4f}: Δev0={d[iM,0]:+.2f} meV, max|Δ|={np.max(np.abs(d[iM])):.2f} meV")
g=np.where(Qn<1e-6)[0]
print(f"[RESULT] (d) Γ (q=0, untouched) max|Δ|={np.max(np.abs(d[g])):.4f} meV (expect 0)")
print(f"[RESULT] (d) global median|Δ|={np.median(np.abs(d)):.2f} meV, max|Δ|={np.max(np.abs(d)):.2f} meV")
np.savez(f"{D}/exciton_flag_shift.npz",iQ=iQ,Qn=Qn,ev_off=evo,ev_on=evn,shift_meV=d)
print("[INFO] saved exciton_flag_shift.npz")
