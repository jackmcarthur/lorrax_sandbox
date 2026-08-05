import h5py, numpy as np, sys
RY=13.605693122994
D=sys.argv[1]
with h5py.File(f"{D}/WFN.h5","r") as f:
    rk=f["/mf_header/kpoints/rk"][()]
    if rk.shape[0]==3 and rk.shape[1]!=3: rk=rk.T
    el=f["/mf_header/kpoints/el"][()]  # (nspin,nrk,nb) Ry
    ifmax=f["/mf_header/kpoints/ifmax"][()]
el=el[0]*RY  # (nrk,nb) eV
nocc=int(np.max(ifmax))
def frac(x): return np.mod(x+1e-6,1.0)-1e-6
def findk(t):
    d=np.minimum(np.linalg.norm(frac(rk-t),axis=1),np.linalg.norm(frac(rk+t),axis=1))
    return int(np.argmin(d))
ikG=findk(np.array([0,0,0])); ikK=findk(np.array([1/3,1/3,0]))
# Fermi (midgap)
vbm=el[:,nocc-1].max(); cbm=el[:,nocc].min(); ef=0.5*(vbm+cbm)
with h5py.File(f"{D}/sigma_mnk.h5","r") as f:
    om=f["/omega_ev"][()]
    sc=f["/sigma_c_kij_ev"]   # (nw,nk,nb,nb)  band index 0..99 maps to WFN band (sigma window start)
    sx=f["/sigma_sx_kij_ev"]  # (nk,nb,nb)
    nbw=sc.shape[2]
    # sigma window: bands 1..100 (1-indexed) => WFN band index 0..99. So sigma idx b == WFN band b.
    def interp(ik,b):
        e_edft=el[ik,b]-ef
        d=np.asarray(sc[:,ik,b,b])
        return np.interp(e_edft, om, d.real), np.interp(e_edft, om, d.imag), e_edft, float(np.asarray(sx[ik,b,b]).real)
print(f"nocc={nocc} VBM_band(0idx)={nocc-1} CBM_band={nocc} Ef(midgap)={ef:.3f} eV")
print(f"ikGamma={ikG} rk={rk[ikG]}  ikK={ikK} rk={rk[ikK]}")
for name,ik in [("Gamma",ikG),("K",ikK)]:
    print(f"\n== {name} (ik={ik}) ==")
    for b,lbl in [(nocc-1,"VBM"),(nocc,"CBM")]:
        scR,scI,edft,sxv=interp(ik,b)
        print(f"  {lbl} band{b}: Edft-Ef={edft:+.3f}  sigX={sxv:+.3f}  Re sigC(Edft)={scR:+.4f}  Im={scI:+.4f} eV")
