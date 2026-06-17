import h5py, numpy as np
WFN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf_bandpath/WFN.h5"
f = h5py.File(WFN,"r")
mf = f["mf_header"]
nk   = mf["kpoints/nrk"][()]
nb   = mf["kpoints/mnband"][()]
ns   = mf["kpoints/nspin"][()]
nsppol = mf["kpoints/nspinor"][()] if "nspinor" in mf["kpoints"] else None
rk   = mf["kpoints/rk"][()]
en   = mf["kpoints/el"][()]
print("nk =", nk, " mnband =", nb, " nspin =", ns, " nspinor =", nsppol)
print("rk shape:", rk.shape, " el shape:", en.shape)
print("first 3 kpts (crys):"); print(np.round(rk[:3],6))
print("k[40] (should be M=0.5,0,0):", np.round(rk[40],6))
print("k[80] (should be K=1/3,1/3,0):", np.round(rk[80],6))
print("k[120] (should be G):", np.round(rk[120],6))
pi = np.load("/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf_bandpath/path_info.npz")
kp = pi["kpts_crys"]
diff = np.abs(((rk - kp + 0.5) % 1.0) - 0.5).max()
print("max |rk - intended| (mod 1):", diff)
RY=13.605693
E_ev = en[0]*RY
print("E range (eV): min", E_ev.min().round(3), "max", E_ev.max().round(3))
print("VBM (band 69) at Gamma:", (E_ev[0,69]).round(4), " CBM (band 70) at Gamma:", (E_ev[0,70]).round(4))
f.close()
