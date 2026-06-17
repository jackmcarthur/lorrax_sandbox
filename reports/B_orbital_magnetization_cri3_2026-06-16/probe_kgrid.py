import h5py, numpy as np
WFN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf_bandpath/WFN.h5"
f=h5py.File(WFN,"r")
mf=f["mf_header"]
print("kgrid dataset:", mf["kpoints/kgrid"][()])
rk=mf["kpoints/rk"][()]
# check exactness vs 240 grid
for N in (120,240):
    err=np.abs(rk[:,:2]*N - np.rint(rk[:,:2]*N)).max()
    print(f"N={N}: max frac err on (kx,ky) =", err)
# distinct integer triples?
ints=np.mod(np.rint(rk*240).astype(int),240)
uniq=np.unique(ints,axis=0)
print("npts=",len(rk)," unique int triples (mod 240) =",len(uniq))
f.close()
