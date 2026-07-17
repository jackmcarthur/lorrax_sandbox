import h5py, numpy as np
p="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/zeta_q.h5"
with h5py.File(p,"r") as f:
    fg=f["mf_header/gspace/FFTgrid"][()]; print("FFTgrid",fg,"n_rtot",int(np.prod(fg)))
    print("kgrid",f["mf_header/kpoints/kgrid"][()])
    rk=f["mf_header/kpoints/rk"][()]; print("rk (q frac):\n",rk)
    print("bdot:\n",f["mf_header/crystal/bdot"][()])
    print("blat",f["mf_header/crystal/blat"][()],"celvol",f["mf_header/crystal/celvol"][()])
    print("nspinor",f["mf_header/kpoints/nspinor"][()])
    rmu=f["isdf_header/centroids/r_mu_crystal"][()]; print("r_mu_crystal[:3]\n",rmu[:3])
    print("zeta_cutoff",f["isdf_header/zeta_cutoff_ry"][()])
