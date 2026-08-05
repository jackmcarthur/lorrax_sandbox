import h5py, numpy as np
RD="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu"
# E_dft-Ef for K,M CBM from debug file
edft={}
for line in open(f"{RD}/sigma_freq_debug.dat"):
    p=line.split()
    if len(p)>=14 and p[0].isdigit() and int(p[0]) in (14,18) and int(p[1])==26:
        edft[int(p[0])]=float(p[3])  # Edft absolute
# need Ef: Edft-Ef = col4; abs Edft col3; Ef = col3-col4
ef={}
for line in open(f"{RD}/sigma_freq_debug.dat"):
    p=line.split()
    if len(p)>=14 and p[0].isdigit() and int(p[0]) in (14,18) and int(p[1])==26:
        ef[int(p[0])]=float(p[3])-float(p[4])
with h5py.File(f"{RD}/sigma_mnk.h5","r") as f:
    omega=f["/omega_ev"][()]
    for ik,name in [(14,"K"),(18,"M")]:
        b=26
        sc=f["/sigma_c_kij_ev"][:,ik,b,b]
        edft_rel = None
        # recompute rel from file
        for line in open(f"{RD}/sigma_freq_debug.dat"):
            p=line.split()
            if len(p)>=14 and p[0].isdigit() and int(p[0])==ik and int(p[1])==b:
                edft_rel=float(p[4]); break
        print(f"\n===== {name} CBM (ik={ik}, n=26), Edft-Ef={edft_rel:+.3f} eV =====")
        print("  omega   Re(Sig_c)   Im(Sig_c)")
        for w,v in zip(omega,sc):
            mark = " <-- eval bracket" if (edft_rel is not None and abs(w-edft_rel)<0.5) else ""
            print(f"  {w:+6.2f}  {v.real:+11.3f}  {v.imag:+13.2f}{mark}")
