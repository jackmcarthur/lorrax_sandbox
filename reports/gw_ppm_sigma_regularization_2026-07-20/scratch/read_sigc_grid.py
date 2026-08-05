import h5py, numpy as np, sys
np.set_printoptions(suppress=True, linewidth=200)
base="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_sigcheck"
for nb in [100, 210]:
    f=f"{base}/diag_nband{nb}/sigma_mnk.h5"
    with h5py.File(f,"r") as h:
        keys=list(h.keys())
        omega=h["/omega_ev"][()]
        sc=h["/sigma_c_kij_ev"]  # (nw, nk, nb, nb)
        print(f"\n########## nband={nb}  keys={keys} sc.shape={sc.shape} ##########")
        # CBM at Gamma: k=0, band index 25 (phys 26, 0-indexed)
        for (ik,bn,lbl) in [(0,25,"Gamma CBM n26"),(0,24,"Gamma VBM n25")]:
            d=sc[:,ik,bn,bn]
            print(f"  --- {lbl} (ik={ik}, b={bn}) on-grid Sigma_c(w) ---")
            for w,v in zip(omega,d):
                flag=" <<<" if abs(v.imag)>50 or abs(v.real)>50 else ""
                print(f"    w={w:+6.2f}  Re={v.real:+12.3f}  Im={v.imag:+14.2f}{flag}")
