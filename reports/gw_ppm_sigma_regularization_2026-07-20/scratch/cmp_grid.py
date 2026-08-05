import h5py, numpy as np
for tag,f in [("MY_RERUN", f"{__import__('os').environ['RPT']}/scratch/dbg_nband210/sigma_mnk.h5"),
              ("ORIG_DIAG","/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_sigcheck/diag_nband210/sigma_mnk.h5")]:
    with h5py.File(f,"r") as h:
        om=h["/omega_ev"][()]
        sc=h["/sigma_c_kij_ev"]
        i0=int(np.argmin(np.abs(om)))
        d=sc[:,0,25,25]
        print(f"{tag}: Sc_CBM(w=0)= {d[i0].real:+.3f}{d[i0].imag:+.3f}j  | range Re[{d.real.min():+.1f},{d.real.max():+.1f}] Im[{d.imag.min():+.1f},{d.imag.max():+.1f}]")
