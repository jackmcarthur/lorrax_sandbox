import h5py, numpy as np, os
f=f"{os.environ['RPT']}/scratch/dbg_nband210/sigma_mnk.h5"
with h5py.File(f,"r") as h:
    om=h["/omega_ev"][()]; sc=h["/sigma_c_kij_ev"][:,0,25,25]
print("1-GPU nband=210 CBM (k0,b25) on-grid Sigma_c(w) [incl head]:")
for w,v in zip(om,sc):
    mk=" <-- near Edft(+1.82)" if abs(w-1.818)<0.3 else ""
    print(f"  w={w:+6.2f}  Re={v.real:+9.3f}  Im={v.imag:+10.2f}{mk}")
