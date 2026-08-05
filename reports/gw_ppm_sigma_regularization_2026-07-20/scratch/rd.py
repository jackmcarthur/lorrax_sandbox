import h5py, numpy as np, sys
f=sys.argv[1]
with h5py.File(f,"r") as h:
    om=h["/omega_ev"][()]; sc=h["/sigma_c_kij_ev"]
    i0=int(np.argmin(np.abs(om)))
    for b,lbl in [(25,"CBM"),(24,"VBM")]:
        d=sc[:,0,b,b]
        print(f"  {lbl}(k0,b{b}): Sc(w=0)={d[i0].real:+.2f}{d[i0].imag:+.2f}j  Re_range[{d.real.min():+.1f},{d.real.max():+.1f}]  Im_range[{d.imag.min():+.1f},{d.imag.max():+.1f}]")
