import h5py, numpy as np
np.set_printoptions(suppress=True, linewidth=160)

RD="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu"

# --- k-point crystal coords from WFN ---
with h5py.File(f"{RD}/WFN.h5","r") as f:
    rk = f["/mf_header/kpoints/rk"][()]   # (nrk,3) or (3,nrk)
    el = f["/mf_header/kpoints/el"][()]   # (1,36,210) Ry
if rk.shape[0]==3 and rk.shape[1]!=3:
    rk = rk.T
nk = rk.shape[0]
def frac(x):
    x=np.mod(x+1e-6,1.0)-1e-6
    return x
# identify Gamma, K(1/3,1/3), M(1/2,0)
targets={"Gamma":(0,0,0),"K":(1/3,1/3,0),"M":(0.5,0,0),"Kp":(2/3,2/3,0),"Mp":(0,0.5,0)}
kidx={}
for name,t in targets.items():
    d=np.min([np.linalg.norm(frac(rk-np.array(t)),axis=1),
              np.linalg.norm(frac(rk+np.array(t)),axis=1)],axis=0)
    kidx[name]=int(np.argmin(d))
print("k index map (crystal coords):")
for name,i in kidx.items():
    print(f"  {name:6s} ik={i:2d} rk={rk[i]}")

# --- sigma_mnk.h5 ---
with h5py.File(f"{RD}/sigma_mnk.h5","r") as f:
    omega = f["/omega_ev"][()]                 # (41,)
    print("\nomega_ev grid:", omega)
    # diagonals we need: bands 24,25,26,27 (0-indexed within 100 sigma bands)
    bands=[24,25,26,27]
    for kn in ["Gamma","K","M"]:
        ik=kidx[kn]
        print(f"\n===== k={kn} (ik={ik}) =====")
        sx = f["/sigma_sx_kij_ev"][ik]         # (100,100) static SX
        # E_dft rel to Ef for these bands: read from el (Ry) minus Ef.
        for b in bands:
            sc_w = f["/sigma_c_kij_ev"][:,ik,b,b]   # (41,) complex
            print(f"  band n={b} (phys {b+1}): static SX_diag={sx[b,b].real:+.4f} eV")
            print(f"    Re Sigma_c(w) over grid:")
            for iw,w in enumerate(omega):
                print(f"      w={w:+6.2f}  Re={sc_w[iw].real:+10.4f}  Im={sc_w[iw].imag:+12.3f}")
