"""Read Sigma_c(omega) from sigma_mnk.h5 and examine behavior across omega=0
(the on-shell point) for near-gap Gamma states — is it a smooth sharp pole or a
genuine discontinuity? omega grid is -10..+10 eV, 0.5 step, RELATIVE to E_dft."""
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

H5 = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/"
      "00_lorrax_gw_gnppm/sigma_mnk.h5")
f = h5py.File(H5, "r")
w = np.asarray(f["omega_ev"][:], float)                       # (41,)
Sc = f["sigma_c_kij_ev"]                                       # (41,36,100,100) compound{r,i}
i0 = int(np.argmin(np.abs(w)))                                 # omega=0 index
print(f"[grid] {w.size} pts, omega=0 at index {i0} (w={w[i0]}), dw={w[1]-w[0]}")

def get(k, n):
    a = Sc[:, k, n, n]                                         # structured {r,i} or complex
    if a.dtype.names:
        return a["r"] + 1j * a["i"]
    return a.astype(complex)

states = [(0, 25, "G VBM(b26)"), (0, 26, "G CBM(b27)"),
          (0, 24, "G b25"), (0, 27, "G b28")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
for k, n, lab in states:
    s = get(k, n)
    print(f"\n[{lab}] Sigma_c(omega) around on-shell omega=0:")
    for i in range(i0 - 3, i0 + 4):
        print(f"   w={w[i]:+4.1f}  Re={s[i].real:+11.4f}  Im={s[i].imag:+13.4f}")
    jumpR = abs(s[i0 + 1].real - s[i0 - 1].real)
    flip = np.sign(s[i0 - 1].real) != np.sign(s[i0 + 1].real)
    onshell_vs_neighbors = s[i0].real - 0.5 * (s[i0 - 1].real + s[i0 + 1].real)
    print(f"   |ReSc(+0.5)-ReSc(-0.5)|={jumpR:.4f} eV; Re sign-flip across 0: {flip}; "
          f"onshell(0) - mean(+-0.5) = {onshell_vs_neighbors:+.4f} eV")
    ax[0].plot(w, s.real, marker=".", label=lab)
    ax[1].plot(w, s.imag, marker=".", label=lab)
for a, t in zip(ax, ("Re Sigma_c(w)", "Im Sigma_c(w)")):
    a.axvline(0, color="0.6", lw=.8); a.set_xlabel("omega - E_dft (eV)")
    a.set_title(t); a.legend(fontsize=8); a.set_xlim(-6, 6)
fig.tight_layout()
out = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/"
       "_sigcheck/sigc_omega.png")
fig.savefig(out); print(f"\n[saved] {out}")
