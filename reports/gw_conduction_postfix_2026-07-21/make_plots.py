"""Before/after figures for the post-fix conduction audit."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_conduction_postfix_2026-07-21"
P = os.path.join(R, "plots")
os.makedirs(P, exist_ok=True)

pre = np.load(os.path.join(R, "pre/scissor_data.npz"), allow_pickle=True)
post = np.load(os.path.join(R, "post/scissor_data.npz"), allow_pickle=True)
vac = np.load(os.path.join(R, "vacuum_weight.npz"))


def zfac(d):
    s0, s1 = d['scissor0'], d['scissor1']
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(s0) > 1e-9, s1 / s0, np.nan)


# ------------------------------------------------------------------ figure 1
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), dpi=140)

# (a) Z factor vs E_dft-EF
for a, d, ttl, c in [(ax[0], pre, "PRE (Cholesky CCT + raw quadrature)", "#b5432c"),
                     (ax[0], post, "POST (rank_truncate 1e-6 + $\\xi$-floor)", "#2c6e8f")]:
    Z = zfac(d)
    a.scatter(d['Erel'].ravel(), Z.ravel(), s=7, alpha=.5, color=c, label=ttl)
ax[0].axhline(1, color='0.6', lw=.8)
ax[0].axhline(0, color='0.6', lw=.8, ls=':')
ax[0].set_yscale('symlog', linthresh=2)
ax[0].set_xlabel("$E_{DFT}-E_F$ (eV)")
ax[0].set_ylabel("$Z = $ scissor1 / scissor0")
ax[0].set_title("(a) PPM Z-factor pole\nPRE: 38 blow-ups to $Z$=+57 / $-$0.61   POST: 0, $Z\\in[0.72,1.00]$",
                fontsize=9)
ax[0].legend(fontsize=7, loc='upper left')
ax[0].grid(alpha=.3)

# (b) scissor1 vs band
nb = pre['scissor1'].shape[1]
for a, d, ttl, c in [(ax[1], pre, "PRE", "#b5432c"), (ax[1], post, "POST", "#2c6e8f")]:
    a.plot(range(nb), np.abs(d['scissor1']).max(axis=0), 'o-', ms=3, lw=1, color=c, label=ttl)
ax[1].set_yscale('log')
ax[1].axvline(46, color='purple', ls=':', lw=1)
ax[1].text(46.7, 1e2, "far-cond\n(vacuum bands)", fontsize=7, color='purple')
ax[1].axvline(38, color='k', ls=':', lw=1)
ax[1].text(30, 3e2, "b=38 Z-pole", fontsize=7)
ax[1].set_xlabel("band index")
ax[1].set_ylabel("max$_k$ |scissor1| (eV)")
ax[1].set_title("(b) where the scissor is unphysical\nin-window CURED; b$\\gtrsim$46 UNCHANGED", fontsize=9)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=.3)

# (c) the separate defect: |dVxc| vs vacuum weight
sc = ax[2].scatter(vac['f_vac'].ravel(), vac['err'].ravel(),
                   c=np.tile(np.arange(nb), (vac['f_vac'].shape[0], 1)).ravel(),
                   s=8, cmap='viridis', alpha=.7)
ax[2].set_xlabel("$f_{vac}$ = fraction of $|\\psi_{nk}|^2$ in the slab vacuum")
ax[2].set_ylabel("$|\\Delta V_{xc}|$ = |(kin_ion+$V_H$) $-$ KIH$_{QE}$| (eV)")
ax[2].set_title("(c) the SEPARATE defect: $V_H$ centroid quadrature\ncorr = +0.958 (POST run; PRE identical)", fontsize=9)
fig.colorbar(sc, ax=ax[2], label="band index")
ax[2].grid(alpha=.3)

fig.suptitle("MoS$_2$ 6$\\times$6 GN-PPM, 1496 recovered-D3h centroids, 16 GPU — identical config, "
             "only the LORRAX self-energy source differs", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(P, "01_conduction_prepost.png"))
print("saved 01_conduction_prepost.png")

# ------------------------------------------------------------------ figure 2
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
zp, zq = zfac(pre), zfac(post)
bins = np.linspace(-1, 2, 80)
ax[0].hist(np.clip(zp.ravel(), -1, 2), bins=bins, alpha=.6, color="#b5432c", label="PRE")
ax[0].hist(np.clip(zq.ravel(), -1, 2), bins=bins, alpha=.6, color="#2c6e8f", label="POST")
ax[0].set_yscale('log')
ax[0].set_xlabel("$Z$ (clipped to [-1,2])")
ax[0].set_ylabel("count")
ax[0].set_title("Z distribution, all 3600 (k,n)", fontsize=10)
ax[0].legend()
ax[0].grid(alpha=.3)

VH, ki, KIH = vac['V_H'], vac['kin_ion'], vac['KIH_ref']
fv = vac['f_vac']
m = fv > 0.30
ax[1].scatter(ki[~m], VH[~m], s=8, color='#2c6e8f', alpha=.5, label="$f_{vac}<0.3$ (slab)")
ax[1].scatter(ki[m], VH[m], s=14, color='#b5432c', alpha=.8, label="$f_{vac}>0.3$ (vacuum)")
x = np.linspace(ki.min(), ki.max(), 10)
ax[1].plot(x, 25.0 - x, 'k--', lw=1, label="$V_H$ that would give KIH$_{QE}\\approx$25 eV")
ax[1].set_xlabel("kin_ion $=\\langle T+V_{ion}+V_{NL}\\rangle$ (eV)")
ax[1].set_ylabel("$V_H$ (ISDF centroid quadrature, eV)")
ax[1].set_ylim(-250, 650)
ax[1].set_title("vacuum bands need $V_H<0$; the centroid\nquadrature can only return $V_H>0$", fontsize=10)
ax[1].legend(fontsize=7)
ax[1].grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(P, "02_z_hist_and_vh_mechanism.png"))
print("saved 02_z_hist_and_vh_mechanism.png")
