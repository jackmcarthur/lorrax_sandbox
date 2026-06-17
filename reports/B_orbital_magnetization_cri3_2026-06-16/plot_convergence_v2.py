"""Talk plots: CrI3 orbital moment convergence. (a) vs band count in the SOS sum
(6x6 IBZ cascade, with LC-vs-IC split), (b) vs k-grid (velocity-matrix npz, N=180).
Plotted as the moment ALONG the spin axis: negative = antiparallel (Hund)."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REP="/pscratch/sd/j/jackm/lorrax_sandbox/reports/B_orbital_magnetization_cri3_2026-06-16"

# ---- (a) band-count convergence from IBZ cascade (mu-linear columns colA - 2 mu colB) ----
d=np.load(f"{REP}/orbmag_FM_6x6_4000.npz",allow_pickle=True)
s=np.sign(float(d['m_spin_z']))                      # spin axis sign (-1 here)
cA,cB=np.asarray(d['colA_z']),np.asarray(d['colB_z']); E=np.asarray(d['E']); nocc=int(d['nocc'])
VBM=E[:,nocc-1].max(); CBM=E[:,nocc].min(); mu=0.5*(VBM+CBM)   # midgap
N=np.arange(1,len(cA)+1)
mTOT=s*(-0.5)*np.cumsum(cA - 2.0*mu*cB).imag          # m_z(N) along spin axis (mu_B/cell)
print(f"6x6 band-conv: m_z(180)={mTOT[179]:+.4f}  m_z(4000)={mTOT[-1]:+.4f}  (stored m_orb_z*s={s*float(d['m_orb'][2]):+.4f})")
cross=N[np.where(np.diff(np.sign(mTOT)))[0]] if np.any(np.diff(np.sign(mTOT))) else []
print("  zero-crossing(s) at N =", cross[:3])

# ---- (b) k-grid convergence from velocity matrices at N=180, physical sign +1 ----
def msum(npz,Nb):
    d=np.load(npz,allow_pickle=True); Vp,Vnl,E=d['Vp'],d['Vnl'],d['E']
    mu=float(d['mu']); nocc=int(d['nocc']); wk=float(d['w_k']); nb=E.shape[1]; Nb=min(Nb,nb)
    v=Vp+1.0*Vnl                                      # FORCE physical sign +1
    tot=0.0
    for k in range(E.shape[0]):
        A=v[k,0,:,:Nb]*v[k,1,:Nb,:].T - v[k,1,:,:Nb]*v[k,0,:Nb,:].T   # (nb,Nb) z-cross
        En=E[k][:,None]; Em=E[k][None,:Nb]; dE=En-Em
        F=np.where(np.abs(dE)>1e-6,(Em+En-2*mu)/np.where(np.abs(dE)>1e-6,dE,1.0)**2,0.0)
        tot += wk*np.sum((-0.5*np.imag(np.sum(A*F,axis=1)))[:nocc])
    return tot, np.sign(float(d['m_spin_z'])), float(d['m_orb'][2])
# calibrate prefactor on 10x10 (msum full should match stored m_orb)
raw10,s10,mo10=msum(f"{REP}/orbmag_FM_10x10.npz",10**9)
pref=mo10/raw10 if abs(raw10)>1e-12 else 1.0
grids=[("6x6",36,"orbmag_FM_nbnd180.npz"),("8x8",64,"orbmag_FM_8x8.npz"),("10x10",100,"orbmag_FM_10x10.npz")]
mk=[]
for nm,nk,f in grids:
    r,sg,_=msum(f"{REP}/{f}",180); mk.append(pref*r*sg)   # *sg -> along spin axis
print("k-grid m_z (along spin):",[f"{x:+.3f}" for x in mk])

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.3))
vis=np.abs(mTOT)<0.15            # mask the intra-occupied near-degenerate transient (cancels by N=nocc)
ax1.plot(N[vis],mTOT[vis],color='crimson',lw=1.8)
ax1.axhline(0,color='gray',lw=0.7,ls='--'); ax1.set_xscale('log'); ax1.set_xlim(60,4000); ax1.set_ylim(-0.095,0.045)
ax1.axvspan(60,180,color='0.92',label='N≤180 (k-grid panel)')
ax1.set_xlabel("# bands in SOS sum"); ax1.set_ylabel(r"$m_z^{\rm orb}$ along spin ($\mu_B$/cell)")
ax1.set_title("(a) vs band count (6$\\times$6) — slow tail"); ax1.legend(fontsize=8,loc='upper right')
ax1.plot([180],[mTOT[179]],'o',color='0.4'); ax1.annotate(f"+{mTOT[179]:.3f} (180 b)",(180,mTOT[179]),textcoords="offset points",xytext=(8,6),fontsize=8,color='0.4')
ax1.plot([N[-1]],[mTOT[-1]],'o',color='crimson'); ax1.annotate(f"{mTOT[-1]:+.3f} (4000 b)",(N[-1],mTOT[-1]),textcoords="offset points",xytext=(-95,8),fontsize=8,color='crimson')
ax1.text(0.5,0.12,"crosses zero → grows ANTIPARALLEL (Hund)",transform=ax1.transAxes,ha='center',fontsize=8,color='crimson')
ax2.plot([g[1] for g in grids],mk,'s-',color='navy',ms=8)
ax2.axhline(0,color='gray',lw=0.6); ax2.set_xticks([g[1] for g in grids]); ax2.set_xticklabels([g[0] for g in grids])
ax2.set_xlabel("k-grid"); ax2.set_ylabel(r"$m_z^{\rm orb}$ along spin ($\mu_B$/cell)")
ax2.set_title("(b) vs k-grid (fixed N=180 b) — flat/fast"); ax2.set_ylim(0,0.03)
for x,y in zip([g[1] for g in grids],mk): ax2.annotate(f"{y:+.3f}",(x,y),textcoords="offset points",xytext=(5,6))
fig.suptitle("Monolayer CrI$_3$ (FM): orbital-moment convergence — band count is the bottleneck (slow SOS tail), k-grid is fast",y=1.02,fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{REP}/cri3_orbmag_convergence.png",dpi=200,bbox_inches='tight')
print("saved",f"{REP}/cri3_orbmag_convergence.png")
