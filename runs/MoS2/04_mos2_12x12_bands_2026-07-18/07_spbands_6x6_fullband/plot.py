import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RY=13.6056980659
d=np.loadtxt("bandstructure.dat")
ik=d[:,0].astype(int); ib=d[:,1].astype(int); s=d[:,5]; e=d[:,6]*RY
nk=ik.max()+1; nb=ib.max()+1
E=np.full((nk,nb),np.nan)
for a,b,x,en in zip(ik,ib,s,e): E[a,b]=en
x=np.array([s[ik==a][0] for a in range(nk)])
vbm=E[:,:52][~np.isnan(E[:,:52])].max() if nb>52 else np.nanmax(E[E<0]) if (E<0).any() else 0
# 26 valence spinor-pairs = 52 spinor states? nb=40 bands requested; spinor bands: nval=26 are spinor states
vbm=np.nanmax(E[:,25])  # band idx 25 = 26th state = VBM (spinor convention)
fig,ax=plt.subplots(figsize=(7,5.5),dpi=150)
for b in range(nb):
    c='#2c6e8f' if b<=25 else '#b5432c'
    ax.plot(x,E[:,b]-vbm,lw=1.1,color=c)
# high-sym: nodes at segment boundaries 0,14,21,33 per 15/8/13 counts
seg=[0,14,21,x.size-1]; lbl=['Γ','M','K','Γ']
for i in seg: ax.axvline(x[i],color='0.75',lw=.6,zorder=0)
ax.set_xticks([x[i] for i in seg]); ax.set_xticklabels(lbl)
ax.set_ylabel("E − E$_{VBM}$ (eV)"); ax.set_xlim(x[0],x[-1]); ax.set_ylim(-8,6)
ax.set_title("MoS₂ htransform SP bands — 3×3 coarse grid, full basis (26v+14c), DFT energies")
fig.tight_layout(); fig.savefig("sp_bands_3x3_fullband.png")
print("saved", nk, nb)
