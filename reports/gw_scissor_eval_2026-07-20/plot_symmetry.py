import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
RUN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/00_lorrax_gw_gnppm"
OUT="/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_scissor_eval_2026-07-20"
cols=None; fd={}
for line in open(RUN+"/sigma_freq_debug.dat"):
    s=line.strip()
    if s.startswith("#"):
        p=s.lstrip("#").split()
        if len(p)>=3 and p[0]=="k" and p[1]=="n": cols=p[2:]
        continue
    if not s or cols is None: continue
    p=s.split()
    if len(p)!=len(cols)+2: continue
    try: k,n=int(p[0]),int(p[1])
    except: continue
    fd[(k,n+1)]={c:(np.nan if v=="nan" else float(v)) for c,v in zip(cols,p[2:])}
eqp1={}; ks=[]; ik=-1
for line in open(RUN+"/eqp1.dat"):
    p=line.split()
    if len(p)==4 and "." in p[0] and "." in p[1]:
        ik+=1; ks.append((float(p[0]),float(p[1]),float(p[2])))
    elif len(p)==4: eqp1[(ik,int(p[1]))]=(float(p[2]),float(p[3]))
def g(k,b,c): return fd[(k,b)][c]

# VBM 6-star
star=[k for k in range(len(ks)) if abs(g(k,26,'E_dft')+5.6829)<0.02]
labs=[f"k{k}\n({ks[k][0]:.2f},{ks[k][1]:.2f})" for k in star]
VH=[g(k,26,'V_H') for k in star]
EQ=[eqp1[(k,26)][1] for k in star]
XB=[g(k,26,'x_bare') for k in star]
ED=[g(k,26,'E_dft') for k in star]

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(14,4.6),dpi=140)
ax1.bar(range(len(star)),VH,color="#c0392b"); ax1.set_title(f"V_H(band 26)  spread={max(VH)-min(VH):.2f} eV\n(BREAKS C3; TRS pairs identical)")
ax1.set_xticks(range(len(star))); ax1.set_xticklabels(labs,fontsize=7); ax1.set_ylabel("eV")
ax1.set_ylim(min(VH)-0.5,max(VH)+0.5)
ax2.bar(range(len(star)),XB,color="#2c7fb8"); ax2.set_title(f"x_bare(band 26)  spread={max(XB)-min(XB):.3f} eV\n(SYMMETRIC — psi/ISDF fine)")
ax2.set_xticks(range(len(star))); ax2.set_xticklabels(labs,fontsize=7); ax2.set_ylabel("eV")
ax2.set_ylim(min(XB)-0.1,max(XB)+0.1)
ax3.bar(range(len(star)),EQ,color="#8e44ad"); ax3.axhline(np.mean(EQ),color="k",ls=":")
ax3.set_title(f"eqp1(band 26)=QP VBM  spread={max(EQ)-min(EQ):.2f} eV\n(gap-defining VBM takes 3 values!)")
ax3.set_xticks(range(len(star))); ax3.set_xticklabels(labs,fontsize=7); ax3.set_ylabel("eV")
fig.suptitle("MoS2 VBM (band 26) across its 6-fold symmetry star — all k have identical E_dft=-5.683 eV",fontsize=11)
fig.tight_layout(); fig.savefig(OUT+"/symmetry_break_vbm.png")
print("star",star,"VH",[round(v,3) for v in VH],"eqp",[round(v,3) for v in EQ])
