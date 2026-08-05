import numpy as np
RUN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/00_lorrax_gw_gnppm"
OUT="/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_scissor_eval_2026-07-20"
# parse freq_debug v2
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
# parse eqp1
eqp1={}; ks=[]; ik=-1
for line in open(RUN+"/eqp1.dat"):
    p=line.split()
    if len(p)==4 and "." in p[0] and "." in p[1]:
        ik+=1; ks.append((float(p[0]),float(p[1]),float(p[2])))
    elif len(p)==4:
        eqp1[(ik,int(p[1]))]=(float(p[2]),float(p[3]))
nk=len(ks); nband=max(b for (_,b) in eqp1)

def g(k,b,c): return fd[(k,b)][c]
def vxc(k,b): return g(k,b,'E_dft')-g(k,b,'kin_ion')-g(k,b,'V_H')
def scis(k,b): return eqp1[(k,b)][1]-eqp1[(k,b)][0]

# Identify stars: group k by full E_dft spectrum rounded to 1e-3 (loose enough to
# keep C3 partners together despite tiny high-band reordering).
def spec(k): return tuple(round(g(k,b,'E_dft'),3) for b in (1,10,26,27,50))
stars={}
for k in range(nk): stars.setdefault(spec(k),[]).append(k)
multi=sorted([v for v in stars.values() if len(v)>1], key=lambda c:-len(c))

L=[]
L.append("=== PER-STAR COMPONENT DECOMPOSITION (stars = identical DFT spectrum) ===")
L.append("Within a star all k are symmetry-equivalent => every quantity MUST be identical.")
L.append("Spread = max-min across the star. Reporting the WORST band per star for each quantity.\n")
def worst_spread(karr, fn):
    w=0.0; wb=-1
    for b in range(1,nband+1):
        vals=np.array([fn(k,b) for k in karr])
        sp=np.nanmax(vals)-np.nanmin(vals)
        if np.isfinite(sp) and sp>w: w=sp; wb=b
    return w,wb
agg={q:0.0 for q in ("E_dft","kin_ion","V_H","kin+VH","Vxc","x_bare","sigc_re","sigc_im","scissor")}
fns=dict(E_dft=lambda k,b:g(k,b,'E_dft'),
         kin_ion=lambda k,b:g(k,b,'kin_ion'),
         V_H=lambda k,b:g(k,b,'V_H'),
         **{"kin+VH":lambda k,b:g(k,b,'kin_ion')+g(k,b,'V_H')},
         Vxc=vxc,
         x_bare=lambda k,b:g(k,b,'x_bare'),
         sigc_re=lambda k,b:g(k,b,'sig_c(Edft).Re'),
         sigc_im=lambda k,b:abs(g(k,b,'sig_c(Edft).Im')),
         scissor=scis)
for si,karr in enumerate(multi):
    L.append(f"Star {si}: k={karr} mult={len(karr)} coords={[tuple(round(x,3) for x in ks[k]) for k in karr]}")
    for q,fn in fns.items():
        sp,wb=worst_spread(karr,fn)
        agg[q]=max(agg[q],sp)
        tag=" <==BREAKS" if (q in("V_H","kin_ion","kin+VH","Vxc","scissor","x_bare") and sp>0.05) else ""
        L.append(f"    {q:9s} spread={sp:10.4f} eV (band {wb}){tag}")
    L.append("")
L.append("=== WORST SPREAD OVER ALL STARS (eV) ===")
for q in fns: L.append(f"  {q:9s}: {agg[q]:.4f}")
L.append("")
L.append("INTERPRETATION:")
L.append(f"  E_dft spread {agg['E_dft']:.4f} (should be ~0: confirms stars are truly equivalent)")
L.append(f"  x_bare (ISDF bare exchange) spread {agg['x_bare']:.4f}")
L.append(f"  V_H  spread {agg['V_H']:.4f} ; kin_ion spread {agg['kin_ion']:.4f} ; kin+VH spread {agg['kin+VH']:.4f}")
L.append(f"  Vxc  spread {agg['Vxc']:.4f} ; final SCISSOR spread {agg['scissor']:.4f}")
open(OUT+"/star_decomp.txt","w").write("\n".join(L)+"\n")
print("\n".join(L))

# Focused table for the -5.6829 star (the one the coordinator cited)
print("\n=== FOCUS: VBM (band 26) across its 6-fold star ===")
target=[k for k in range(nk) if abs(g(k,26,'E_dft')-(-5.6829))<0.02]
print("k   coord            E_dft    kin_ion     V_H     kin+VH     Vxc     x_bare  sigc_re  sigc_im   eqp1    scissor")
for k in target:
    print(f"{k:2d} {tuple(round(x,3) for x in ks[k])}  {g(k,26,'E_dft'):8.4f} {g(k,26,'kin_ion'):9.3f} {g(k,26,'V_H'):8.3f} "
          f"{g(k,26,'kin_ion')+g(k,26,'V_H'):8.3f} {vxc(k,26):8.3f} {g(k,26,'x_bare'):7.3f} {g(k,26,'sig_c(Edft).Re'):7.3f} "
          f"{g(k,26,'sig_c(Edft).Im'):8.2f} {eqp1[(k,26)][1]:8.3f} {scis(k,26):7.3f}")
