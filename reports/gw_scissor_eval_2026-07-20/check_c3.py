import numpy as np
RUN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/00_lorrax_gw_gnppm"
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
# parse eqp1 coords
ks=[];
for line in open(RUN+"/eqp1.dat"):
    p=line.split()
    if len(p)==4 and "." in p[0] and "." in p[1]:
        ks.append((float(p[0]),float(p[1]),float(p[2])))
nk=len(ks)
def col(k,b,c): return fd[(k,b)][c]
print(f"nk={nk}")
print("idx  (kx,ky,kz)          Edft(b1)   Edft(b26)  Edft(b27)   V_H(b1)     V_H(b26)   sigX(b26)  sigCre(b26) Edft-Ef(b26)")
for k in range(nk):
    kx,ky,kz=ks[k]
    print(f"{k:3d}  ({kx:.4f},{ky:.4f},{kz:.2f})  {col(k,1,'E_dft'):9.4f}  {col(k,26,'E_dft'):9.4f}  {col(k,27,'E_dft'):9.4f}  "
          f"{col(k,1,'V_H'):9.3f}  {col(k,26,'V_H'):9.3f}  {col(k,26,'x_bare'):8.3f}  {col(k,26,'sig_c(Edft).Re'):8.3f}  {col(k,26,'Edft-Ef'):8.3f}")

# Build C3 star check: hexagonal C3 on crystal coords (h,k)->(-k, h-k) mod 1
def wrap(x): return x-np.round(x)
def c3(kc):
    h,kk,l=kc; return (wrap(-kk), wrap(h-kk), l)
def key(kc): return (round(wrap(kc[0]),4), round(wrap(kc[1]),4), round(kc[2],4))
kidx={key(kc):i for i,kc in enumerate(ks)}
print("\nC3-star check (does C3 map to a grid point, and do DFT/V_H match?):")
seen=set()
for k in range(nk):
    if k in seen: continue
    star=[k]; cur=ks[k]
    for _ in range(2):
        cur=c3(cur); j=kidx.get(key(cur))
        if j is not None and j not in star: star.append(j)
    # add TRS
    for s in list(star):
        neg=(wrap(-ks[s][0]),wrap(-ks[s][1]),ks[s][2]); j=kidx.get(key(neg))
        if j is not None and j not in star: star.append(j)
    for s in star: seen.add(s)
    if len(star)>1:
        ed26=[col(s,26,'E_dft') for s in star]
        vh26=[col(s,26,'V_H') for s in star]
        sx26=[col(s,26,'x_bare') for s in star]
        sc26=[col(s,26,'sig_c(Edft).Re') for s in star]
        print(f" star {star}: dE_dft(b26) spread={max(ed26)-min(ed26):.4f}  "
              f"dV_H spread={max(vh26)-min(vh26):.4f}  dsigX spread={max(sx26)-min(sx26):.4f}  "
              f"dsigCre spread={max(sc26)-min(sc26):.4f} eV")
