import re, sys, numpy as np
# parse sigma_diag.dat -> dict[(k,n)] = (reSigC, imSigC)
def parse(path):
    d={}; k=None
    for ln in open(path):
        m=re.match(r'\s*k-point\s+(\d+):',ln)
        if m: k=int(m.group(1)); continue
        m=re.match(r'\s*n=(\d+)\s+sigX=\s*([-\d.]+)\s+sigC=\s*([-\d.]+)\+\s*([-\d.eE]+)i',ln)
        if m and k is not None:
            n=int(m.group(1)); d[(k,n)]=(float(m.group(3)),float(m.group(4)))
    return d
path=sys.argv[1]; label=sys.argv[2] if len(sys.argv)>2 else path
d=parse(path)
KP={"Gamma":0,"K":14,"M":18}
print(f"### {label}")
for nm,ik in KP.items():
    for n in (25,26):  # VBM, CBM
        if (ik,n) in d:
            re_,im_=d[(ik,n)]
            print(f"  {nm:6s} band{n} ({'VBM' if n==25 else 'CBM'}): ReSigC={re_:+9.3f}  ImSigC={im_:+12.3f} eV")
# max |Im| over conduction bands (26..99) all k
ims=[abs(v[1]) for (k,n),v in d.items() if n>=26]
res=[v[0] for (k,n),v in d.items() if n>=26]
print(f"  conduction(n>=26): max|ImSigC|={max(ims):.3e} eV   max|ReSigC|={max(abs(x) for x in res):.3f} eV   median|ImSigC|={np.median(ims):.3f} eV")
