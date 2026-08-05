import sys, numpy as np
def parse_eqp(path):
    out = {}; ik = -1
    for line in open(path):
        if line.startswith('#'): continue
        p = line.split()
        if len(p) == 4 and '.' in p[0]: ik += 1; continue
        if len(p) == 4: out[(ik, int(p[1])-1)] = float(p[3])
    return out
e1, e2 = parse_eqp(sys.argv[1]), parse_eqp(sys.argv[2])
ks = sorted(set(e1) & set(e2))
nk = max(k for k,_ in ks) + 1
for ik in range(nk):
    d = np.array([e2[k]-e1[k] for k in ks if k[0]==ik])
    print(f"k={ik}: max|d|={np.abs(d).max():.4e}  median={np.median(np.abs(d)):.4e}  n={len(d)}")
