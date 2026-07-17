import numpy as np, sys
from file_io import WfnLoader as WFNReader
from common.symmetry_maps import SymMaps
wfnpath = sys.argv[1]
wfn = WFNReader(wfnpath)
sym = SymMaps(wfn)
q = np.asarray(sym.q_irr_kgrid_int, dtype=int)
kg = np.asarray(wfn.kgrid, dtype=int)
print("kgrid =", kg.tolist())
print("n_q_ibz =", q.shape[0])
print("q_irr_kgrid_int (integer steps):")
for i,qq in enumerate(q):
    qf = qq*1.0
    qflat = int(qq[0])*int(kg[1])*int(kg[2]) + int(qq[1])*int(kg[2]) + int(qq[2])
    print(f"  i={i}  q={qq.tolist()}  C-order flat={qflat}")
print("irr_idx_q (full->ibz):", np.asarray(sym.irr_idx_q).tolist())
