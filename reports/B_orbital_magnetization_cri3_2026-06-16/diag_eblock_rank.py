"""Localize the extra D-matrix rank inside vnl_ops: for each pseudo, print the
j-groups per l, rank(f_blocks_lj) vs expected 2j+1, and rank of the per-l
E-block vs expected Sum_j rank(D_sub_j)*(2j+1). Extra rank => the SOC spin-
angular assembly over-couples."""
import os, sys, numpy as np
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from psp.pseudos import load_pseudopotentials
from psp.radial.build_projectors_qe import (build_E_blocks_full, f_blocks_lj,
                                             U_complex_from_real)

PDIR = sys.argv[1]
pseudos = load_pseudopotentials(PDIR)
for elem, pseudo in pseudos.items():
    print(f"\n===== {elem} =====")
    ppnl = pseudo.pp_nonlocal
    betas = list(ppnl.pp_beta)
    # j-groups per l
    per_l = {}
    for idx, b in enumerate(betas):
        l = int(getattr(b,'lll',getattr(b,'angular_momentum',0)))
        j = float(getattr(b,'jjj', l+0.5))
        per_l.setdefault(l, []).append(j)
    for l in sorted(per_l):
        js = per_l[l]
        from collections import Counter
        jc = Counter(round(x,1) for x in js)
        print(f"  l={l} msize={2*l+1}: betas-by-j {dict(jc)}")
        U_l = U_complex_from_real(l)
        for j in sorted(jc):
            f = f_blocks_lj(l, float(j), U_l)            # (2,2,m,m)
            M = np.transpose(f,(0,2,1,3)).reshape(2*(2*l+1), 2*(2*l+1))
            r = np.linalg.matrix_rank(M, tol=1e-8)
            print(f"      f_blocks_lj(l={l},j={j}): rank {r}  expected 2j+1={int(2*j+1)}  {'OK' if r==int(2*j+1) else '*** MISMATCH ***'}")

    E_blocks = build_E_blocks_full(pseudo)
    tot = 0
    for l in sorted(E_blocks):
        E = E_blocks[l]; R = E.shape[-1]
        M = np.transpose(E,(0,2,1,3)).reshape(2*R, 2*R)
        r = np.linalg.matrix_rank((M+M.conj().T)/2, tol=1e-8)
        tot += r
        print(f"  E-block l={l}: size 2x{R}={2*R}  nonzero-rank {r}")
    print(f"  TOTAL nonzero rank for {elem}: {tot}")
