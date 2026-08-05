"""Compare two exciton-bands .dat files per-Q (interp rows only).

Usage: python3 compare_dat.py A.dat B.dat [--label-a NAME --label-b NAME]
Prints per-Q max|ΔE| (meV) over the n_eig eigenvalues, the global
max/mean, and the worst-Q.  Used for:
  * 16-GPU cusolverMp OFF  vs dir-10 4-GPU native OFF   (gate #2, must be tiny)
  * 16-GPU ON             vs 16-GPU OFF                 (flag-on head shift)
Saves the per-Q deltas to <B>.cmp.npz.
"""
import sys
import numpy as np


def load_dat(path):
    """Return (iQ int[nQ], s_path[nQ], Q[nQ,3], E[nQ, n_eig]) for interp rows."""
    iQ, s, Q, E = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            t = line.split()
            # iQ s Qx Qy Qz mode E1..En
            if t[5] != "interp":
                continue
            iQ.append(int(t[0]))
            s.append(float(t[1]))
            Q.append([float(t[2]), float(t[3]), float(t[4])])
            E.append([float(x) for x in t[6:]])
    order = np.argsort(iQ)
    iQ = np.asarray(iQ)[order]
    return (iQ, np.asarray(s)[order], np.asarray(Q)[order],
            np.asarray(E)[order])


def main():
    a, b = sys.argv[1], sys.argv[2]
    la = sys.argv[sys.argv.index("--label-a") + 1] if "--label-a" in sys.argv else a
    lb = sys.argv[sys.argv.index("--label-b") + 1] if "--label-b" in sys.argv else b
    iA, sA, QA, EA = load_dat(a)
    iB, sB, QB, EB = load_dat(b)
    assert np.array_equal(iA, iB), "Q-index sets differ"
    assert EA.shape == EB.shape, f"shape {EA.shape} vs {EB.shape}"

    dE = (EB - EA)              # eV
    dE_meV = dE * 1e3
    per_q_max = np.max(np.abs(dE_meV), axis=1)   # (nQ,) meV
    gmax_meV = float(np.max(np.abs(dE_meV)))
    gmean_meV = float(np.mean(np.abs(dE_meV)))
    worst = int(np.argmax(per_q_max))

    print(f"# compare  A={la}\n#          B={lb}")
    print(f"# {EA.shape[0]} Q-points, {EA.shape[1]} eigenvalues each")
    print(f"# per-Q max|ΔE| (meV):  E_1 lowest exciton and the full n_eig block")
    print(f"{'iQ':>4} {'s_path':>9} {'|Q|':>8}  {'max|dE|(meV)':>13} {'dE_E1(meV)':>11}")
    for j in range(EA.shape[0]):
        qn = float(np.linalg.norm(QA[j]))
        print(f"{iA[j]:4d} {sA[j]:9.5f} {qn:8.5f}  {per_q_max[j]:13.4e} "
              f"{dE_meV[j,0]:11.4e}")
    print(f"\n# GLOBAL max|ΔE| = {gmax_meV:.4e} meV = {gmax_meV*1e-3:.4e} eV "
          f"(worst at iQ={iA[worst]}, s={sA[worst]:.4f}, |Q|={np.linalg.norm(QA[worst]):.5f})")
    print(f"# GLOBAL mean|ΔE| = {gmean_meV:.4e} meV")
    print(f"# E_1(Γ=iQ0): A={EA[0,0]:.6f}  B={EB[0,0]:.6f}  ΔE1(Γ)={dE_meV[0,0]:.4e} meV")

    out = b + ".cmp.npz"
    np.savez(out, iQ=iA, s_path=sA, Q=QA, EA=EA, EB=EB,
             dE_meV=dE_meV, per_q_max_meV=per_q_max,
             gmax_meV=gmax_meV, gmean_meV=gmean_meV,
             label_a=la, label_b=lb)
    print(f"# saved {out}")


if __name__ == "__main__":
    main()
