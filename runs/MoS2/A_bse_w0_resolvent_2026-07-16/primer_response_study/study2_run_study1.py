"""study2_run_study1 — STUDY 1: is the b26p long-range polynomial basis
FUNDAMENTAL or empirical?

Head-to-head of the empirical monomial ladder (b26p) against principled
bandlimited/orthogonal disk bases at MATCHED coefficient count, in the exact
sec-13 physical harness (Tikhonov gauge, alpha=0.30, nR7 SR stencil, LOO
over all coarse q, verdict = gap-window B = M^H V_Q M + TDA exciton swap).

Rungs (all fit with the identical v_LR-weighted per-q normal-block LSQ; only
the per-channel design matrix Phi(K_par) changes):
  monomial(b26p)   {0:3,1:2,2:0,3:0}  26 coeff/mu   -- continuity anchor
  zernike          same degrees, disk-orthogonal polys (SAME span as monomial)
  bessel           Fourier-Bessel Neumann disk harmonics, matched count
  bessel3          bessel, angular m == 0 (mod 3)  (C3v/D3h symmetry-adapted)
  zernike3         zernike, angular m == 0 (mod 3)
  monomial(b45p)   {0:4,1:3,2:1,3:0}  richer budget, both bases
  bessel(b45p)

Measurements returned:
  (1) B med/max + exciton med/max per rung (accuracy at matched count)
  (2) per-channel normal-block conditioning + LOO coefficient stability
      (the transfer-relevant robustness the disk bases should win)
  (3) angular power spectrum of M_mu(K): fraction of |M|^2 in m == 0 (mod 3)
      harmonics, weighted, averaged over mu (owner (b), measured directly)
  (4) across-mu SVD spectrum of the fitted coefficient matrix in each basis
      (reconcile the sec-13.2 SVD-multipole failure: slow decay is a
      mu-property, orthogonal to the K-basis choice)
  (5) grid transfer 3x3-fit -> 6x6-deploy for the principled bases

Run: JID=<jid> ./proto1_run.sh python3 -u study2_run_study1.py [MoS2_6x6]
"""
import sys
import time
import numpy as np

import REFERENCE_arbitrary_q_vq as R
import study2_basis_lib as L

FIXNAME = sys.argv[1] if len(sys.argv) > 1 else "MoS2_6x6"
RY2MEV = R.RY2MEV
t00 = time.time()
NPZ = {}


def loo_B_exc(fx, prep, basis, with_exc=True, hdir_cache=None):
    """LOO over all coarse q: honest refit, nR7 SR stencil, B + exciton vs
    stored-fit truth.  hdir_cache shares (D, Hdir, ev_true) across rungs."""
    R7 = R.stencil_r7(fx)
    Bs, Es = [], []
    for q0 in range(fx["nq"]):
        train = [q for q in range(fx["nq"]) if q != q0]
        w = R.stencil_weights(fx["qfr"][train], fx["qfr"][q0], R7)
        x = R.gap_window_pairs(fx, q0)
        B_true = R.b_block(x, R.make_vq(fx, fx["ZG"][q0], q0))
        SRi = np.tensordot(w, prep["V_SRc"][train], axes=(0, 0))
        C_loo = basis.coeffs(exclude=q0)
        M = basis.model_M(C_loo, fx["qfr"][q0])
        Vp = SRi + L.lr_tile_from_M(fx, prep, M, fx["qfr"][q0])
        Bp = R.b_block(x, Vp)
        Bs.append(R.relF(Bp, B_true))
        if with_exc:
            if q0 not in hdir_cache:
                D, Hdir = R.build_hdir(fx, q0)
                hdir_cache[q0] = (D, Hdir, R.exciton_evs(fx, D, Hdir, B_true))
            D, Hdir, ev_true = hdir_cache[q0]
            ev_p = R.exciton_evs(fx, D, Hdir, Bp)
            Es.append(float(np.max(np.abs(ev_p - ev_true)) * RY2MEV))
    return np.array(Bs), (np.array(Es) if with_exc else None)


def angular_symmetry(fx, prep):
    """Weighted angular power of M_mu(K) per signed channel: decompose the
    cleaned form-factor samples over cos/sin(m theta) by weighted LSQ at
    fixed radius bins, report the mu-averaged fraction of power in m==0(mod3)
    vs the rest.  Uses the |G_z|=0 channel (the richest, degree-3 in b26p)."""
    g = 0
    cols = prep["gz_cols"][g]
    Ks, ths, ws, Ys = [], [], [], []
    for q in range(fx["nq"]):
        qG = fx["qfr"][q][:, None] + prep["GS"][:, cols].astype(float)
        Kpar = (fx["bvec"].T @ qG)[:2]
        r = np.hypot(Kpar[0], Kpar[1])
        th = np.arctan2(Kpar[1], Kpar[0])
        Ks.append(r)
        ths.append(th)
        ws.append(prep["W"][q][cols])
        Ys.append(prep["Fch"][q][:, cols])            # (nmu, m)
    r = np.concatenate(Ks)
    th = np.concatenate(ths)
    w = np.concatenate(ws)
    Y = np.concatenate(Ys, axis=1)                    # (nmu, nsamp)
    # angular harmonic decomposition at each of a few radius shells: fit
    # M(r,theta) ~ sum_m [a_m cos(m th) + b_m sin(m th)] within a shell, and
    # accumulate weighted power per |m| across shells and mu.
    rmax = r.max()
    edges = np.linspace(0.05 * rmax, rmax, 9)
    MMAX = 9
    pw_m = np.zeros(MMAX + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi)
        if sel.sum() < 2 * MMAX + 4:
            continue
        A = [np.ones(sel.sum())]
        for m in range(1, MMAX + 1):
            A.append(np.cos(m * th[sel]))
            A.append(np.sin(m * th[sel]))
        A = np.stack(A, 1)                             # (nsel, 2M+1)
        ww = w[sel]
        AtA = A.T @ (ww[:, None] * A) + 1e-9 * np.eye(A.shape[1])
        AtY = A.T @ (ww[:, None] * Y[:, sel].T)        # (2M+1, nmu)
        coef = np.linalg.solve(AtA, AtY)               # (2M+1, nmu)
        wshell = float(ww.sum())
        pw_m[0] += wshell * np.sum(np.abs(coef[0]) ** 2)
        for m in range(1, MMAX + 1):
            pw_m[m] += wshell * (np.sum(np.abs(coef[2 * m - 1]) ** 2)
                                 + np.sum(np.abs(coef[2 * m]) ** 2))
    tot = pw_m.sum()
    frac3 = (pw_m[[m for m in range(MMAX + 1) if m % 3 == 0]].sum()) / tot
    return pw_m / tot, frac3


def across_mu_svd(basis, coeffs, nsv=32):
    """SVD spectrum of the (n_coeff x n_mu) coefficient matrix in the basis's
    own weighted-Gram-orthonormal metric (block-diag per channel).  Slow
    decay = the sec-13.2 'no low rank across mu' finding, basis-independent."""
    blocks = []
    for g, (AtA, _) in basis.blocks.items():
        A = AtA.sum(0)
        A = A + 1e-12 * (np.trace(A).real / A.shape[0]) * np.eye(A.shape[0])
        Lm = np.linalg.cholesky(A)
        blocks.append(Lm.conj().T @ coeffs[g])
    Chat = np.vstack(blocks)
    s = np.linalg.svd(Chat, compute_uv=False)
    return s[:nsv]


def main():
    print(f"[study1] fixture {FIXNAME} — principled LR basis head-to-head")
    fx = R.load_fixture(FIXNAME)
    C_q = R.build_cq(fx)
    R.run_gates(fx, C_q)
    prep = R.prepare_coarse(fx, C_q)               # Tik gauge, alpha=0.30

    # ---- build all rungs ----
    B45 = {0: 4, 1: 3, 2: 1, 3: 0}
    C45 = {0: 15, 1: 10, 2: 3, 3: 1}               # monomials of deg {4,3,1,0}
    rungs = {
        "monomial_b26p": L.make_monomial(fx, prep, L.DEG_B26P),
        "zernike_b26p":  L.make_zernike(fx, prep, L.DEG_B26P),
        "bessel_b26p":   L.make_bessel(fx, prep, L.CNT_B26P),
        "bessel3_b26p":  L.make_bessel(fx, prep, L.CNT_B26P, symm3=True),
        "zernike3_b26p": L.make_zernike(fx, prep, L.DEG_B26P, symm3=True),
        "monomial_b45p": L.make_monomial(fx, prep, B45),
        "bessel_b45p":   L.make_bessel(fx, prep, C45),
    }

    # ---- (3) angular symmetry (direct measurement) ----
    pw_m, frac3 = angular_symmetry(fx, prep)
    print("\n  [angular] |G_z|=0 M_mu(K) mu-avg angular power by |m| "
          "(0..9):")
    print("    " + " ".join(f"m{m}:{pw_m[m]:.3f}" for m in range(10)))
    print(f"    power fraction in m==0(mod 3): {frac3:.3f}  "
          f"(uniform-null baseline would need m in {{0,3,6,9}} vs 10 -> 0.40)")
    NPZ["angular_pw_m"] = pw_m
    NPZ["angular_frac3"] = np.array([frac3])

    # ---- (1)+(2)+(4) per-rung accuracy, conditioning, stability, svd ----
    hdir_cache = {}
    print("\n  ===== rung ladder (LOO over all coarse q) =====")
    hdr = (f"    {'rung':<16s} {'c/mu':>5s} {'condmax':>9s} {'stab_med':>9s} "
           f"{'B med':>10s} {'B max':>10s} {'exc med':>8s} {'exc max':>8s}")
    print(hdr)
    summary = {}
    for name, basis in rungs.items():
        t0 = time.time()
        C0 = basis.coeffs()
        cond = basis.cond_blocks()
        condmax = max(cond.values())
        # LOO coeff stability (global coeffs; only which q withheld varies)
        cst = []
        for q0 in range(fx["nq"]):
            Cl = basis.coeffs(exclude=q0)
            cst.append(max(R.relF(Cl[g], C0[g]) for g in Cl))
        stab_med = float(np.median(cst))
        Bs, Es = loo_B_exc(fx, prep, basis, with_exc=True,
                           hdir_cache=hdir_cache)
        summary[name] = dict(nc=basis.n_coeff(), condmax=condmax,
                             stab=stab_med, B_med=float(np.median(Bs)),
                             B_max=float(np.max(Bs)),
                             exc_med=float(np.median(Es)),
                             exc_max=float(np.max(Es)))
        s = summary[name]
        print(f"    {name:<16s} {s['nc']:>5d} {condmax:>9.2e} "
              f"{stab_med:>9.2e} {s['B_med']:>10.3e} {s['B_max']:>10.3e} "
              f"{s['exc_med']:>8.3f} {s['exc_max']:>8.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        NPZ[f"B__{name}"] = Bs
        NPZ[f"exc__{name}"] = Es
        # (4) across-mu svd (b26p-budget rungs only, comparable coeff count)
        if name.endswith("b26p"):
            sv = across_mu_svd(basis, C0)
            NPZ[f"svd_mu__{name}"] = sv
            print(f"      across-mu svd (norm) top16: "
                  + " ".join(f"{x/sv[0]:.1e}" for x in sv[:16]))

    # ---- (5) grid transfer for the principled bases (3x3-fit -> 6x6) ----
    if FIXNAME == "MoS2_6x6":
        print("\n  ===== grid transfer 3x3-fit -> 6x6-deploy =====")
        fx3 = R.load_fixture("MoS2_3x3")
        C_q3 = R.build_cq(fx3)
        prep3 = R.prepare_coarse(fx3, C_q3)
        p3 = np.argsort(fx3["rmu_flat"])
        p6 = np.argsort(fx["rmu_flat"])
        assert np.array_equal(fx3["rmu_flat"][p3], fx["rmu_flat"][p6])
        perm = np.empty(fx["n_mu"], dtype=int)
        perm[p6] = p3
        R7 = R.stencil_r7(fx)
        for name, mk in (("monomial_b26p",
                          lambda: (L.make_monomial(fx3, prep3, L.DEG_B26P),
                                   L.make_monomial(fx, prep, L.DEG_B26P))),
                         ("bessel_b26p",
                          lambda: (L.make_bessel(fx3, prep3, L.CNT_B26P),
                                   L.make_bessel(fx, prep, L.CNT_B26P))),
                         ("zernike_b26p",
                          lambda: (L.make_zernike(fx3, prep3, L.DEG_B26P),
                                   L.make_zernike(fx, prep, L.DEG_B26P)))):
            b3, b6 = mk()
            C3 = b3.coeffs()
            C3on6 = {g: C3[g][:, perm] for g in C3}
            Bx = []
            for q0 in range(fx["nq"]):
                train = [q for q in range(fx["nq"]) if q != q0]
                w = R.stencil_weights(fx["qfr"][train], fx["qfr"][q0], R7)
                x = R.gap_window_pairs(fx, q0)
                B_true = R.b_block(x, R.make_vq(fx, fx["ZG"][q0], q0))
                M = b6.model_M(C3on6, fx["qfr"][q0])
                Vp = np.tensordot(w, prep["V_SRc"][train], axes=(0, 0)) \
                    + L.lr_tile_from_M(fx, prep, M, fx["qfr"][q0])
                Bx.append(R.relF(R.b_block(x, Vp), B_true))
            print(f"    {name:<16s} transfer B med {np.median(Bx):.3e} "
                  f"max {np.max(Bx):.3e}")
            NPZ[f"transfer__{name}"] = np.array(Bx)

    np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
             "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
             f"study2_study1_{FIXNAME}_results.npz", **NPZ)
    print(f"\n[study1 {FIXNAME}] DONE in {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
