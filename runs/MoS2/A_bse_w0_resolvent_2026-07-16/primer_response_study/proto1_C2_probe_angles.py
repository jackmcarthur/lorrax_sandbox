"""proto1_C2_probe_angles — discriminate REAL pair-subspace rotation from
convention/sewing artifacts in the collapsed principal-cosine spectra.

The first four-tails run (torus rows, plain-overlap band transport) measured
median principal cosine 0.13 on every edge — response 10B case 3 ("subspace
underresolved") — but the response also warns naive conventions fake failure.
Variants on two edges (one +b1, one +b2), all self-consistent:

  v1  torus rows, torus frames, t = polar(<u|u'>_G)          [what ran]
  v2  torus rows, torus frames, t = I                        [is t hurting?]
  v3  torus rows, NO band transport at all but rows matched by (k,n,m) with
      left orbital indexed at the SAME wrapped k-point kappa for both ends
      (i.e. compare X(qa) with X(qb) rows paired by kappa = k-qa == k'-qb,
      k' = k + b): removes the band-frame mismatch EXACTLY (same left
      orbital object), isolating the pure right-leg k -> k+b variation.
  v5  torus rows, band window 0..nbw (nbw=40,26): does the low-energy pair
      subspace transport, with the chaos confined to high bands?
  v6  PHYS (glued) rows + phys frames + G-shifted band overlaps — the
      response's sec-6 sewing done exactly.  If v6 ~ v1: rotation is real.
  v6w phys + window nbw=40.

Principal cosines: svals of M = S^-1 R^H H R' S'^-1 (whitened angles).
"""
import sys, time
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
                   "A_bse_w0_resolvent_2026-07-16/primer_response_study")
from proto1_prep import Fixture, polar

name = sys.argv[1] if len(sys.argv) > 1 else "MoS2_3x3"
fx = Fixture(name)
kg = fx.kgrid
N1, N2 = int(kg[0]), int(kg[1])


def gidx(i, j):
    return fx.k_lookup[(i % N1, j % N2, 0)]


def frames(C, rmax=None):
    lam, R = np.linalg.eigh(0.5 * (C + C.conj().T))
    lam = lam[::-1]
    R = R[:, ::-1]
    r = int(np.sum(lam > 0)) if rmax is None else min(rmax, int(np.sum(lam > 0)))
    return R[:, :r], np.sqrt(lam[:r])


def build_X_win(q, nbw, phys):
    """X with band window 0..nbw on BOTH legs; torus or phys rows."""
    out = np.empty((fx.nk, nbw, nbw, fx.n_mu), dtype=np.complex128)
    for k in range(fx.nk):
        kq, G0 = fx.kq_index(k, q)
        row = np.einsum("nsm,Msm->nMm", np.conj(fx.psi[kq][:nbw]), fx.psi[k][:nbw])
        if phys and np.any(G0):
            row = row * np.exp(-2j * np.pi * (fx.rmu_frac @ G0.astype(float)))[None, None, :]
        out[k] = row
    return out


def probe(qa, qb, mode, nbw):
    phys = mode.startswith("v6")
    Xa4 = build_X_win(qa, nbw, phys)
    Xb4 = build_X_win(qb, nbw, phys)
    Ca = np.conj(Xa4.reshape(-1, fx.n_mu).T) @ Xa4.reshape(-1, fx.n_mu)
    Cb = np.conj(Xb4.reshape(-1, fx.n_mu).T) @ Xb4.reshape(-1, fx.n_mu)
    Ra, Sa = frames(Ca)
    Rb, Sb = frames(Cb)
    r = min(len(Sa), len(Sb))
    Ra, Sa, Rb, Sb = Ra[:, :r], Sa[:r], Rb[:, :r], Sb[:r]
    H = np.zeros((fx.n_mu, fx.n_mu), dtype=np.complex128)
    if mode == "v3":
        # pair rows re-labeled by common kappa: row (kappa,n,m_at_k=kappa+qa)
        # vs (kappa,n,m_at_k'=kappa+qb): SAME left orbital, right leg steps.
        for k in range(fx.nk):
            ka, _ = fx.kq_index(k, qa)          # kappa
            # find k' with wrap(k'-qb) == ka:  k' = ka + qb
            kp = fx.k_lookup[tuple((fx.k_int[ka] + fx.k_int[qb]) % kg)]
            Xa = np.einsum("nsm,Msm->nMm", np.conj(fx.psi[ka][:nbw]),
                           fx.psi[k][:nbw]).reshape(-1, fx.n_mu)
            Xb = np.einsum("nsm,Msm->nMm", np.conj(fx.psi[ka][:nbw]),
                           fx.psi[kp][:nbw]).reshape(-1, fx.n_mu)
            H += np.conj(Xa.T) @ Xb
    else:
        for k in range(fx.nk):
            ka, G0a = fx.kq_index(k, qa)
            kb, G0b = fx.kq_index(k, qb)
            if mode == "v2":
                t = np.eye(nbw)
            elif phys:
                O = fx.band_overlap_G(ka, kb, Gshift=tuple(G0a - G0b), nbmax=nbw)
                t = polar(O)[0]
            else:
                t = polar(fx.band_overlap_G(ka, kb, nbmax=nbw))[0]
            Xa = Xa4[k].reshape(-1, fx.n_mu)
            rot = np.einsum("nN,NMm->nMm", t, Xb4[k]).reshape(-1, fx.n_mu)
            H += np.conj(Xa.T) @ rot
    M = (np.conj(Ra.T) @ H @ Rb) / Sa[:, None] / Sb[None, :]
    s = np.linalg.svd(M, compute_uv=False)
    return s, r


edges = [("b1", gidx(0, 0), gidx(1, 0)), ("b2", gidx(1, 0), gidx(1, 1))]
print(f"[{name}] principal-cosine probe; nb={fx.nb} nv={fx.nv}")
print(f"    {'edge':<4s} {'variant':<10s} {'r':>4s} {'p100':>7s} {'p90':>7s} "
      f"{'p50':>7s} {'p10':>7s} {'n>0.9':>6s} {'n>0.5':>6s}")
for ename, qa, qb in edges:
    for mode, nbw in [("v1", fx.nb), ("v2", fx.nb), ("v3", fx.nb),
                      ("v5", 40), ("v5", 26), ("v3", 26),
                      ("v6", fx.nb), ("v6w", 40), ("v6w", 26)]:
        t0 = time.time()
        s, r = probe(qa, qb, mode, nbw)
        lbl = f"{mode}_nb{nbw}"
        print(f"    {ename:<4s} {lbl:<10s} {r:>4d} {s.max():>7.4f} "
              f"{np.percentile(s,90):>7.4f} {np.median(s):>7.4f} "
              f"{np.percentile(s,10):>7.4f} {int(np.sum(s>0.9)):>6d} "
              f"{int(np.sum(s>0.5)):>6d}   ({time.time()-t0:.0f}s)", flush=True)
# sanity: self-edge must give all cosines 1
s, r = probe(gidx(1, 0), gidx(1, 0), "v1", fx.nb)
print(f"    sanity self-edge v1: min cosine {s.min():.6f} (must be ~1)")


# ---------------------------------------------------------------------------
# second pass: rank-truncated whitened angles — is there a transportable
# low-rank core (top-sigma directions only)?  Response sec 9 fixed-rank option.
# ---------------------------------------------------------------------------
def probe_rank(qa, qb, nbw, rtr):
    Xa4 = build_X_win(qa, nbw, False)
    Xb4 = build_X_win(qb, nbw, False)
    Ca = np.conj(Xa4.reshape(-1, fx.n_mu).T) @ Xa4.reshape(-1, fx.n_mu)
    Cb = np.conj(Xb4.reshape(-1, fx.n_mu).T) @ Xb4.reshape(-1, fx.n_mu)
    Ra, Sa = frames(Ca, rtr)
    Rb, Sb = frames(Cb, rtr)
    H = np.zeros((fx.n_mu, fx.n_mu), dtype=np.complex128)
    for k in range(fx.nk):
        ka, _ = fx.kq_index(k, qa)
        kb, _ = fx.kq_index(k, qb)
        t = polar(fx.band_overlap_G(ka, kb, nbmax=nbw))[0]
        Xa = Xa4[k].reshape(-1, fx.n_mu)
        rot = np.einsum("nN,NMm->nMm", t, Xb4[k]).reshape(-1, fx.n_mu)
        H += np.conj(Xa.T) @ rot
    M = (np.conj(Ra.T) @ H @ Rb) / Sa[:, None] / Sb[None, :]
    s = np.linalg.svd(M, compute_uv=False)
    return s


print(f"\n    rank-truncated whitened angles (torus rows, t=G-polar), edge b1:")
print(f"    {'window':<8s} {'r':>5s} {'p100':>7s} {'p90':>7s} {'p50':>7s} "
      f"{'n>0.9':>6s} {'n>0.5':>6s}")
for nbw in (fx.nb, 26):
    for rtr in (320, 160, 80, 40, 20, 10):
        s = probe_rank(gidx(0, 0), gidx(1, 0), nbw, rtr)
        print(f"    nb{nbw:<6d} {rtr:>5d} {s.max():>7.4f} "
              f"{np.percentile(s,90):>7.4f} {np.median(s):>7.4f} "
              f"{int(np.sum(s>0.9)):>6d} {int(np.sum(s>0.5)):>6d}", flush=True)
