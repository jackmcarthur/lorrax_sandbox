"""study2_run_study2 — STUDY 2: audit every SVD/rank/Tikhonov cutoff in the
arbitrary-Q V_Q interpolation + trainer pipeline, and decide an eps/rank
policy that is safe from 640 to ~20k centroids.

The pipeline's cutoffs (REFERENCE_arbitrary_q_vq.py + lr_prep.py):
  EPS_TIK = 1e-4  cleaning:  g_eps(lam)=lam^2/(lam^2+(eps*lam_max)^2)
                             RELATIVE to lam_max (per q).  <-- the only cutoff
                             coupled to the C_q spectrum, which DEEPENS with n_mu.
  RIDGE   = 1e-11 fit LSQ:   A += RIDGE*(trace(A)/nb)*I, per K-space normal
                             block (nb<=15).  Relative to the block; nb is the
                             BASIS size, NOT n_mu -> n_mu-independent.
  pinv    = 1e-15 stencil:   f0 @ pinv(F), F = (nq_train x nR) trig matrix.
                             Pure q-geometry; NOT n_mu-dependent.
  EPS_LR  = 1e-8  LR ball:   K2max = 4 a^2 ln(1/eps_LR).  Physical Gaussian-tail
                             radius; depends on alpha+lattice, NOT n_mu.

Because both slab fixtures carry the SAME 640 centroids (cond_C ~1.6-1.8e7 in
both), n_mu cannot be varied physically here.  We probe the n_mu->20k regime by
SYNTHETIC SPECTRUM STRETCHING: lam' = lam_max (lam/lam_max)^sigma raises the
small-lam tail's depth (cond_C -> cond_C^sigma), eigenvectors fixed, mimicking
the deeper gapless tail of a few-hundred-atom / 20k-centroid C_q.  We then ask,
at FIXED eps_rel: does the physical B = M^H V_Q M drift?  (Prediction from the
junk-inertness thesis: little, because the deepened tail is inert under M^H..M
— unlike GW Sigma, ridge_ab 2026-07-17: junk-SENSITIVE 4-200 meV, eps must
scale relative, crossover eps* ~ cond(C)^-1/2.)

Sections:
  A  cutoff inventory + measured spectrum / crossover per fixture
  B  cleaning-eps sweep to completion (the killed stress axisA) + hard-cut
  C  fit-ridge sweep (the killed stress axisB) — n_mu-independence check
  D  SYNTHETIC STRETCH: B drift vs sigma at fixed-eps_rel / crossover-scaled /
     absolute-floor policies; Tikhonov vs hard rank-cut
Run: JID=<jid> ./proto1_run.sh python3 -u study2_run_study2.py [MoS2_6x6]
"""
import sys
import time
import numpy as np

import REFERENCE_arbitrary_q_vq as R

FIXNAME = sys.argv[1] if len(sys.argv) > 1 else "MoS2_6x6"
RY2MEV = R.RY2MEV
ALPHA = R.ALPHA
t00 = time.time()
NPZ = {}


# ---------------------------------------------------------------------------
# prepare the coarse-side bundle from a (possibly stretched) C_q spectrum,
# with either Tikhonov g_eps cleaning or a hard top-r projector.  Mirrors
# REFERENCE_arbitrary_q_vq.prepare_coarse exactly at sigma=1, eps_tik, tik.
# ---------------------------------------------------------------------------
def prepare(fx, C_q, alpha=ALPHA, eps_tik=R.EPS_TIK, sigma=1.0,
            mode="tik", rank_frac=None, stretch_mode="uniform",
            tail_keep=0.5):
    """stretch_mode: 'uniform' deepens ALL sub-maximal eigs lam'=lmax
    (lam/lmax)^sigma (pessimistic — also deepens resolved modes); 'tail'
    keeps the top tail_keep fraction fixed and deepens ONLY the lower tail
    (faithful to 'larger n_mu adds inert junk, physics unchanged')."""
    nq, n_mu = fx["nq"], fx["n_mu"]
    GS = R.lr_gset(fx, alpha)
    nG = GS.shape[1]
    S = np.empty((nq, n_mu, n_mu), dtype=complex)
    V_SRc = np.empty((nq, n_mu, n_mu), dtype=complex)
    Fch = np.empty((nq, n_mu, nG), dtype=complex)
    W = np.empty((nq, nG))
    ndamp, conds = [], []
    for q in range(nq):
        lam, Rv = np.linalg.eigh(0.5 * (C_q[q] + C_q[q].conj().T))
        lam = np.clip(lam, 1e-300, None)
        lmax = lam.max()
        if sigma != 1.0:
            if stretch_mode == "uniform":
                lam = lmax * (lam / lmax) ** sigma
            else:                                    # deepen only the tail
                kcut = int(round((1.0 - tail_keep) * n_mu))
                lam = lam.copy()
                lam[:kcut] = lmax * (lam[:kcut] / lmax) ** sigma
        conds.append(lmax / max(lam.min(), 1e-300))
        if mode == "tik":
            floor = eps_tik * lmax
            g = lam ** 2 / (lam ** 2 + floor ** 2)
        elif mode == "hard":
            r = int(round(rank_frac * n_mu))
            g = np.zeros_like(lam)
            g[-r:] = 1.0                              # keep top-r
        ndamp.append(int(np.sum(g < 0.5)))
        S[q] = (Rv * g[None, :]) @ Rv.conj().T
        Sc = np.conj(S[q])
        V_ref = R.make_vq(fx, fx["ZG"][q], q)
        V_LR = R.make_vq(fx, fx["ZG"][q], q, kind="slab_lr", alpha=alpha)
        V_SRc[q] = Sc @ (V_ref - V_LR) @ Sc
        zt = S[q] @ fx["ZG"][q]
        idx = R._sphere_slot(fx, q, GS)
        zt_ext = np.concatenate([zt, np.zeros((n_mu, 1), complex)], 1)
        qG = fx["qfr"][q][None, :] + GS.T.astype(float)
        ph = np.exp(2j * np.pi * (fx["rmu_frac"] @ qG.T))
        Fch[q] = ph * zt_ext[:, idx]
        W[q] = R.v_slab_on_set(fx, fx["qfr"][q], GS, kind="slab_lr",
                               alpha=alpha)
    prep = {"alpha": alpha, "eps_tik": eps_tik, "GS": GS, "S": S,
            "V_SRc": V_SRc, "Fch": Fch, "W": W,
            "gz_cols": {int(g): np.where(GS[2] == g)[0]
                        for g in np.unique(GS[2])}}
    return prep, float(np.mean(ndamp)), float(np.median(conds))


def loo_B(fx, prep, ridge=R.RIDGE, with_exc=False, hdir_cache=None):
    des = R.lr_design_blocks(fx, prep)
    R.RIDGE = ridge                                   # module-level knob
    R7 = R.stencil_r7(fx)
    Bs, Es = [], []
    for q0 in range(fx["nq"]):
        train = [q for q in range(fx["nq"]) if q != q0]
        w = R.stencil_weights(fx["qfr"][train], fx["qfr"][q0], R7)
        x = R.gap_window_pairs(fx, q0)
        B_true = R.b_block(x, R.make_vq(fx, fx["ZG"][q0], q0))
        SRi = np.tensordot(w, prep["V_SRc"][train], axes=(0, 0))
        C_loo = R.fit_lr_model(des, exclude=q0)
        Vp = SRi + R.lr_model_tile(fx, prep, des, C_loo, fx["qfr"][q0])
        Bp = R.b_block(x, Vp)
        Bs.append(R.relF(Bp, B_true))
        if with_exc:
            if q0 not in hdir_cache:
                D, Hdir = R.build_hdir(fx, q0)
                hdir_cache[q0] = (D, Hdir, R.exciton_evs(fx, D, Hdir, B_true))
            D, Hdir, ev_true = hdir_cache[q0]
            Es.append(float(np.max(np.abs(
                R.exciton_evs(fx, D, Hdir, Bp) - ev_true)) * RY2MEV))
    return np.array(Bs), (np.array(Es) if with_exc else None)


def main():
    print(f"[study2] fixture {FIXNAME} — cutoff audit + n_mu scaling")
    fx = R.load_fixture(FIXNAME)
    C_q = R.build_cq(fx)
    R.run_gates(fx, C_q)

    # ---- A. spectrum + crossover ----
    lams = np.array([np.linalg.eigvalsh(0.5 * (C_q[q] + C_q[q].conj().T))
                     for q in range(fx["nq"])])
    lmax = lams[:, -1]
    cond = lmax / np.clip(lams[:, 0], 1e-300, None)
    condmed = float(np.median(cond))
    print(f"\n  [A] cond_C med {condmed:.2e}  crossover eps*=cond^-1/2 "
          f"{1/np.sqrt(condmed):.2e}  (EPS_TIK={R.EPS_TIK:.0e})")
    print(f"      #eig<eps*lam_max at eps=1e-4: "
          f"{np.mean(np.sum(lams<1e-4*lmax[:,None],axis=1)):.0f}/{fx['n_mu']}")
    NPZ["cond"] = cond

    hc = {}
    # ---- B. cleaning-eps sweep (complete the killed axisA) + hard-cut ----
    print("\n  [B] cleaning sweep (Tikhonov g_eps, fit rebuilt per gauge)")
    print(f"    {'eps_rel':>9s} {'ndamp':>6s} {'B med':>10s} {'B max':>10s} "
          f"{'exc med':>8s} {'exc max':>8s}")
    for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        prep, nd, _ = prepare(fx, C_q, eps_tik=eps)
        Bs, Es = loo_B(fx, prep, with_exc=(eps in (1e-4, 1e-6)),
                       hdir_cache=hc)
        es = (f"{np.median(Es):>8.3f} {np.max(Es):>8.3f}" if Es is not None
              else "      --       --")
        print(f"    {eps:>9.0e} {nd:>6.0f} {np.median(Bs):>10.3e} "
              f"{np.max(Bs):>10.3e} {es}", flush=True)
        NPZ[f"B_eps_{eps:.0e}"] = Bs
    print("    hard rank-cut Pi=top-r (fraction of n_mu):")
    for rf in (0.75, 0.5, 0.375, 0.25):
        prep, nd, _ = prepare(fx, C_q, mode="hard", rank_frac=rf)
        Bs, _ = loo_B(fx, prep)
        print(f"    r={rf:>5.3f}n_mu ({int(rf*fx['n_mu'])})  ndamp {nd:>4.0f} "
              f"B med {np.median(Bs):>10.3e} max {np.max(Bs):>10.3e}",
              flush=True)
        NPZ[f"B_hard_{rf}"] = Bs

    # ---- C. fit-ridge sweep (complete the killed axisB) ----
    print("\n  [C] fit-ridge sweep (gauge eps=1e-4, b26p) — n_mu-independence")
    prep, _, _ = prepare(fx, C_q, eps_tik=1e-4)
    for ridge in (1e-6, 1e-8, 1e-11, 1e-14, 0.0):
        Bs, _ = loo_B(fx, prep, ridge=ridge)
        print(f"    RIDGE={ridge:>7.0e}  B med {np.median(Bs):>10.3e} "
              f"max {np.max(Bs):>10.3e}", flush=True)
        NPZ[f"B_ridge_{ridge:.0e}"] = Bs
    R.RIDGE = 1e-11

    # ---- D. synthetic spectrum stretch: B drift vs sigma ----
    # Two stretch models: 'uniform' (pessimistic, deepens resolved modes too)
    # and 'tail' (faithful: larger n_mu adds inert junk, physics fixed).
    # Two eps policies: fixed 1e-4 (current) vs crossover-tracking
    # eps=cond_eff^-1/2 (the ridge_ab scaling law), plus hard-cut for contrast.
    print("\n  [D] synthetic spectrum stretch (mimic larger n_mu); B drift "
          "vs sigma under fixed vs crossover-tracking eps")
    print(f"    {'stretch':>8s} {'sigma':>5s} {'cond_eff':>9s} {'policy':>11s}"
          f" {'eps_used':>9s} {'ndamp':>6s} {'B med':>10s} {'B max':>10s}")
    sigmas = (1.0, 1.25, 1.5, 1.75, 2.0)
    for smode in ("uniform", "tail"):
        for sigma in sigmas:
            # measure cond_eff from the actual stretched spectrum (fixed eps
            # prep), then run each policy at that cond
            _, _, cond_eff = prepare(fx, C_q, sigma=sigma, eps_tik=1e-4,
                                     stretch_mode=smode)
            eps_star = cond_eff ** -0.5
            for pol, eps in (("fixed_1e-4", 1e-4),
                             ("track_c^-.5", eps_star)):
                prep, nd, _ = prepare(fx, C_q, sigma=sigma, eps_tik=eps,
                                      stretch_mode=smode)
                Bs, _ = loo_B(fx, prep)
                print(f"    {smode:>8s} {sigma:>5.2f} {cond_eff:>9.1e} "
                      f"{pol:>11s} {eps:>9.1e} {nd:>6.0f} "
                      f"{np.median(Bs):>10.3e} {np.max(Bs):>10.3e}",
                      flush=True)
                NPZ[f"stretch_{smode}_{pol}_s{sigma}"] = Bs
            prep, nd, _ = prepare(fx, C_q, sigma=sigma, mode="hard",
                                  rank_frac=0.5, stretch_mode=smode)
            Bs, _ = loo_B(fx, prep)
            print(f"    {smode:>8s} {sigma:>5.2f} {cond_eff:>9.1e} "
                  f"{'hard_0.5':>11s} {'--':>9s} {nd:>6.0f} "
                  f"{np.median(Bs):>10.3e} {np.max(Bs):>10.3e}", flush=True)
            NPZ[f"stretch_{smode}_hard0.5_s{sigma}"] = Bs

    np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
             "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
             f"study2_study2_{FIXNAME}_results.npz", **NPZ)
    print(f"\n[study2 {FIXNAME}] DONE in {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
