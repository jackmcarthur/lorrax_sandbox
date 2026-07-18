"""REFERENCE_arbitrary_q_vq — the consolidated reference implementation of
the arbitrary-Q Coulomb-tile interpolation formalism.

This is the single self-contained distillation of the winning pipeline of
the 2026-07 primer-response campaign (arbitrary_q_bse.md secs 12-13,
F_SCHEME_NOTE.html, ARBITRARY_Q_PRIMER.md sec III.5): the F-scheme with the
b26p global long-range fit, Tikhonov gauge.  It supersedes the ~30 scratch
scripts in this directory as the runnable statement of the formalism (see
README.md, the scratch ledger).  It is a STUDY artifact, not production
code: production adoption into src/ is the owner's call, and the production
default remains the per-Q zeta refit until an off-grid ground truth exists.

--------------------------------------------------------------------------
THE PIPELINE (three stages; all per-element math in the stage docstrings)
--------------------------------------------------------------------------
Objects: the ISDF fit at each coarse q stores zeta~_mu(q+G) on a |q+G|^2
sphere (zeta_q.h5); the bare-exchange tile is the frame-free contraction
    V_q[mu,nu] = sum_G conj(zt_mu(q+G)) v(q+G) zt_nu(q+G).
Everything the BSE consumes is B = M^H V_Q M with M gap-window pair rows;
B (+ the exciton swap shift) is the verdict metric throughout.

  stage 1  prepare_coarse — offline, per coarse q_j, each in its own frame
    (a) Tikhonov-clean WITHOUT forming Z ("clean" = smooth-filtered
        rank regularization of the stored fit):
            eigh: C_q = R diag(lam) R^H
            g_eps(lam) = lam^2 / (lam^2 + (eps_tik * lam_max)^2)
            S_q = R g_eps(lam) R^H            (Hermitian, ~projector)
            zeta_c = S_q zeta_stored  =>  V_c = conj(S_q) V_ref conj(S_q)
        [analytic filter, NOT a hard cut: hard-cut projectors rotate
         freely inside C_q's gapless spectrum (Davis-Kahan); g_eps is the
         operator-theory-licensed variant — sec 12.3 — and the fit gauge
         in which M(K) is single-valued to ~1% — sec 13.1]
    (b) Gaussian SR/LR split of the kernel, per G-channel, K = q+G:
            v_LR(K) = v(K) exp(-K^2/4a^2),  v_SR = v * (-expm1(-K^2/4a^2))
            V_SRc(q_j) = V_c(q_j) - conj(S_q) V_LR(q_j) conj(S_q)
        The LR channel (everything divergent, everything winding) is
        confined to the FIXED global Miller superset gset(alpha) =
        {G : min_{q in BZ} |q+G|^2 <= 4 a^2 ln(1/eps_LR)}; the sphere tail
        beyond the stored cutoff is bounded by exp(-cutoff/4a^2).
    (c) Phase-factored LR form-factor samples on gset(alpha):
            F_mu(q_j;G) = e^{+2 pi i (q_j+G).s_mu} (S_q zeta~)_mu(q_j+G)
        — the exact form factor M_mu(K) of the cleaned fitting vector,
        sampled at K = q_j+G, centroid winding phase factored analytically.

  stage 2  fit_lr_model — GLOBAL, one weighted LSQ (the "b26p" fit)
    The samples F_mu(q;G) over ALL coarse q and all G in gset are scattered
    samples of one function M_mu(K) on the LR ball (single-valued to ~1% in
    this gauge — sec 13.1).  Since q_z = 0 on the slab coarse grid, K_z =
    G_z b_3 is an exact discrete channel; fit per-G_z in-plane polynomials
        M_mu(K_par, G_z) ~= sum_b c_b[mu, G_z] * (K_x/2a)^p (K_y/2a)^r
    with in-plane degrees {|G_z|=0: 3, 1: 2, 2: 0, 3: 0} (10+2*6+2*1+2*1
    = 26 complex coefficients per mu TOTAL — not per q), by ONE
    v_LR-weighted least squares over all samples:
        min_c  sum_{q,G} v_LR(q+G) | Phi(K_par) c - F(q;G) |^2 .
    The weight makes the objective exactly ||Delta A||_F^2 of the LR tile
    factor A = zt sqrt(v_LR): the fit minimizes what the physical
    contraction sees.  The design matrix depends on K only, so one normal
    solve per (G_z, LOO-exclusion) serves all n_mu rows; normal blocks are
    accumulated per coarse q so leave-one-out refits are honest and cheap.
    [Do not "improve" this with: literal real-space moments or moment
     pinning (refuted twice, secs 12.2/13.3), SVD/learned-multipole
     compression (no low rank to find, sec 13.2), multi-width GTO radial
     ladders (conditioning casualty, sec 13.2), hard-cut cleaning of the
     fitted channels (the q-fiber is the cut edge, sec 13.1).]

  stage 3  eval_vq — per target Q, cheap, NO solve / eigh / r_tot object
    (a) trigonometric stencil weights on a truncated lattice set R:
            w_j(Q) = f0 @ pinv(F),  F_ji = e^{-2 pi i q_j . R_i},
            f0_i = e^{-2 pi i Q . R_i}       (nR = 7 in-plane, sec 11)
    (b) V(Q) = sum_j w_j V_SRc(q_j)  +  V_LR_model(Q)
        with the closed-form LR rebuild (phase and kernel are analytic
        functions of Q, never interpolated):
            M(K) = poly model at K = Q+G,  G in gset(alpha)
            zt_mu = e^{-2 pi i (Q+G).s_mu} M_mu(Q+G)
            A = zt sqrt(v_LR(Q+G)),  V_LR_model = conj(A) A^T .

VALIDATED (MoS2 6x6, LOO over all 36 coarse q, Tik gauge, alpha=0.3,
nR7; arbitrary_q_bse.md sec 13.2/13.3, log lr_basis_ladder_6x6_tik.log):
    b26p:      B med 5.368e-3 / max 3.960e-2; excitons 0.043/0.144 meV
    F-anchor:  B med 5.848e-3 / max 3.779e-2; excitons 0.045/0.167 meV
    exact-LR ceiling C: 5.402e-3 — the model fit COSTS NOTHING vs exact
    LR channels at 1/467 the storage (26 global vs 337-per-q per mu).
    LOO coefficient stability 0.69% med / 1.74% max; grid transfer
    3x3-fit -> 6x6-deploy B med 5.382e-3 (zero downstream loss).
These are this file's acceptance targets (run mode `acceptance`).

SCOPE: slab systems, q_z = 0 coarse grids (per-G_z channels are exact
there).  3D-bulk needs a K_z-continuous basis — untested, sec 13.5(4).
Off-grid targets evaluate analytically but NO off-grid ground truth exists
yet (sec 11.4/12.6); quote no off-grid capability number.

PROVENANCE (single source of truth = this file; originals kept as
evidence, see README.md).  Copied-with-attribution from the validated
scratch scripts, arithmetic preserved bit-level where marked:
    fixture loader / recon / to_sphere / slab kernel / build_Cq /
    gap_window_pairs / truncR_weights      <- proto1_prep.py
    fix_sphere_wrap / gates / Hdir+exciton <- offgrid_prep.py
    gset / v_on_set / F-channels / rebuild <- tile_prep.py (TileStudy)
    basis / weighted normal blocks / LOO   <- lr_prep.py (ChannelFit)
    Tik gauge construction                 <- lr_basis_ladder.py / lr_transfer.py

USAGE (never on a login node; ./proto1_run.sh is the module-free
srun+shifter wrapper, PYTHONPATH already carries this directory):
    JID=<jid> ./proto1_run.sh python3 -u REFERENCE_arbitrary_q_vq.py acceptance
    JID=<jid> ./proto1_run.sh python3 -u REFERENCE_arbitrary_q_vq.py transfer
    JID=<jid> ./proto1_run.sh python3 -u test_reference_e2e.py   (smoke, 3x3)

READ-ONLY on all fixtures and on sources/lorrax_A.  No Z_r (n_mu x r_tot)
array is ever allocated at or after interpolation.
"""
import sys
import time

import h5py
import numpy as np

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox"
FIX = {   # (proto1_prep.py::FIX, verbatim)
    "MoS2_3x3": {
        "restart": f"{BASE}/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/isdf_tensors_640.h5",
        "zeta":    f"{BASE}/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/zeta_q.h5",
    },
    "MoS2_6x6": {
        "restart": f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax/tmp/isdf_tensors_640.h5",
        "zeta":    f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax/tmp/zeta_q.h5",
    },
}

# pipeline constants (sec 13.5 production shape)
ALPHA = 0.30          # Gaussian split width, 1/bohr; broad optimum ~1.5-2x dq
EPS_TIK = 1e-4        # relative Tikhonov filter width (fit gauge, sec 13.1)
EPS_LR = 1e-8         # Gaussian weight bound defining the LR G-superset
DEG_B26P = {0: 3, 1: 2, 2: 0, 3: 0}   # in-plane poly degree per |G_z|
RIDGE = 1e-11         # normal-equation ridge (lr_prep.ChannelFit.RIDGE)
RY2MEV = 13605.693

# sec-13 pinned acceptance targets (grep-verified, lr_basis_ladder_6x6_tik.log
# + lr_transfer.log; tolerances: reruns must reproduce to well under these)
PIN_6X6 = {
    "b26p":     {"B_med": 5.368e-3, "B_max": 3.960e-2,
                 "exc_med": 0.043, "exc_max": 0.144},
    "F_anchor": {"B_med": 5.848e-3, "B_max": 3.779e-2,
                 "exc_med": 0.045, "exc_max": 0.167},
    "coeff_stab": {"med": 6.899e-3, "max": 1.739e-2},
}
PIN_TRANSFER = {"B_med": 5.382e-3, "B_max": 3.908e-2}
TOL_B_REL = 0.02      # relative tolerance on B med/max vs pins
TOL_EXC_MEV = 0.005   # absolute tolerance on exciton med/max (meV)


def relF(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ===========================================================================
# fixture loading (proto1_prep.py::Fixture.__init__ + offgrid_prep.py::
# fix_sphere_wrap, condensed to what the pipeline needs; plain-dict bundle)
# ===========================================================================
def load_fixture(name):
    """Load one fixture into a plain-dict bundle `fx`.

    q-LABELING (the two wrap traps, KNOWN_SANDBOX_ERRORS 2026-07-17):
    zeta_q.h5 mf_header/kpoints/rk stores the UNWRAPPED QE list, while the
    stored zeta spheres are centered on the BGW-WRAPPED q (worth a measured
    155x on the physical interp ladder).  np.round is round-half-to-even,
    so components at exactly 1/2 need the sphere itself to pick the sign:
    take the candidate wrap minimizing max|q+G|^2 over the stored sphere
    (offgrid_prep.fix_sphere_wrap).  Every downstream phase/kernel uses
    these wrapped labels.
    """
    p = FIX[name]
    fx = {"name": name}
    with h5py.File(p["restart"], "r") as f:
        fx["psi"] = f["psi_full_y"][()]          # (nk, nb, ns, n_mu) u at centroids
        fx["kgrid"] = f["kgrid"][()].astype(int)
        fx["Vqmunu"] = f["V_qmunu"][()]          # disk tile (gate reference)
        fx["W0"] = f["W0_qmunu"][()]             # screened tile (exciton Hdir)
        fx["enk"] = f["enk_full"][()]            # (nk, nb) Ry
    with h5py.File(p["zeta"], "r") as f:
        fx["ZG"] = f["zeta_q_G"][()]             # (nq, n_mu, ngkmax) c128
        fx["gvec"] = f["isdf_header/gvec_components"][()].astype(np.int64)
        fx["ngk"] = f["isdf_header/ngk"][()].astype(int)
        fg = f["mf_header/gspace/FFTgrid"][()].astype(int)
        qraw = f["mf_header/kpoints/rk"][()]
        fx["adot"] = f["mf_header/crystal/adot"][()]
        blat = float(np.real(f["mf_header/crystal/blat"][()]))
        # BGW stores bvec in units of blat=2pi/alat; physical bohr^-1
        # (|bvec^T g|^2 in Ry) needs the blat factor (measured: 10.4%
        # makeVq-vs-disk residual without it)
        fx["bvec"] = f["mf_header/crystal/bvec"][()] * blat
        fx["celvol"] = float(np.real(f["mf_header/crystal/celvol"][()]))
        rmu_idx = f["isdf_header/centroids/r_mu_fft_idx"][()].astype(int)
        fx["zeta_cutoff"] = float(f["isdf_header/zeta_cutoff_ry"][()])
        ifmax = f["mf_header/kpoints/ifmax"][()]
    fx["nk"], fx["nb"], fx["ns"], fx["n_mu"] = fx["psi"].shape
    fx["nq"] = fx["ZG"].shape[0]
    fx["ngkmax"] = fx["ZG"].shape[2]
    fx["nx"], fx["ny"], fx["nz"] = [int(x) for x in fg]
    fx["n_rtot"] = fx["nx"] * fx["ny"] * fx["nz"]
    assert fx["nk"] == fx["nq"]
    fx["qfr_raw"] = qraw
    fx["qfr"] = qraw - np.round(qraw)            # BGW-wrapped (pre half-fix)
    kg = fx["kgrid"]
    fx["k_int"] = np.rint(qraw * kg[None, :]).astype(int) % kg[None, :]
    fx["k_lookup"] = {tuple(v): i for i, v in enumerate(fx["k_int"])}
    assert len(fx["k_lookup"]) == fx["nq"]
    rx = np.arange(fx["nx"]) / fx["nx"]
    ry = np.arange(fx["ny"]) / fx["ny"]
    rz = np.arange(fx["nz"]) / fx["nz"]
    RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing="ij")
    fx["rfrac"] = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
    dims = np.array([fx["nx"], fx["ny"], fx["nz"]])
    fx["rmu_frac"] = rmu_idx / dims[None, :]     # centroid frac coords s_mu
    fx["rmu_flat"] = ((rmu_idx[:, 0] * fx["ny"]) + rmu_idx[:, 1]) * fx["nz"] \
        + rmu_idx[:, 2]
    fx["nv"] = int(ifmax.ravel()[0])
    assert np.all(ifmax == fx["nv"])
    _fix_sphere_wrap(fx)
    return fx


def _fix_sphere_wrap(fx):
    """(offgrid_prep.py::fix_sphere_wrap) Half-boundary wrap disambiguation:
    per q, among the +-1/2 sign candidates keep the one whose sphere fits
    max|q+G|^2 <= cutoff.  No-op on grids without half components."""
    changed = 0
    for q in range(fx["nq"]):
        base = fx["qfr_raw"][q] - np.round(fx["qfr_raw"][q])
        cands = [[]]
        for c in range(3):
            opts = [0.5, -0.5] if abs(abs(base[c]) - 0.5) < 1e-9 else [base[c]]
            cands = [cc + [o] for cc in cands for o in opts]
        n = int(fx["ngk"][q])
        G = fx["gvec"][q][:, :n].astype(np.float64)
        best, bestm = None, None
        for cc in cands:
            qc = np.asarray(cc)
            K = fx["bvec"].T @ (qc[:, None] + G)
            m = float(np.max(np.sum(K * K, axis=0)))
            if bestm is None or m < bestm:
                best, bestm = qc, m
        assert bestm <= fx["zeta_cutoff"] + 1e-9, \
            f"q={q}: no candidate wrap fits the stored sphere"
        if np.max(np.abs(best - fx["qfr"][q])) > 1e-12:
            changed += 1
        fx["qfr"][q] = best
    if changed:
        print(f"  [wrapfix] {fx['name']}: {changed} of {fx['nq']} q relabeled "
              f"to the sphere-derived center")


# ===========================================================================
# grid / sphere / kernel primitives (proto1_prep.py::Fixture methods +
# tile_prep.py::TileStudy.v_on_set, arithmetic verbatim)
# ===========================================================================
def flat_idx(fx, gv):
    """(3, n) int Miller -> flat C-order FFT index."""
    return ((gv[0] % fx["nx"]) * fx["ny"] + gv[1] % fx["ny"]) * fx["nz"] \
        + gv[2] % fx["nz"]


def recon(fx, q):
    """zeta_q(mu, r) in the lab frame on the full FFT grid (gates only)."""
    ZGq = fx["ZG"][q]
    box = np.zeros((fx["n_mu"], fx["n_rtot"]), dtype=np.complex128)
    fi = flat_idx(fx, fx["gvec"][q])
    n = int(fx["ngk"][q])
    box[:, fi[:n]] = ZGq[:, :n]
    R = np.fft.ifftn(box.reshape(fx["n_mu"], fx["nx"], fx["ny"], fx["nz"]),
                     axes=(1, 2, 3), norm="backward"
                     ).reshape(fx["n_mu"], fx["n_rtot"])
    return R * np.exp(2j * np.pi * (fx["rfrac"] @ fx["qfr"][q]))[None, :]


def to_sphere(fx, zr, q):
    """rows(r) -> rows(G) on sphere(q) (gates only)."""
    ph = np.exp(-2j * np.pi * (fx["rfrac"] @ fx["qfr"][q]))
    box = np.fft.fftn((zr * ph[None, :]).reshape(-1, fx["nx"], fx["ny"],
                                                 fx["nz"]),
                      axes=(1, 2, 3), norm="backward"
                      ).reshape(zr.shape[0], fx["n_rtot"])
    fi = flat_idx(fx, fx["gvec"][q])
    n = int(fx["ngk"][q])
    out = np.zeros((zr.shape[0], fx["ngkmax"]), dtype=np.complex128)
    out[:, :n] = box[:, fi[:n]]
    return out


def v_slab_on_set(fx, qfrac, GS, kind="slab", alpha=None):
    """Slab-truncated Coulomb kernel on an explicit Miller set at momentum
    qfrac (wrapped fractional), per G-channel, K = q+G Cartesian (1/bohr):
        v(K) = 8 pi / K^2 * f2d / V_cell,
        f2d  = 1 - exp(-z_c |K_par|) cos(K_z z_c),   z_c = pi / b3_z
    Only the true divergence K^2 < 1e-12 is zeroed (the q=0 G=0 slot);
    at q != 0 the finite G=0 term is part of the body (measured: zeroing
    it moves makeVq-vs-disk from ~1e-9 to 0.33).  Split:
        slab_lr: v * exp(-K^2/4a^2)      slab_sr: v * (-expm1(-K^2/4a^2))
    (stable expm1: vSR+vLR == v to 1e-13, gated)."""
    K = fx["bvec"].T @ (np.asarray(qfrac)[:, None] + GS.astype(np.float64))
    K2 = np.sum(K * K, axis=0)
    zero = K2 < 1e-12
    K2s = np.where(zero, 1.0, K2)
    zc = np.pi / fx["bvec"][2, 2]
    f2d = 1.0 - np.exp(-zc * np.sqrt(K[0] ** 2 + K[1] ** 2)) \
        * np.cos(K[2] * zc)
    v = 8.0 * np.pi / K2s * f2d / fx["celvol"]
    if kind == "slab_lr":
        v = v * np.exp(-K2 / (4.0 * alpha ** 2))
    elif kind == "slab_sr":
        v = v * (-np.expm1(-K2 / (4.0 * alpha ** 2)))
    return np.where(zero, 0.0, v)


def v_sphere(fx, q, kind="slab", alpha=None):
    """Kernel on the stored sphere at coarse q. Returns (v, n_G)."""
    n = int(fx["ngk"][q])
    v = v_slab_on_set(fx, fx["qfr"][q], fx["gvec"][q][:, :n], kind, alpha)
    return v, n


def make_vq(fx, zt, q, kind="slab", alpha=None):
    """V[mu,nu] = sum_G conj(zt_mu(q+G)) v(q+G) zt_nu(q+G) on sphere(q)."""
    v, n = v_sphere(fx, q, kind, alpha)
    A = zt[:, :n] * np.sqrt(v)[None, :]
    return np.conj(A) @ A.T


def build_cq(fx):
    """C_q Gram rebuild from psi at centroids (proto1_prep.py::build_Cq,
    order-robust R-space route, arithmetic verbatim):
        C_q[mu,nu] = sum_{k,mn} conj(rho_kmn(r_mu)) rho_kmn(r_nu),
        rho_kmn(r) = sum_s conj(u_{m, wrap(k-q), s}(r)) u_{n, k, s}(r)
    with stored cell-periodic spinors at WRAPPED k labels (torus
    convention, no umklapp phases; gate: X^H X == C_q)."""
    psi = fx["psi"]
    nq, nb, ns, n_mu = fx["nq"], fx["nb"], fx["ns"], fx["n_mu"]
    kg = fx["kgrid"]
    psiX = np.conj(psi).transpose(0, 3, 1, 2)
    P = np.einsum("kmna,knbr->karmb", psiX, psi, optimize=True)
    Rall = np.array([[rx, ry, rz] for rx in range(kg[0])
                     for ry in range(kg[1]) for rz in range(kg[2])])
    Rw = ((Rall + kg // 2) % kg) - (kg // 2)
    EqR = np.exp(2j * np.pi * (fx["qfr"] @ Rw.T))
    P_R = (EqR.T @ P.reshape(nq, -1)).reshape(len(Rw), ns, n_mu, n_mu, ns)
    C_R = np.einsum("ravmb,ravmb->rvm", np.conj(P_R), P_R, optimize=True)
    C_q = np.transpose(((np.exp(-2j * np.pi * (fx["qfr"] @ Rw.T)) / nq)
                        @ C_R.reshape(len(Rw), -1)
                        ).reshape(nq, n_mu, n_mu), (0, 2, 1))
    return C_q


def kq_index(fx, ki, qi):
    """Index of wrap(k - q) on the stored grid."""
    d = fx["k_int"][ki] - fx["k_int"][qi]
    return fx["k_lookup"][tuple(d % fx["kgrid"])]


def gap_window_pairs(fx, q, nvw=3, ncw=3):
    """Spin-traced BSE exchange rows M_cvk(mu) = sum_s conj(u_{c,k-q,s})
    u_{v,k,s} at the centroids; top-nvw valence x bottom-ncw conduction x
    all k -> (npair, n_mu).  These contract the tile into the physical
    gap-window block B = M^H V M — the campaign verdict variable.
    [Caveat inherited from the campaign: 3v x 3c splits Kramers doublets;
    fine for tile-swap COMPARISONS (both sides share the window), not for
    sub-permille absolute claims — CAMPAIGN_REPORT sec 6.]"""
    nv = fx["nv"]
    cs = list(range(nv, nv + ncw))
    vs = list(range(nv - nvw, nv))
    rows = np.empty((fx["nk"], ncw, nvw, fx["n_mu"]), dtype=np.complex128)
    for k in range(fx["nk"]):
        kq = kq_index(fx, k, q)
        rows[k] = np.einsum("csm,vsm->cvm",
                            np.conj(fx["psi"][kq][cs]), fx["psi"][k][vs])
    return rows.reshape(-1, fx["n_mu"])


def b_block(x, V):
    """B[p,p'] = sum_{mu,nu} conj(x[p,mu]) V[mu,nu] x[p',nu]."""
    return np.conj(x) @ V @ x.T


# ===========================================================================
# gate battery (offgrid_prep.py::run_gates, condensed; every value printed;
# any FAIL stops the run — KNOWN_SANDBOX_ERRORS rule)
# ===========================================================================
def run_gates(fx, C_q):
    ok = True

    def log(k, v, tol=None):
        nonlocal ok
        flag = "" if tol is None else ("  OK" if v <= tol else "  ** FAIL **")
        if tol is not None and v > tol:
            ok = False
        print(f"    [gate] {k:<44s} {v:.3e}{flag}")

    print(f"  [gates] {fx['name']}:")
    n0 = int(fx["ngk"][0])
    zt = to_sphere(fx, recon(fx, 0), 0)
    log("recon_roundtrip_sphere_Gamma",
        relF(zt[:, :n0], fx["ZG"][0][:, :n0]), 1e-13)
    k2max = max(np.max(np.sum((fx["bvec"].T @ (fx["qfr"][q][:, None]
                               + fx["gvec"][q][:, :int(fx["ngk"][q])]
                               .astype(np.float64))) ** 2, axis=0))
                for q in range(fx["nq"]))
    log("sphere_max|q+G|^2_minus_cutoff", max(0.0, k2max - fx["zeta_cutoff"]),
        1e-9)
    vd = [relF(make_vq(fx, fx["ZG"][q], q), fx["Vqmunu"][q])
          for q in range(fx["nq"])]
    log("makeVq_vs_disk_Vqmunu_allq_max", float(np.max(vd)), 5e-6)
    # X^H X == C_q (torus convention) at q=0
    q = 0
    X = np.empty((fx["nk"], fx["nb"], fx["nb"], fx["n_mu"]),
                 dtype=np.complex128)
    for k in range(fx["nk"]):
        kq = kq_index(fx, k, q)
        X[k] = np.einsum("nsm,Msm->nMm", np.conj(fx["psi"][kq]), fx["psi"][k])
    X = X.reshape(-1, fx["n_mu"])
    log("XHX_vs_Cq_torus_q0", relF(np.conj(X.T) @ X, C_q[0]), 1e-8)
    for q in range(2):
        v, n = v_sphere(fx, q)
        vs, _ = v_sphere(fx, q, kind="slab_sr", alpha=0.63)
        vl, _ = v_sphere(fx, q, kind="slab_lr", alpha=0.63)
        log(f"vSR+vLR==v_q{q}",
            float(np.max(np.abs(vs[:n] + vl[:n] - v[:n]))
                  / max(np.max(np.abs(v[:n])), 1e-300)), 1e-13)
    # slab-axis separability (per-Gz channels need b3 || z, b1/b2 in-plane)
    bv = fx["bvec"]
    log("slab_axes_offdiag", float(max(np.max(np.abs(bv[2, :2])),
                                       np.max(np.abs(bv[:2, 2])))
                                   / np.abs(bv[2, 2])), 1e-12)
    assert ok, "gate battery FAILED — stop (KNOWN_SANDBOX_ERRORS rule)"


# ===========================================================================
# STAGE 1 — offline preparation at the coarse grid points
# ===========================================================================
def lr_gset(fx, alpha=ALPHA):
    """(tile_prep.py::TileStudy.gset) Fixed global Miller superset of the
    LR channel: all G with min_{q in BZ, q_z=0} |q+G|^2 <= 4 a^2 ln(1/eps),
    minimized over a 13x13 in-plane q sample.  337 G at alpha=0.3 (MoS2)."""
    K2max = 4.0 * alpha ** 2 * np.log(1.0 / EPS_LR)
    Kmax = np.sqrt(K2max)
    nmax = [int(np.ceil(Kmax / np.linalg.norm(fx["bvec"][i]))) + 1
            for i in range(3)]
    gr = [np.arange(-n, n + 1) for n in nmax]
    GX, GY, GZ = np.meshgrid(*gr, indexing="ij")
    Gall = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 0)
    ts = np.linspace(-0.5, 0.5, 13, endpoint=False)
    m = np.full(Gall.shape[1], np.inf)
    for tx in ts:
        for ty in ts:
            qf = np.array([tx, ty, 0.0])
            K = fx["bvec"].T @ (qf[:, None] + Gall.astype(np.float64))
            m = np.minimum(m, np.sum(K * K, axis=0))
    return np.ascontiguousarray(Gall[:, m <= K2max])


def _sphere_slot(fx, q, GS):
    """(tile_prep.py::TileStudy.sphere_slot) Miller columns of GS -> stored
    sphere slots at q; -1 where outside the sphere (channel is zero in the
    stored representation; Gaussian weight bounded by exp(-cutoff/4a^2))."""
    n = int(fx["ngk"][q])
    lut = {tuple(g): i for i, g in enumerate(fx["gvec"][q][:, :n].T)}
    return np.array([lut.get(tuple(g), -1) for g in GS.T])


def prepare_coarse(fx, C_q, alpha=ALPHA, eps_tik=EPS_TIK):
    """STAGE 1.  Returns the coarse-side bundle `prep`:
      S       (nq, nmu, nmu)  Tikhonov cleaning operators S_q (Hermitian)
      V_SRc   (nq, nmu, nmu)  cleaned short-range tiles (the stencil data)
      GS      (3, nG)         fixed LR Miller superset gset(alpha)
      Fch     (nq, nmu, nG)   phase-factored cleaned LR form-factor samples
      W       (nq, nG)        the LSQ weights v_LR(q+G) (head slot -> 0)
      gz_cols {gz: cols}      per-G_z column index into GS
    Per-element math in the module docstring, stage 1.  Storage: one n_mu^2
    tile per coarse q (V_SRc); Fch/W are fit inputs only — after stage 2
    they can be dropped (the b26p coefficients replace them)."""
    nq, n_mu = fx["nq"], fx["n_mu"]
    assert np.max(np.abs(fx["qfr"][:, 2])) < 1e-12, "slab pipeline needs q_z=0"
    GS = lr_gset(fx, alpha)
    nG = GS.shape[1]
    S = np.empty((nq, n_mu, n_mu), dtype=np.complex128)
    V_SRc = np.empty((nq, n_mu, n_mu), dtype=np.complex128)
    Fch = np.empty((nq, n_mu, nG), dtype=np.complex128)
    W = np.empty((nq, nG))
    n_out = 0
    for q in range(nq):
        # (a) Tikhonov cleaning operator from eigh(C_q) — a function of the
        # spectrum, invariant under degenerate-subspace rotation (stable)
        lam, R = np.linalg.eigh(0.5 * (C_q[q] + C_q[q].conj().T))
        g = lam ** 2 / (lam ** 2 + (eps_tik * lam.max()) ** 2)
        S[q] = (R * g[None, :]) @ R.conj().T
        Sc = np.conj(S[q])
        # (b) cleaned SR tile: conj(S) [V_ref - V_LR] conj(S) on the sphere
        V_ref = make_vq(fx, fx["ZG"][q], q)
        V_LR = make_vq(fx, fx["ZG"][q], q, kind="slab_lr", alpha=alpha)
        V_SRc[q] = Sc @ (V_ref - V_LR) @ Sc
        # (c) phase-factored cleaned form factors on the superset
        zt = S[q] @ fx["ZG"][q]
        idx = _sphere_slot(fx, q, GS)
        n_out += int(np.sum(idx < 0))
        zt_ext = np.concatenate([zt, np.zeros((n_mu, 1), np.complex128)], 1)
        qG = fx["qfr"][q][None, :] + GS.T.astype(np.float64)
        ph = np.exp(2j * np.pi * (fx["rmu_frac"] @ qG.T))
        Fch[q] = ph * zt_ext[:, idx]
        W[q] = v_slab_on_set(fx, fx["qfr"][q], GS, kind="slab_lr",
                             alpha=alpha)
    tail = float(np.exp(-fx["zeta_cutoff"] / (4.0 * alpha ** 2)))
    print(f"  [prep] {fx['name']}: gset({alpha}) = {nG} G; {n_out} "
          f"out-of-sphere (q,G) channels zero-filled; sphere-tail bound "
          f"{tail:.1e}")
    return {"alpha": alpha, "eps_tik": eps_tik, "GS": GS, "S": S,
            "V_SRc": V_SRc, "Fch": Fch, "W": W,
            "gz_cols": {int(g): np.where(GS[2] == g)[0]
                        for g in np.unique(GS[2])}}


# ===========================================================================
# STAGE 2 — the global b26p LR fit
# ===========================================================================
def _poly_spec(d):
    """[(a, b)] with a+b <= d, graded order (lr_prep.py::poly_pairs)."""
    return [(a, t - a) for t in range(d + 1) for a in range(t + 1)]


def _eval_basis(Kpar, spec, alpha):
    """Design matrix (n_samples, nb): (K_x/2a)^p (K_y/2a)^r, real
    (lr_prep.py::eval_basis, polynomial branch)."""
    s = 1.0 / (2.0 * alpha)
    x, y = Kpar[0] * s, Kpar[1] * s
    return np.stack([(x ** a) * (y ** b) if (a or b) else np.ones_like(x)
                     for a, b in spec], 1)


def lr_design_blocks(fx, prep, degrees=None):
    """Per-(G_z, q) weighted normal blocks of the LR fit (lr_prep.py::
    ChannelFit.__init__):
        AtA[gz][q] = Phi^T diag(w) Phi     (nb, nb)   real basis
        AtY[gz][q] = Phi^T diag(w) Y       (nb, n_mu)
    with Phi the in-plane design at the (q, G) samples of channel gz,
    w = v_LR(q+G), Y = Fch samples.  Channels with |gz| absent from
    `degrees` are model-zero (dropped).  Per-q blocks make LOO refits
    honest (target's samples excluded) at O(nb^2) cost."""
    if degrees is None:
        degrees = DEG_B26P
    nq = fx["nq"]
    des = {"specs": {}, "AtA": {}, "AtY": {}, "alpha": prep["alpha"]}
    for g, cols in prep["gz_cols"].items():
        if abs(g) not in degrees:
            continue
        spec = _poly_spec(degrees[abs(g)])
        nb = len(spec)
        assert nb <= 0.6 * len(cols) * nq, f"gz={g}: basis under-determined"
        AtA = np.empty((nq, nb, nb))
        AtY = np.empty((nq, nb, fx["n_mu"]), dtype=np.complex128)
        for q in range(nq):
            qG = fx["qfr"][q][:, None] + prep["GS"][:, cols].astype(np.float64)
            Kpar = (fx["bvec"].T @ qG)[:2]
            Phi = _eval_basis(Kpar, spec, prep["alpha"])
            w = prep["W"][q][cols]
            Pw = Phi * w[:, None]
            AtA[q] = Phi.T @ Pw
            AtY[q] = Pw.T @ prep["Fch"][q][:, cols].T
        des["specs"][g] = spec
        des["AtA"][g] = AtA
        des["AtY"][g] = AtY
    ncoef = sum(len(s) for s in des["specs"].values())
    print(f"  [fit] LR design: degrees {degrees} -> {ncoef} complex "
          f"coefficients per mu (global)")
    return des


def fit_lr_model(des, exclude=None):
    """STAGE 2 solve (lr_prep.py::ChannelFit.coeffs): one ridge-stabilized
    normal solve per G_z channel over all coarse q except `exclude`.
    Returns {gz: C (nb, n_mu)} — n_mu x 26 complex TOTAL for b26p."""
    nq = next(iter(des["AtA"].values())).shape[0]
    sel = [q for q in range(nq) if q != exclude]
    out = {}
    for g in des["specs"]:
        A = des["AtA"][g][sel].sum(0)
        Y = des["AtY"][g][sel].sum(0)
        A = A + RIDGE * (np.trace(A) / A.shape[0]) * np.eye(A.shape[0])
        out[g] = np.linalg.solve(A, Y)
    return out


# ===========================================================================
# STAGE 3 — evaluation at arbitrary target Q
# ===========================================================================
def stencil_r7(fx):
    """The campaign's in-plane truncated-R stencil: 7 shortest lattice
    vectors [i, j, 0] in the adot metric (offgrid_prep.py::sorted_stencil)."""
    Rall = np.array([[i, j, 0] for i in range(-2, 4) for j in range(-2, 4)])
    d = np.sqrt(np.einsum("ri,ij,rj->r", Rall, fx["adot"], Rall))
    return Rall[np.argsort(d)][:7]


def stencil_weights(q_train, q0, Rset):
    """(proto1_prep.py::truncR_weights) Trigonometric interpolation weights
    on the truncated-R Fourier set: w = f0 @ pinv(F), F_ji = e^{-2pi i
    q_j.R_i}.  Same weights for every matrix element; exact (delta) when q0
    is a training point and Rset resolves the grid."""
    F = np.exp(-2j * np.pi * (q_train @ Rset.T))
    f0 = np.exp(-2j * np.pi * (np.asarray(q0) @ Rset.T))
    return f0 @ np.linalg.pinv(F)


def lr_model_tile(fx, prep, des, coeffs, qfrac):
    """Closed-form LR tile at ANY qfrac from the fitted model
    (lr_prep.py::ChannelFit.model_F + tile_prep.py::TileStudy.V_from_F):
        M_mu(K)  = Phi(K_par) @ C[gz]           per exact G_z channel
        zt_mu    = e^{-2 pi i (q+G).s_mu} M_mu  (analytic winding phase)
        A        = zt sqrt(v_LR(q+G)),   V = conj(A) A^T .
    Channels absent from the model are zero (their v_LR weight is ~0)."""
    GS = prep["GS"]
    qf = np.asarray(qfrac, dtype=np.float64)
    M = np.zeros((fx["n_mu"], GS.shape[1]), dtype=np.complex128)
    Kall = fx["bvec"].T @ (qf[:, None] + GS.astype(np.float64))
    for g, spec in des["specs"].items():
        cols = prep["gz_cols"][g]
        Phi = _eval_basis(Kall[:2][:, cols], spec, prep["alpha"])
        M[:, cols] = (Phi @ coeffs[g]).T
    v = v_slab_on_set(fx, qf, GS, kind="slab_lr", alpha=prep["alpha"])
    qG = qf[None, :] + GS.T.astype(np.float64)
    zt = np.exp(-2j * np.pi * (fx["rmu_frac"] @ qG.T)) * M
    A = zt * np.sqrt(v)[None, :]
    return np.conj(A) @ A.T


def eval_vq(fx, prep, des, coeffs, qfrac, train=None, Rset=None):
    """STAGE 3: the assembled prediction at target qfrac,
        V(Q) = sum_j w_j(Q) V_SRc(q_j)  +  V_LR_model(Q).
    `train` = coarse indices entering the stencil (default: all);
    `Rset` = truncated lattice set (default: the campaign nR7).
    No solve, no eigh, no r_tot object — n_mu^2 AXPYs + the closed-form
    LR rebuild."""
    if train is None:
        train = list(range(fx["nq"]))
    if Rset is None:
        Rset = stencil_r7(fx)
    w = stencil_weights(fx["qfr"][train], qfrac, Rset)
    V_SR = np.tensordot(w, prep["V_SRc"][train], axes=(0, 0))
    return V_SR + lr_model_tile(fx, prep, des, coeffs, qfrac)


# ===========================================================================
# exciton swap metric (offgrid_prep.py::build_Hdir / exciton_evs, verbatim:
# TDA gap-window Hamiltonian with the direct term from the stored W0 tiles;
# only the exchange block B is swapped between truth and prediction)
# ===========================================================================
def build_hdir(fx, q0, nvw=3, ncw=3):
    kg = fx["kgrid"]
    cs = list(range(fx["nv"], fx["nv"] + ncw))
    vs = list(range(fx["nv"] - nvw, fx["nv"]))
    npair = fx["nk"] * ncw * nvw
    kqs = np.array([kq_index(fx, k, q0) for k in range(fx["nk"])])
    qkk = np.array([[fx["k_lookup"][tuple((fx["k_int"][k] - fx["k_int"][kp])
                                          % kg)]
                     for kp in range(fx["nk"])] for k in range(fx["nk"])])
    D = np.array([fx["enk"][kqs[k], c] - fx["enk"][k, v]
                  for k in range(fx["nk"]) for c in cs for v in vs])
    psic = np.ascontiguousarray(fx["psi"][:, cs])
    psiv = np.ascontiguousarray(fx["psi"][:, vs])
    psic_kq = psic[kqs]
    H = np.zeros((npair, npair), dtype=np.complex128)
    bs = ncw * nvw
    for k in range(fx["nk"]):
        Tc = np.einsum("csm,KCsm->KcCm", np.conj(psic_kq[k]), psic_kq,
                       optimize=True)
        Tv = np.einsum("vsm,KVsm->KvVm", psiv[k], np.conj(psiv),
                       optimize=True)
        Wg = fx["W0"][qkk[k]]
        blk = np.einsum("KcCm,Kmn,KvVn->KcvCV", Tc, Wg, Tv, optimize=True)
        H[k * bs:(k + 1) * bs] = blk.transpose(1, 2, 0, 3, 4).reshape(bs,
                                                                      npair)
    return D, H / fx["nk"]


def exciton_evs(fx, D, Hdir, B, nstate=4):
    H = np.diag(D).astype(np.complex128) - Hdir + B / fx["nk"]
    H = 0.5 * (H + np.conj(H.T))
    return np.linalg.eigvalsh(H)[:nstate]


# ===========================================================================
# nulls (must hold at machine level before any accuracy number is read)
# ===========================================================================
def run_nulls(fx, prep, des, coeffs):
    ok = True

    def log(k, v, tol):
        nonlocal ok
        flag = "  OK" if v <= tol else "  ** FAIL **"
        if v > tol:
            ok = False
        print(f"    [null] {k:<44s} {v:.3e}{flag}")

    # exact-stencil reproduction: with the FULL R lattice the trig weights
    # are a delta, so "interpolating" to a training point returns its data
    kg = fx["kgrid"]
    Rfull = np.array([[i - kg[0] // 2, j - kg[1] // 2, 0]
                      for i in range(kg[0]) for j in range(kg[1])])
    q0 = 1
    w = stencil_weights(fx["qfr"], fx["qfr"][q0], Rfull)
    log("exact_stencil_VSRc_train_point",
        relF(np.tensordot(w, prep["V_SRc"], axes=(0, 0)),
             prep["V_SRc"][q0]), 1e-9)
    log("exact_stencil_Fch_train_point",
        relF(np.tensordot(w, prep["Fch"], axes=(0, 0)), prep["Fch"][q0]),
        1e-9)
    # own F rebuild == cleaned LR tile (channel machinery consistency;
    # bounded by the out-of-sphere zero-fill, pinned 1.9e-9 on MoS2_6x6)
    rr = []
    for q in range(fx["nq"]):
        Sc = np.conj(prep["S"][q])
        VLRc = Sc @ make_vq(fx, fx["ZG"][q], q, kind="slab_lr",
                            alpha=prep["alpha"]) @ Sc
        qG = fx["qfr"][q][None, :] + prep["GS"].T.astype(np.float64)
        zt = np.exp(-2j * np.pi * (fx["rmu_frac"] @ qG.T)) * prep["Fch"][q]
        v = prep["W"][q]
        A = zt * np.sqrt(v)[None, :]
        rr.append(relF(np.conj(A) @ A.T, VLRc))
    log("F_own_rebuild_vs_cleaned_LR_tile_max", float(np.max(rr)), 1e-6)
    assert ok, "null battery FAILED — stop"


# ===========================================================================
# acceptance run — reproduce the sec-13 pinned numbers (MoS2 6x6 LOO)
# ===========================================================================
def loo_ladder(fx, prep, des, with_excitons=True, with_anchor=True):
    """LOO over all coarse q: per target, honest refit (target's samples
    excluded from the LSQ), nR7 SR stencil, B + exciton metrics vs the
    stored-fit truth V_ref(q0) = makeVq(zeta_stored).  Returns rows dict."""
    R7 = stencil_r7(fx)
    rows = {"b26p": [], "F_anchor": []}
    exc = {"b26p": [], "F_anchor": []}
    for q0 in range(fx["nq"]):
        t0 = time.time()
        train = [q for q in range(fx["nq"]) if q != q0]
        w = stencil_weights(fx["qfr"][train], fx["qfr"][q0], R7)
        x = gap_window_pairs(fx, q0)
        B_true = b_block(x, make_vq(fx, fx["ZG"][q0], q0))
        SRi = np.tensordot(w, prep["V_SRc"][train], axes=(0, 0))
        preds = {}
        C_loo = fit_lr_model(des, exclude=q0)
        preds["b26p"] = SRi + lr_model_tile(fx, prep, des, C_loo,
                                            fx["qfr"][q0])
        if with_anchor:
            # F-anchor: stencil the exact channels instead of the model
            # (the sec-12 F-scheme; continuity anchor for the ladder)
            Fi = np.tensordot(w, prep["Fch"][train], axes=(0, 0))
            v = v_slab_on_set(fx, fx["qfr"][q0], prep["GS"], kind="slab_lr",
                              alpha=prep["alpha"])
            qG = fx["qfr"][q0][None, :] + prep["GS"].T.astype(np.float64)
            zt = np.exp(-2j * np.pi * (fx["rmu_frac"] @ qG.T)) * Fi
            A = zt * np.sqrt(v)[None, :]
            preds["F_anchor"] = SRi + np.conj(A) @ A.T
        if with_excitons:
            D, Hdir = build_hdir(fx, q0)
            ev_true = exciton_evs(fx, D, Hdir, B_true)
        for lbl, Vp in preds.items():
            rows[lbl].append(relF(b_block(x, Vp), B_true))
            if with_excitons:
                ev_p = exciton_evs(fx, D, Hdir, b_block(x, Vp))
                exc[lbl].append(float(np.max(np.abs(ev_p - ev_true))
                                      * RY2MEV))
        print(f"    q0={q0}: B[b26p]={rows['b26p'][-1]:.3e}"
              + (f" exc={exc['b26p'][-1]:.3f}meV" if with_excitons else "")
              + f" ({time.time()-t0:.0f}s)", flush=True)
    return rows, exc


def _check(label, got, pin, tol, absolute=False):
    d = abs(got - pin) if absolute else abs(got - pin) / pin
    ok = d <= tol
    print(f"    [pin] {label:<28s} got {got:.4e}  pin {pin:.4e}  "
          f"{'OK' if ok else '** FAIL **'}")
    return ok


def run_acceptance():
    t0 = time.time()
    print("[acceptance] MoS2_6x6 — reproduce arbitrary_q_bse.md sec 13.2")
    fx = load_fixture("MoS2_6x6")
    C_q = build_cq(fx)
    run_gates(fx, C_q)
    prep = prepare_coarse(fx, C_q)
    des = lr_design_blocks(fx, prep)
    coeffs = fit_lr_model(des)
    run_nulls(fx, prep, des, coeffs)
    # LOO coefficient stability (global coefficients; only q-dependence is
    # which q the LOO withholds — pinned med 6.899e-3 / max 1.739e-2)
    cst = [max(relF(fit_lr_model(des, exclude=q0)[g], coeffs[g])
               for g in coeffs) for q0 in range(fx["nq"])]
    print(f"  [fit] b26p LOO-coefficient stability: med "
          f"{np.median(cst):.3e} max {np.max(cst):.3e}")
    print(f"  [loo] {fx['nq']} targets, nR7 stencil")
    rows, exc = loo_ladder(fx, prep, des)
    ok = True
    print("\n  ===== acceptance vs sec-13 pins (lr_basis_ladder_6x6_tik.log)"
          " =====")
    for lbl in ("b26p", "F_anchor"):
        p = PIN_6X6[lbl]
        ok &= _check(f"{lbl} B med", float(np.median(rows[lbl])), p["B_med"],
                     TOL_B_REL)
        ok &= _check(f"{lbl} B max", float(np.max(rows[lbl])), p["B_max"],
                     TOL_B_REL)
        ok &= _check(f"{lbl} exc med (meV)", float(np.median(exc[lbl])),
                     p["exc_med"], TOL_EXC_MEV, absolute=True)
        ok &= _check(f"{lbl} exc max (meV)", float(np.max(exc[lbl])),
                     p["exc_max"], TOL_EXC_MEV, absolute=True)
    ok &= _check("coeff stability med", float(np.median(cst)),
                 PIN_6X6["coeff_stab"]["med"], 0.05)
    ok &= _check("coeff stability max", float(np.max(cst)),
                 PIN_6X6["coeff_stab"]["max"], 0.05)
    print(f"\n[acceptance] {'PASS' if ok else 'FAIL'} "
          f"({time.time()-t0:.0f}s)")
    return ok


def run_transfer():
    """Grid transfer (sec 13.3): fit the b26p model on the 3x3 data alone,
    deploy on the 6x6 fixture (the fixtures share all 640 centroids; the
    two zeta fits are independent — representation transfer, not leakage).
    Pinned: 6x6 LOO B med 5.382e-3 / max 3.908e-2 (lr_transfer.log)."""
    t0 = time.time()
    print("[transfer] b26p 3x3-fit -> 6x6-deploy (sec 13.3)")
    out = {}
    for name in ("MoS2_3x3", "MoS2_6x6"):
        fx = load_fixture(name)
        C_q = build_cq(fx)
        run_gates(fx, C_q)
        prep = prepare_coarse(fx, C_q)
        des = lr_design_blocks(fx, prep)
        out[name] = (fx, prep, des, fit_lr_model(des))
    fx3, _, _, C3 = out["MoS2_3x3"]
    fx6, prep6, des6, _ = out["MoS2_6x6"]
    # centroid alignment (order may differ; the SET is identical)
    p3 = np.argsort(fx3["rmu_flat"])
    p6 = np.argsort(fx6["rmu_flat"])
    assert np.array_equal(fx3["rmu_flat"][p3], fx6["rmu_flat"][p6])
    perm = np.empty(fx6["n_mu"], dtype=int)
    perm[p6] = p3
    C3on6 = {g: C3[g][:, perm] for g in C3}
    R7 = stencil_r7(fx6)
    Bx = []
    for q0 in range(fx6["nq"]):
        train = [q for q in range(fx6["nq"]) if q != q0]
        w = stencil_weights(fx6["qfr"][train], fx6["qfr"][q0], R7)
        x = gap_window_pairs(fx6, q0)
        B_true = b_block(x, make_vq(fx6, fx6["ZG"][q0], q0))
        Vp = np.tensordot(w, prep6["V_SRc"][train], axes=(0, 0)) \
            + lr_model_tile(fx6, prep6, des6, C3on6, fx6["qfr"][q0])
        Bx.append(relF(b_block(x, Vp), B_true))
    ok = _check("transfer B med", float(np.median(Bx)), PIN_TRANSFER["B_med"],
                TOL_B_REL)
    ok &= _check("transfer B max", float(np.max(Bx)), PIN_TRANSFER["B_max"],
                 TOL_B_REL)
    print(f"\n[transfer] {'PASS' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "acceptance"
    passed = {"acceptance": run_acceptance,
              "transfer": run_transfer}[mode]()
    sys.exit(0 if passed else 1)
