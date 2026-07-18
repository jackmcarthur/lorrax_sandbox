"""offgrid_prep — shared machinery for the DECISIVE off-grid-with-truth test of
the campaign's surviving candidate (CAMPAIGN_REPORT.md sec 5a items 1-2):
plain rank-cut ingredient interpolation of C_q / Z_q in the production
BGW-wrapped q-labeling, solved in the target's own frame, judged on the
PHYSICAL metric (gap-window B = M^H V_Q M + TDA exciton swap shift).

Reuses the campaign's null-tested harness: proto1_prep.Fixture (wrapped-q
loaders, recon/to_sphere, slab Coulomb, build_Cq, gap_window_pairs,
truncR_weights) — conventions documented there. Adds:

1. fix_sphere_wrap(fx): the HALF-INTEGER extension of the rk unwrap trap
   (KNOWN_SANDBOX_ERRORS 2026-07-17 item 1). proto1_prep wraps
   q = rk - round(rk); np.round is round-half-to-even, so q-components at
   exactly 1/2 stay at +1/2 — but the stored zeta spheres at those q may be
   centered on -1/2 (the 6x6 fourtails log shows the sphere gate FAILING at
   +14.45 Ry on MoS2_6x6, which has 11 half-boundary q's). Fix: derive the
   sphere center per q from the sphere itself — the unique candidate wrap
   with max|q_c+G|^2 <= cutoff over the stored G's. On 3x3 fixtures this is
   a no-op (no half components).

2. SiOldFixture: loader for the Si 4x4x4 FULL-BZ fixture work_old/tmp
   (n_mu=960, 64 q, stored zeta at ALL 64 q). NOTE: the campaign report
   sec 5a item 2 points at work_sym/tmp/isdf_tensors_792.h5, but that
   fixture (and work_demo) has IBZ-ONLY zeta (8 spheres) — unusable for
   per-q-truth scoring; work_old is the only full-BZ-zeta Si restart on
   disk. Its zeta_q.h5 mf_header/kpoints/rk is the IBZ list (8 rows) while
   zeta_q_G/ngk/gvec_components are full-BZ (64): the full q list is the
   row-major kgrid enumeration (last index fastest), verified by the gates
   (per-q sphere-center existence + makeVq-vs-disk at all 64 q + XHX).

3. Condensed gate battery (inherited from the campaign: recon roundtrip,
   sphere gate == 0, production tile disk-match, XHX==C_q, vSR+vLR) plus
   the solve-chain NULL (true C / true Z through SVD solve + to_sphere +
   B-metric: must be machine-level; C3 logged 6.6e-13) and the trig-interp
   exactness null (interpolating TO a training point reproduces C exactly).

4. Ladder solver (same arithmetic as proto1_C2_loo.solve_zeta, SVD factored
   out so all rungs share one decomposition) and a BATCHED per-k rebuild of
   the proto1_C2_loo exciton machinery (identical H_dir arithmetic,
   numpy-loop overhead moved into einsum batches).

READ-ONLY on all fixtures and on sources/lorrax_A. psi_full_y band-span trap
(KNOWN_SANDBOX_ERRORS 2026-07-17 item 2) respected: NO WFN-content truth is
built anywhere here — truth is the stored per-q full-rank fit, closed over
the restart's own psi_full_y/zeta_q_G.
"""
import numpy as np
import h5py

import proto1_prep
from proto1_prep import Fixture, relF, truncR_weights  # noqa: F401

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox"
SI_OLD = {
    "restart": f"{BASE}/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/"
               f"work_old/tmp/isdf_tensors_960.h5",
    "zeta":    f"{BASE}/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/"
               f"work_old/tmp/zeta_q.h5",
    "wfn":     f"{BASE}/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/"
               f"work_old/WFN.h5",   # IBZ (nrk=8) — WFN-side checks skipped
}
RY2MEV = 13605.693


class SiOldFixture(Fixture):
    """Si 4x4x4 work_old loader. Duplicates proto1_prep.Fixture.__init__ with
    ONE change (study-script duplication only): the full-BZ q list is the
    row-major kgrid enumeration, not mf_header/kpoints/rk (IBZ, 8 rows)."""

    def __init__(self, name="Si_4x4x4_old"):
        p = SI_OLD
        self.name = name
        with h5py.File(p["restart"], "r") as f:
            self.psi = f["psi_full_y"][()]
            self.kgrid = f["kgrid"][()].astype(int)
            self.Vqmunu = f["V_qmunu"][()]
            self.W0 = f["W0_qmunu"][()]
            self.enk = f["enk_full"][()]
            self.vhead = float(np.real(f["vhead"][()])) if "vhead" in f else None
            self.g0vec = f["G0_mu_nu"][()] if "G0_mu_nu" in f else None
        with h5py.File(p["zeta"], "r") as f:
            self.ZG = f["zeta_q_G"][()]
            self.gvec = f["isdf_header/gvec_components"][()].astype(np.int64)
            self.ngk = f["isdf_header/ngk"][()].astype(int)
            self.fg = f["mf_header/gspace/FFTgrid"][()].astype(int)
            self.bdot = f["mf_header/crystal/bdot"][()]
            self.adot = f["mf_header/crystal/adot"][()]
            self.blat = float(np.real(f["mf_header/crystal/blat"][()]))
            self.bvec = f["mf_header/crystal/bvec"][()] * self.blat
            self.celvol = float(np.real(f["mf_header/crystal/celvol"][()]))
            self.r_mu_fft_idx = f["isdf_header/centroids/r_mu_fft_idx"][()].astype(int)
            self.zeta_cutoff = float(f["isdf_header/zeta_cutoff_ry"][()])
            self.ifmax = f["mf_header/kpoints/ifmax"][()]
        kg = self.kgrid
        # row-major full-BZ enumeration, last component fastest (the LORRAX
        # full-list order; MoS2 fixtures store exactly this in rk). Verified
        # downstream: per-q sphere-center gate + makeVq-vs-disk at all q.
        qraw = np.array([[i / kg[0], j / kg[1], l / kg[2]]
                         for i in range(kg[0]) for j in range(kg[1])
                         for l in range(kg[2])])
        self.qfr_raw = qraw
        self.qfr = qraw - np.round(qraw)
        self.wfn_path = p["wfn"]
        self.nk, self.nb, self.ns, self.n_mu = self.psi.shape
        self.nq = self.ZG.shape[0]
        self.ngkmax = self.ZG.shape[2]
        self.nx, self.ny, self.nz = [int(x) for x in self.fg]
        self.n_rtot = self.nx * self.ny * self.nz
        assert self.nk == self.nq
        self.k_int = np.rint(self.qfr_raw * kg[None, :]).astype(int) % kg[None, :]
        self.k_lookup = {tuple(v): i for i, v in enumerate(self.k_int)}
        assert len(self.k_lookup) == self.nq
        rx = np.arange(self.nx) / self.nx
        ry = np.arange(self.ny) / self.ny
        rz = np.arange(self.nz) / self.nz
        RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing="ij")
        self.rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
        self.rmu_frac = self.r_mu_fft_idx / np.array([self.nx, self.ny, self.nz])[None, :]
        self.rmu_flat = ((self.r_mu_fft_idx[:, 0] * self.ny) + self.r_mu_fft_idx[:, 1]) \
            * self.nz + self.r_mu_fft_idx[:, 2]
        self.nv = int(self.ifmax.ravel()[0])
        assert np.all(self.ifmax == self.nv)
        self._wfn_cache = None
        self.vkind = "bare3d"     # Si bulk: production kernel is bare 3D
        self._vhead_tab = None

    def vq(self, q, kind="slab", alpha=None, taper=None):
        """Production 3D kernel: bare 8pi/K^2 PLUS the mini-BZ MC-averaged
        head at G=0 for q != 0 (compute_v_q_per_G v_head_miniBZ path,
        v_q_g_flat.py:526 — mc_average_vcoul_body=True default for
        sys_dim=3). Without it makeVq-vs-disk fails at med 9.6e-3 (the
        head is the largest v); with it the gate is machine-level. The
        bare_coulomb_cutoff (= ecutwfc = 25 Ry) is a no-op on the zeta
        sphere (max|q+G|^2 == cutoff, kept by the strict > test)."""
        v, n = super().vq(q, kind=kind, alpha=alpha, taper=taper)
        if kind == "bare3d":
            if self._vhead_tab is None:
                from gw.compute_vcoul import build_v_head_miniBZ_avg_3d
                self._vhead_tab = build_v_head_miniBZ_avg_3d(
                    tuple(int(x) for x in self.kgrid), self.bvec,
                    self.celvol)
            kg = self.kgrid
            qi = tuple(int(np.round(self.qfr[q][c] * kg[c])) % int(kg[c])
                       for c in range(3))
            hv = float(self._vhead_tab[qi])
            if hv != 0.0:
                g0 = np.all(self.gvec[q][:, :n] == 0, axis=0)
                v = v.copy()
                v[:n][g0] = hv
        return v, n


# ---------------------------------------------------------------------------
# sphere-derived wrap (half-integer extension of the rk unwrap trap)
# ---------------------------------------------------------------------------
def fix_sphere_wrap(fx, verbose=True):
    changed = []
    for q in range(fx.nq):
        base = fx.qfr_raw[q] - np.round(fx.qfr_raw[q])
        cands = [[]]
        for c in range(3):
            opts = [0.5, -0.5] if abs(abs(base[c]) - 0.5) < 1e-9 else [base[c]]
            cands = [cc + [o] for cc in cands for o in opts]
        n = int(fx.ngk[q])
        G = fx.gvec[q][:, :n].astype(np.float64)
        best, bestm = None, None
        for cc in cands:
            qc = np.asarray(cc)
            K = fx.bvec.T @ (qc[:, None] + G)
            m = float(np.max(np.sum(K * K, axis=0)))
            if bestm is None or m < bestm:
                best, bestm = qc, m
        if np.max(np.abs(best - fx.qfr[q])) > 1e-12:
            changed.append((q, tuple(fx.qfr[q]), tuple(best),
                            bestm - fx.zeta_cutoff))
        assert bestm <= fx.zeta_cutoff + 1e-9, \
            f"q={q}: no candidate wrap fits the stored sphere " \
            f"(best max|q+G|^2 = {bestm:.6f} vs cutoff {fx.zeta_cutoff})"
        fx.qfr[q] = best
    if verbose:
        print(f"  [wrapfix] {fx.name}: {len(changed)} of {fx.nq} q relabeled "
              f"to the sphere-derived center (half-boundary trap)")
        for q, old, new, m in changed[:6]:
            print(f"    q={q}: {np.round(old,4)} -> {np.round(new,4)}")
        if len(changed) > 6:
            print(f"    ... (+{len(changed)-6} more)")
    return changed


# ---------------------------------------------------------------------------
# condensed gate battery (campaign-inherited) — every value printed
# ---------------------------------------------------------------------------
def vkind(fx):
    """Production Coulomb kernel for this fixture: slab (2D, MoS2 default)
    or bare3d (Si bulk). Gate-verified via makeVq-vs-disk at all q."""
    return getattr(fx, "vkind", "slab")


def run_gates(fx, C_q, xhx_q=(0,), wfn_check=True):
    print(f"  [gates] {fx.name}: (Coulomb kind = {vkind(fx)})")
    ok = True

    def log(k, v, tol=None):
        nonlocal ok
        flag = "" if tol is None else ("  OK" if v <= tol else "  ** FAIL **")
        if tol is not None and v > tol:
            ok = False
        print(f"    [gate] {k:<44s} {v:.3e}{flag}")

    n0 = int(fx.ngk[0])
    zt = fx.to_sphere(fx.recon(0), 0)
    log("recon_roundtrip_sphere_Gamma", relF(zt[:, :n0], fx.ZG[0][:, :n0]), 1e-13)
    k2max = max(fx.Kvecs(q)[1].max() for q in range(fx.nq))
    log("sphere_max|q+G|^2_minus_cutoff (post-wrapfix)",
        max(0.0, k2max - fx.zeta_cutoff), 1e-9)
    vd = [relF(fx.make_Vq(fx.ZG[q], q, kind=vkind(fx)), fx.Vqmunu[q])
          for q in range(fx.nq)]
    log("makeVq_vs_disk_Vqmunu_allq_med", float(np.median(vd)))
    log("makeVq_vs_disk_Vqmunu_allq_max", float(np.max(vd)), 5e-6)
    for q in xhx_q:
        X = fx.build_X(q)
        log(f"XHX_vs_Cq_torus_q{q}", relF(np.conj(X.T) @ X, C_q[q]), 1e-8)
    if wfn_check:
        ug = fx.u_grid(0, nbmax=fx.nb)[:, :, fx.rmu_flat]
        sc = np.vdot(ug, fx.psi[0]) / np.vdot(ug, ug)
        log("wfn_vs_psi_full_y_scale_resid_k0",
            np.linalg.norm(sc * ug - fx.psi[0]) / np.linalg.norm(fx.psi[0]), 1e-8)
    if vkind(fx) == "slab":
        for q in range(min(2, fx.nq)):
            v, n = fx.vq(q)
            vs, _ = fx.vq(q, kind="slab_sr", alpha=0.63)
            vl, _ = fx.vq(q, kind="slab_lr", alpha=0.63)
            log(f"vSR+vLR==v_q{q}", float(np.max(np.abs(vs[:n] + vl[:n] - v[:n]))
                                          / max(np.max(np.abs(v[:n])), 1e-300)), 1e-13)
    assert ok, "gate battery FAILED — stop (KNOWN_SANDBOX_ERRORS rule)"
    return vd


# ---------------------------------------------------------------------------
# ladder solver (proto1_C2_loo.solve_zeta arithmetic, SVD factored out)
# ---------------------------------------------------------------------------
RUNGS = [("raw", 0.0), ("tikhonov", 1e-6), ("rankcut", 1e-2), ("rankcut", 1e-3),
         ("rankcut", 1e-4), ("rankcut", 1e-5)]


def rung_label(mode, lam):
    return mode if mode == "raw" else f"{mode}_{lam:.0e}"


def svd_herm(C):
    Ch = 0.5 * (C + C.conj().T)
    U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    return U, s, Vh


def rung_sinv(s, mode, lam):
    if mode == "raw":
        return 1.0 / s
    if mode == "tikhonov":
        return 1.0 / (s + lam * (s.sum() / len(s)))
    return np.where(s > lam * s[0], 1.0 / np.where(s > lam * s[0], s, 1), 0.0)


def apply_rung(U, s, Vh, Z, mode, lam):
    return (Vh.conj().T * rung_sinv(s, mode, lam)) @ (U.conj().T @ Z)


# ---------------------------------------------------------------------------
# physical metric helpers
# ---------------------------------------------------------------------------
def B_from_MG(fx, Mg, q0, **kw):
    kw.setdefault("kind", vkind(fx))
    v, n = fx.vq(q0, **kw)
    A = Mg[:, :n] * np.sqrt(v[:n])[None, :]
    return np.conj(A) @ A.T


def make_tile(fx, zt, q0):
    return fx.make_Vq(zt, q0, kind=vkind(fx))


def top_decile_rel(Bp, Bt):
    at = np.abs(Bt).ravel()
    thr = np.quantile(at, 0.9)
    m = at >= thr
    return np.median(np.abs((Bp - Bt).ravel()[m]) / at[m])


# ---------------------------------------------------------------------------
# TDA exciton swap machinery — proto1_C2_loo arithmetic, batched per k
# ---------------------------------------------------------------------------
def build_Hdir(fx, q0, nvw=3, ncw=3):
    kg = fx.kgrid
    cs = list(range(fx.nv, fx.nv + ncw))
    vs = list(range(fx.nv - nvw, fx.nv))
    npair = fx.nk * ncw * nvw
    kqs = np.array([fx.kq_index(k, q0)[0] for k in range(fx.nk)])
    qkk = np.array([[fx.k_lookup[tuple((fx.k_int[k] - fx.k_int[kp]) % kg)]
                     for kp in range(fx.nk)] for k in range(fx.nk)])
    D = np.array([fx.enk[kqs[k], c] - fx.enk[k, v]
                  for k in range(fx.nk) for c in cs for v in vs])
    psic = np.ascontiguousarray(fx.psi[:, cs])
    psiv = np.ascontiguousarray(fx.psi[:, vs])
    psic_kq = psic[kqs]                       # (nk, ncw, ns, nmu), indexed by k
    H = np.zeros((npair, npair), dtype=np.complex128)
    bs = ncw * nvw
    for k in range(fx.nk):
        Tc = np.einsum("csm,KCsm->KcCm", np.conj(psic_kq[k]), psic_kq,
                       optimize=True)
        Tv = np.einsum("vsm,KVsm->KvVm", psiv[k], np.conj(psiv), optimize=True)
        Wg = fx.W0[qkk[k]]
        blk = np.einsum("KcCm,Kmn,KvVn->KcvCV", Tc, Wg, Tv, optimize=True)
        H[k * bs:(k + 1) * bs] = blk.transpose(1, 2, 0, 3, 4).reshape(bs, npair)
    return D, H / fx.nk


def exciton_evs(fx, D, Hdir, B, nstate=4):
    H = np.diag(D).astype(np.complex128) - Hdir + B / fx.nk
    H = 0.5 * (H + np.conj(H.T))
    return np.linalg.eigvalsh(H)[:nstate]


# ---------------------------------------------------------------------------
# null tests (per fixture, per target): solve-chain + trig-exactness
# ---------------------------------------------------------------------------
def null_solve_chain(fx, C_q, q0):
    """TRUE C / TRUE Z through SVD solve + to_sphere + B metric. raw rung is
    the machine-level null (C3: 6.6e-13); rankcut 1e-4 is the truncation
    floor on TRUE data (junk-inertness row)."""
    ztrue_r = fx.recon(q0)
    Ztrue = C_q[q0] @ ztrue_r
    x = fx.gap_window_pairs(q0, 3, 3)
    B_true = B_from_MG(fx, x @ fx.ZG[q0], q0)
    U, s, Vh = svd_herm(C_q[q0])
    out = {}
    for mode, lam in (("raw", 0.0), ("rankcut", 1e-4)):
        zt0 = fx.to_sphere(apply_rung(U, s, Vh, Ztrue, mode, lam), q0)
        out[rung_label(mode, lam)] = relF(B_from_MG(fx, x @ zt0, q0), B_true)
    out["cond_C"] = float(s[0] / s[-1])
    return out


def null_trig_exact(fx, C_q, train, Rset):
    """Interpolating TO a training point with the exact stencil must
    reproduce C_q there to machine precision (weights -> delta)."""
    q0 = train[0]
    w = truncR_weights(fx.qfr[train], fx.qfr[q0], Rset)
    C0 = np.tensordot(w, C_q[train], axes=(0, 0))
    return relF(C0, C_q[q0])


# ---------------------------------------------------------------------------
# shared driver machinery (used by offgrid_mos2.py and offgrid_si.py)
# ---------------------------------------------------------------------------
def sorted_stencil(fx, Rall):
    Rall = np.asarray(Rall)
    d = np.sqrt(np.einsum("ri,ij,rj->r", Rall, fx.adot, Rall))
    return Rall[np.argsort(d)]


def ladder_at_target(fx, C_q, Zr_all, train, q0, stencils, res, tag,
                     exciton_stencils=(), Ztrue_r=None):
    """One target: interp C/Z from `train` rows of Zr_all (nq, n_mu*n_rtot),
    solve ladder, physical metrics vs stored-fit truth. The rung solve is
    applied on the target sphere (solve/to_sphere commute; gated)."""
    x = fx.gap_window_pairs(q0, 3, 3)
    B_true = B_from_MG(fx, x @ fx.ZG[q0], q0)
    V_true = make_tile(fx, fx.ZG[q0], q0)
    exc_ready = False
    for sname, Rset in stencils:
        w = truncR_weights(fx.qfr[train], fx.qfr[q0], Rset)
        C0 = np.tensordot(w, C_q[train], axes=(0, 0))
        Z0_r = np.zeros(fx.n_mu * fx.n_rtot, dtype=np.complex128)
        for wi, t in zip(w, train):        # AXPY loop: no big index-copy
            Z0_r += wi * Zr_all[t]
        Z0_r = Z0_r.reshape(fx.n_mu, fx.n_rtot)
        dC = relF(C0, C_q[q0])
        dZ = relF(Z0_r, Ztrue_r) if Ztrue_r is not None else np.nan
        Zt0 = fx.to_sphere(Z0_r, q0)
        U, s, Vh = svd_herm(C0)
        UZ = np.conj(U.T) @ Zt0
        for mode, lam in RUNGS:
            zt0 = (np.conj(Vh.T) * rung_sinv(s, mode, lam)) @ UZ
            B_p = B_from_MG(fx, x @ zt0, q0)
            met = {"B": relF(B_p, B_true),
                   "Bdec": top_decile_rel(B_p, B_true),
                   "tile": relF(make_tile(fx, zt0, q0), V_true),
                   "dC": dC, "dZ": dZ}
            if sname in exciton_stencils:
                if not exc_ready:
                    D, Hdir = build_Hdir(fx, q0)
                    ev_true = exciton_evs(fx, D, Hdir, B_true)
                    exc_ready = True
                ev_p = exciton_evs(fx, D, Hdir, B_p)
                met["exc_meV"] = float(np.max(np.abs(ev_p - ev_true)) * RY2MEV)
            res.setdefault(f"{tag}_{sname}_{rung_label(mode, lam)}",
                           {})[q0] = met


def report(res, title):
    print(f"\n  ========== {title}: median / max over targets ==========")
    print(f"    {'label':<40s} {'B med':>10s} {'B max':>10s} {'Bdec md':>9s} "
          f"{'tile md':>9s} {'dC med':>9s} {'dZ med':>9s} {'exc med':>9s} "
          f"{'exc max':>9s}")
    for lbl in sorted(res):
        rows = res[lbl]

        def g(key):
            return [rows[q][key] for q in rows
                    if key in rows[q] and np.isfinite(rows[q][key])]

        Bm, em, dZ = g("B"), g("exc_meV"), g("dZ")
        dz_s = f"{np.median(dZ):>9.2e}" if dZ else "       --"
        em_s = f"{np.median(em):>9.3f}" if em else "       --"
        ex_s = f"{np.max(em):>9.3f}" if em else "       --"
        print(f"    {lbl:<40s} {np.median(Bm):>10.3e} {np.max(Bm):>10.3e} "
              f"{np.median(g('Bdec')):>9.2e} {np.median(g('tile')):>9.2e} "
              f"{np.median(g('dC')):>9.2e} {dz_s} {em_s} {ex_s}")
    for lbl in sorted(res):
        for q in sorted(res[lbl]):
            m = res[lbl][q]
            extra = f" exc={m['exc_meV']:.3f}meV" if "exc_meV" in m else ""
            print(f"      [row] {lbl} q0={q}: B={m['B']:.3e} "
                  f"Bdec={m['Bdec']:.3e} tile={m['tile']:.3e} "
                  f"dC={m['dC']:.3e} dZ={m['dZ']:.3e}{extra}")


def save_res(res, tag, npz):
    for lbl in res:
        qs = sorted(res[lbl])
        npz[f"{tag}__{lbl}__q0"] = np.array(qs)
        for key in ("B", "Bdec", "tile", "dC", "dZ", "exc_meV"):
            vals = [res[lbl][q].get(key, np.nan) for q in qs]
            npz[f"{tag}__{lbl}__{key}"] = np.array(vals, dtype=float)
