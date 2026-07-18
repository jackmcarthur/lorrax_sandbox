"""tile_prep — shared machinery for the OWNER-SPEC-COMPLIANT arbitrary-Q
prototype: everything n_mu^2-level, NO r_tot-dimensional object (no Z_q)
anywhere in the interpolation machinery or the per-target cost.

Constructions (driver: tile_t1t2_mos2.py):

  T1  cleaned-tile SR interpolation. At each coarse q, rank-cut-clean the
      stored V_q tile in its own C_q frame, split off the finite-alpha
      Gaussian LR channel, stencil-interpolate the cleaned SR tiles, re-add
      an analytic LR tile at the target.
  T2  phase-factored multipole moments of the CLEANED zeta: slab-adapted
      per-G_z in-plane moments (monopole/dipole/quadrupole about each
      centroid, e^{-iK.s_mu} factored analytically) -> the LR tile at any
      target Q from interpolated moment vectors (n_mu x small).

PER-ELEMENT MATH (conventions inherited from proto1_prep: BGW-wrapped q
labels after offgrid_prep.fix_sphere_wrap; conj-on-left contraction):

(1) Cleaning == rank-cut solve, without Z.  eigh: C_q = R diag(lam) R^H
    (descending), r(q) = #{lam_i > rc * lam_0}.  P = R_r R_r^H (Hermitian).
    Rank-cutting the stored fit is a PROJECTION of the stored zeta:
        zeta_rc = R_r lam_r^{-1} R_r^H Z_true = P @ zeta_stored,
    because Z_true = C zeta_stored.  Since the tile is
        V[mu,nu] = sum_G conj(zt_mu) v zt_nu  ( = conj(A) A^T, A = zt sqrt(v) ),
    replacing zt -> P zt gives
        V_c = conj(P) V P^T = PI V PI,   PI := conj(P)  (Hermitian projector).
    So the CLEANED TILE is computable from the stored tile + C_q alone
    (both n_mu^2; C_q comes from psi at centroids).  Gate:
    makeVq(P @ ZG) == PI V PI to machine precision, and the B-metric
    clean-floor must reproduce the campaign's "rankcut floor on TRUE data"
    (3x3: 3.58e-3; 6x6: 3.2-3.7e-3) — the continuity anchor.

(2) Gaussian split (finite alpha), per G on the stored sphere (owner ruling
    arbitrary_q_bse.md sec 9.8; slab kernel keeps its full f2d envelope):
        v_LR(K) = v(K) exp(-K^2/(4 alpha^2)),  v_SR = v - v_LR (stable expm1),
    K = q+G Cartesian (sqrt(Ry) = 1/bohr).  Coarse-side LR tile from the
    stored zeta is COMPLETE on the stored sphere up to the Gaussian tail:
    contributions with |q+G|^2 > zeta_cutoff are suppressed by
    exp(-cutoff/(4 alpha^2)) (printed; < 1e-9 for alpha <= 0.6, 30 Ry).

(3) T2 moments (slab: q_z = 0 on the coarse grid, so K_z = G_z is a DISCRETE
    exact channel — no z-Taylor, the pasted response's sec-10 refinement).
    Phase factoring:  zeta~_mu(K) = e^{-i K.s_mu} M_mu(K),
        M_mu(K) = sum_delta zeta_lab(s_mu + delta) e^{-i K.delta}
    (delta = r - s_mu; the z part e^{-i Gz_c delta_z} is image-independent
    because Gz is a reciprocal vector, so it is computed as a plain z-FFT;
    the in-plane part is Taylored with MINIMAL-IMAGE fractional in-plane
    displacements, Cartesian via avec):
        M_mu(K_par, Gz) ~= m0_mu(Gz) - i K_par . d_mu(Gz)
                           - 1/2 K_par^T Th_mu(Gz) K_par
        m0_mu(Gz)    = e^{+2 pi i Gz z_mu} sum_xy Fz_mu(x, y, Gz)
        d_mu(Gz)[c]  = e^{+2 pi i Gz z_mu} sum_xy dpar_c(x,y;mu) Fz_mu(x,y,Gz)
        Th_mu(Gz)[cc']= ... dpar_c dpar_c' ...,     Fz = FFT_z[zeta_lab]
    (numpy fft sign e^{-2 pi i k n / N} matches to_sphere's forward
    convention; sums carry NO 1/N — identical normalization to to_sphere).
    The model form factor at ANY K = Q+G:
        ztMP_mu(Q+G) = e^{-2 pi i (Q+G).s_mu,frac} [ m0 - i Kpar.d
                        - 1/2 Kpar.Th.Kpar ]      (order = 0 | 1 | 2)
    e^{-iK.s} is a genuine function of the physical vector K — no G-slot
    label anywhere — so BZ-boundary winding is carried analytically (the
    g0-winding cure).  Moments are n_mu x nGz x {1,2,3} complex — "n_mu x
    small".  They are computed ONCE per coarse q from the cleaned stored fit
    (a one-pass coarse-grid reduction, same license as the production fit);
    the interpolation machinery and every per-target evaluation touch only
    the moment vectors and n_mu^2 tiles.

(4) The LR G-superset (response sec 6: fixed global set, BZ-periodic): all
    Miller G with min_{q in BZ} |q+G|^2 <= 4 alpha^2 ln(1/eps_LR), sampled
    over a fine in-plane q grid; eps_LR = 1e-8.  v evaluated with the same
    slab kernel + zero rule (only the true divergence K^2 < 1e-12 is zeroed)
    as production.

(5) Variant assembly at target q0 (w = truncR stencil weights, train = LOO):
    A rawtile     V = sum_i w_i V_ref(q_i)
    B cleantile   V = sum_i w_i V_c(q_i)
    C cleanSR+LRex V = sum_i w_i [V_c - V_LR_c](q_i) + V_LR_c(q0)
                  (V_LR_c(q0) = PI_0 V_LR(q0) PI_0 from the TARGET's stored
                   zeta — DIAGNOSTIC ceiling only: breaks LOO on the LR
                   channel; PI_0 needs only C_q0, which is psi-level)
    D cleanSR+LRmp V = sum_i w_i [V_c - V_LR_c](q_i) + V_MP[interp mom](q0)
                  (MIXED split: exact-LR subtract, model-LR re-add — the
                   response sec-4 inconsistency, kept to QUANTIFY it)
    E mpconsist   V = sum_i w_i [V_c - V_MP_own](q_i) + V_MP[interp mom](q0)
                  (same model on both sides: algebraically exact at coarse
                   points regardless of model quality — the pasted
                   dipole+quadrupole approach's structure, frame-free)
    Truth: V_ref(q0) = makeVq(stored zeta), the campaign's truth.

READ-ONLY on all fixtures and on sources/lorrax_A.  No Z_r array is ever
allocated in this file or its drivers.
"""
import numpy as np

from proto1_prep import relF, truncR_weights  # noqa: F401

EPS_LR = 1e-8          # Gaussian weight bound defining the LR G-superset
LN_EPS = np.log(1.0 / EPS_LR)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def avec_from_bvec(fx):
    """Real-space lattice vectors (rows a_i, Cartesian bohr): a_i.b_j = 2pi
    delta_ij with fx.bvec rows = b_i in 1/bohr (blat-scaled). Gate vs adot."""
    A = 2.0 * np.pi * np.linalg.inv(fx.bvec.T)        # A @ fx.bvec.T = 2pi I
    err = np.max(np.abs(A @ A.T - fx.adot)) / np.max(np.abs(fx.adot))
    return A, err


def check_slab_axes(fx):
    """z-separability gate: b3 || z and b1,b2 in-plane (slab fixtures)."""
    off = max(np.max(np.abs(fx.bvec[2, :2])), np.max(np.abs(fx.bvec[:2, 2])))
    return off / np.abs(fx.bvec[2, 2])


# ---------------------------------------------------------------------------
# per-fixture study state: eigh of C_q, cleaned tiles, splits, moments
# ---------------------------------------------------------------------------
class TileStudy:
    def __init__(self, fx, C_q):
        self.fx = fx
        self.C_q = C_q
        self.avec, self.avec_err = avec_from_bvec(fx)
        self.eig = []                      # per q: (lam desc, R desc)
        for q in range(fx.nq):
            lam, R = np.linalg.eigh(0.5 * (C_q[q] + C_q[q].conj().T))
            self.eig.append((lam[::-1].copy(), R[:, ::-1].copy()))
        # reference tiles (stored zeta, slab kernel; gate: == disk to ~5e-6)
        self.V_ref = np.stack([fx.make_Vq(fx.ZG[q], q, kind="slab")
                               for q in range(fx.nq)])
        self._P = {}                       # (q, rc) -> P
        self._Vc = {}                      # rc -> (nq, mu, mu)
        self._VLRc = {}                    # (rc, alpha) -> (nq, mu, mu)
        self._mom = {}                     # rc -> moments dict
        self._gset = {}                    # alpha -> (3, nG) int
        self._gz = None                    # global Gz list (ints)

    # -------------------- cleaning --------------------------------------
    def P(self, q, rc):
        key = (q, rc)
        if key not in self._P:
            lam, R = self.eig[q]
            r = int(np.sum(lam > rc * lam[0]))
            Rr = R[:, :r]
            self._P[key] = Rr @ Rr.conj().T
        return self._P[key]

    def rank(self, q, rc):
        lam, _ = self.eig[q]
        return int(np.sum(lam > rc * lam[0]))

    def Vc(self, rc):
        if rc not in self._Vc:
            fx = self.fx
            out = np.empty_like(self.V_ref)
            for q in range(fx.nq):
                PI = np.conj(self.P(q, rc))
                out[q] = PI @ self.V_ref[q] @ PI
            self._Vc[rc] = out
        return self._Vc[rc]

    # -------------------- exact LR on the stored sphere -----------------
    def VLR_exact_c(self, rc, alpha):
        """PI V_LR PI at every coarse q == makeVq(P zeta, slab_lr)."""
        key = (rc, alpha)
        if key not in self._VLRc:
            fx = self.fx
            out = np.empty_like(self.V_ref)
            for q in range(fx.nq):
                VLR = fx.make_Vq(fx.ZG[q], q, kind="slab_lr", alpha=alpha)
                PI = np.conj(self.P(q, rc))
                out[q] = PI @ VLR @ PI
            self._VLRc[key] = out
        return self._VLRc[key]

    def sphere_tail_bound(self, alpha):
        return float(np.exp(-self.fx.zeta_cutoff / (4.0 * alpha ** 2)))

    # -------------------- LR G-superset ---------------------------------
    def gset(self, alpha):
        """Fixed global Miller set: min_{q in BZ} |q+G|^2 <= 4 a^2 ln(1/eps)."""
        if alpha not in self._gset:
            fx = self.fx
            K2max = 4.0 * alpha ** 2 * LN_EPS
            Kmax = np.sqrt(K2max)
            nmax = [int(np.ceil(Kmax / np.linalg.norm(fx.bvec[i]))) + 1
                    for i in range(3)]
            gr = [np.arange(-n, n + 1) for n in nmax]
            GX, GY, GZ = np.meshgrid(*gr, indexing="ij")
            Gall = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 0)
            ts = np.linspace(-0.5, 0.5, 13, endpoint=False)
            m = np.full(Gall.shape[1], np.inf)
            for tx in ts:
                for ty in ts:
                    qf = np.array([tx, ty, 0.0])
                    K = fx.bvec.T @ (qf[:, None] + Gall.astype(np.float64))
                    m = np.minimum(m, np.sum(K * K, axis=0))
            self._gset[alpha] = np.ascontiguousarray(Gall[:, m <= K2max])
        return self._gset[alpha]

    def gz_list(self, alpha_max):
        if self._gz is None:
            GS = self.gset(alpha_max)
            gz = np.unique(GS[2])
            assert np.max(np.abs(gz)) < self.fx.nz // 2, "Gz aliases FFT box"
            self._gz = gz
        return self._gz

    def v_on_set(self, qfrac, GS, kind="slab", alpha=None):
        """Slab kernel on an explicit Miller set at momentum qfrac (wrapped),
        same zero rule as production (only K^2 < 1e-12 zeroed)."""
        fx = self.fx
        K = fx.bvec.T @ (np.asarray(qfrac)[:, None] + GS.astype(np.float64))
        K2 = np.sum(K * K, axis=0)
        zero = K2 < 1e-12
        K2s = np.where(zero, 1.0, K2)
        zc = np.pi / fx.bvec[2, 2]
        f2d = 1.0 - np.exp(-zc * np.sqrt(K[0] ** 2 + K[1] ** 2)) \
            * np.cos(K[2] * zc)
        v = 8.0 * np.pi / K2s * f2d / fx.celvol
        if kind == "slab_lr":
            v = v * np.exp(-K2 / (4.0 * alpha ** 2))
        elif kind == "slab_sr":
            v = v * (-np.expm1(-K2 / (4.0 * alpha ** 2)))
        return np.where(zero, 0.0, v)

    # -------------------- T2 moments ------------------------------------
    def _recon_rows(self, q, ZGq):
        """proto1_prep.Fixture.recon body with supplied sphere coefficients."""
        fx = self.fx
        nmu = ZGq.shape[0]
        box = np.zeros((nmu, fx.n_rtot), dtype=np.complex128)
        fi = fx.flat_idx(fx.gvec[q])
        n = int(fx.ngk[q])
        box[:, fi[:n]] = ZGq[:, :n]
        R = np.fft.ifftn(box.reshape(nmu, fx.nx, fx.ny, fx.nz),
                         axes=(1, 2, 3), norm="backward"
                         ).reshape(nmu, fx.n_rtot)
        return R * np.exp(2j * np.pi * (fx.rfrac @ fx.qfr[q]))[None, :]

    def moments(self, rc, alpha_max, zdiag=False):
        """Per-Gz in-plane moments of the CLEANED zeta at every coarse q.
        Returns dict with m0 (nq, nmu, nGz), d (nq, nmu, 2, nGz),
        Th (nq, nmu, 3, nGz) [xx, xy, yy], gz (nGz,) ints; if zdiag, also
        z-moments dz/Thzz at Gz=0 (for the pure-3D-Taylor diagnostic)."""
        if rc in self._mom:
            return self._mom[rc]
        fx = self.fx
        gz = self.gz_list(alpha_max)
        nGz = len(gz)
        sel = [int(g) % fx.nz for g in gz]
        Apar = self.avec[:2, :2]           # in-plane lattice rows (gated)
        # fractional in-plane grid (nx*ny, 2)
        fx_x = np.arange(fx.nx) / fx.nx
        fx_y = np.arange(fx.ny) / fx.ny
        FXY = np.stack(np.meshgrid(fx_x, fx_y, indexing="ij"),
                       -1).reshape(-1, 2)
        zfr = np.arange(fx.nz) / fx.nz
        m0 = np.zeros((fx.nq, fx.n_mu, nGz), dtype=np.complex128)
        dd = np.zeros((fx.nq, fx.n_mu, 2, nGz), dtype=np.complex128)
        Th = np.zeros((fx.nq, fx.n_mu, 3, nGz), dtype=np.complex128)
        dz = np.zeros((fx.nq, fx.n_mu), dtype=np.complex128)
        Tz = np.zeros((fx.nq, fx.n_mu), dtype=np.complex128)
        for q in range(fx.nq):
            ZGc = self.P(q, rc) @ fx.ZG[q] if rc is not None else fx.ZG[q]
            zlab = self._recon_rows(q, ZGc).reshape(fx.n_mu, fx.nx, fx.ny,
                                                    fx.nz)
            Fz = np.fft.fft(zlab, axis=3)[:, :, :, sel]   # (nmu,nx,ny,nGz)
            Fz = Fz.reshape(fx.n_mu, fx.nx * fx.ny, nGz)
            zph = np.exp(2j * np.pi * np.outer(fx.rmu_frac[:, 2],
                                               gz.astype(float)))
            m0[q] = zph * np.einsum("mxg->mg", Fz)
            # per-mu minimal-image in-plane displacement (Cartesian)
            for mu in range(fx.n_mu):
                df = FXY - fx.rmu_frac[mu, :2][None, :]
                df = (df + 0.5) % 1.0 - 0.5
                dc = df @ Apar                       # (nx*ny, 2) bohr
                w = np.stack([dc[:, 0], dc[:, 1],
                              dc[:, 0] * dc[:, 0], dc[:, 0] * dc[:, 1],
                              dc[:, 1] * dc[:, 1]], 0)
                mom = w @ Fz[mu]                     # (5, nGz)
                dd[q, mu] = zph[mu][None, :] * mom[:2]
                Th[q, mu] = zph[mu][None, :] * mom[2:]
            if zdiag:
                dzf = (zfr - fx.rmu_frac[:, 2][:, None] + 0.5) % 1.0 - 0.5
                dzc = dzf * (self.avec[2, 2])        # bohr (c || z, gated)
                zl = zlab.reshape(fx.n_mu, fx.nx * fx.ny, fx.nz)
                s_xy = np.einsum("mxz->mz", zl)
                dz[q] = np.einsum("mz,mz->m", s_xy, dzc)
                Tz[q] = np.einsum("mz,mz->m", s_xy, dzc * dzc)
            del zlab, Fz
        out = {"m0": m0, "d": dd, "Th": Th, "gz": gz, "dz": dz, "Tzz": Tz}
        self._mom[rc] = out
        return out

    # -------------------- moment-model form factor + LR tile ------------
    def model_zt(self, mom_q, qfrac, GS, order):
        """ztMP (n_mu, nG) at momentum qfrac on Miller set GS from ONE set of
        moment vectors mom_q = dict(m0 (nmu,nGz), d, Th, gz)."""
        fx = self.fx
        gz = mom_q["gz"]
        gzpos = {int(g): i for i, g in enumerate(gz)}
        idx = np.array([gzpos[int(g)] for g in GS[2]])
        K = fx.bvec.T @ (np.asarray(qfrac)[:, None] + GS.astype(np.float64))
        ser = mom_q["m0"][:, idx].astype(np.complex128).copy()
        if order >= 1:
            ser -= 1j * (K[0][None, :] * mom_q["d"][:, 0][:, idx]
                         + K[1][None, :] * mom_q["d"][:, 1][:, idx])
        if order >= 2:
            ser -= 0.5 * (K[0] ** 2 * mom_q["Th"][:, 0][:, idx]
                          + 2.0 * K[0] * K[1] * mom_q["Th"][:, 1][:, idx]
                          + K[1] ** 2 * mom_q["Th"][:, 2][:, idx])
        qG = np.asarray(qfrac)[None, :] + GS.T.astype(np.float64)  # (nG, 3)
        phase = np.exp(-2j * np.pi * (fx.rmu_frac @ qG.T))         # (nmu, nG)
        return phase * ser

    def V_MP(self, mom_q, qfrac, GS, alpha, order, drop_g0=False):
        v = self.v_on_set(qfrac, GS, kind="slab_lr", alpha=alpha)
        if drop_g0:
            v = np.where(np.all(GS == 0, axis=0), 0.0, v)
        zt = self.model_zt(mom_q, qfrac, GS, order)
        A = zt * np.sqrt(v)[None, :]
        return np.conj(A) @ A.T

    # -------------------- T2': phase-factored LR channels (no Taylor) ----
    def sphere_slot(self, q, GS):
        """Map Miller columns of GS to stored-sphere slots at q; -1 where a
        superset G is outside the stored sphere at this q (the stored fit is
        band-limited there — the channel is zero in the stored
        representation, and its Gaussian weight at any such q is bounded by
        exp(-cutoff/(4 alpha^2)))."""
        fx = self.fx
        n = int(fx.ngk[q])
        lut = {tuple(g): i for i, g in enumerate(fx.gvec[q][:, :n].T)}
        return np.array([lut.get(tuple(g), -1) for g in GS.T])

    def F_channels(self, rc, alpha, verbose=False):
        """F[q][mu, j] = e^{+2 pi i (q+G_j).s_mu} zeta_c,mu(q+G_j) on
        gset(alpha) — the exact form factor M_mu(K) sampled at K = q+G
        (the T2 moment model is its in-plane Taylor).  n_mu x nG_LR per q;
        interpolating it componentwise at fixed Miller G is winding-safe
        because the centroid phase is factored analytically.  Out-of-sphere
        (q, G) channels are zero (stored-representation-consistent); their
        worst-case Gaussian weight is printed."""
        GS = self.gset(alpha)
        fx = self.fx
        out = np.empty((fx.nq, fx.n_mu, GS.shape[1]), dtype=np.complex128)
        wmax, nmiss = 0.0, 0
        for q in range(fx.nq):
            zt = self.P(q, rc) @ fx.ZG[q] if rc is not None else fx.ZG[q]
            idx = self.sphere_slot(q, GS)
            miss = idx < 0
            if np.any(miss):
                nmiss += int(miss.sum())
                K = fx.bvec.T @ (fx.qfr[q][:, None]
                                 + GS[:, miss].astype(np.float64))
                K2 = np.sum(K * K, axis=0)
                wmax = max(wmax, float(np.exp(-K2 / (4 * alpha ** 2)).max()))
            zt_ext = np.concatenate(
                [zt, np.zeros((fx.n_mu, 1), dtype=np.complex128)], axis=1)
            qG = fx.qfr[q][None, :] + GS.T.astype(np.float64)
            ph = np.exp(2j * np.pi * (fx.rmu_frac @ qG.T))
            out[q] = ph * zt_ext[:, idx]
        if verbose or nmiss:
            print(f"  [info] F_channels(rc={rc}, a={alpha}): {nmiss} "
                  f"out-of-sphere (q,G) channels zero-filled; worst "
                  f"Gaussian weight {wmax:.2e}")
        return out

    def V_from_F(self, Fq, qfrac, GS, alpha, drop_g0=False):
        """LR tile at qfrac from an F-channel array (own or interpolated)."""
        v = self.v_on_set(qfrac, GS, kind="slab_lr", alpha=alpha)
        if drop_g0:
            v = np.where(np.all(GS == 0, axis=0), 0.0, v)
        qG = np.asarray(qfrac)[None, :] + GS.T.astype(np.float64)
        zt = np.exp(-2j * np.pi * (self.fx.rmu_frac @ qG.T)) * Fq
        A = zt * np.sqrt(v)[None, :]
        return np.conj(A) @ A.T

    def mom_at(self, mom, sel):
        """Slice per-q moment fields -> the mom_q dict for model_zt."""
        return {"m0": mom["m0"][sel], "d": mom["d"][sel],
                "Th": mom["Th"][sel], "gz": mom["gz"]}

    def mom_interp(self, mom, w, train):
        """Stencil-combined moment vectors (componentwise; winding lives in
        the analytic e^{-iK.s} phase, never in these fields)."""
        return {"m0": np.tensordot(w, mom["m0"][train], axes=(0, 0)),
                "d": np.tensordot(w, mom["d"][train], axes=(0, 0)),
                "Th": np.tensordot(w, mom["Th"][train], axes=(0, 0)),
                "gz": mom["gz"]}


# ---------------------------------------------------------------------------
# physical metric on tiles
# ---------------------------------------------------------------------------
def B_tile(x, V):
    """B[p,p'] = sum_{mu nu} conj(x[p,mu]) V[mu,nu] x[p',nu]  — identical to
    offgrid_prep.B_from_MG(x @ zeta) by bilinearity (gated in the driver)."""
    return np.conj(x) @ V @ x.T
