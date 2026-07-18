"""lr_prep — shared machinery for the LR-channel COMPACT-REPRESENTATION study.

HYPOTHESIS UNDER TEST (parent reframing of arbitrary_q_bse.md sec 12): the
phase-factored form factors F_mu(q;G) = e^{+iK.s_mu} zeta~_c,mu(K), K = q+G,
carried by the F-scheme as n_mu x |gset| explicit channels PER COARSE q, are
scattered SAMPLES of one would-be smooth function M_mu(K) on the LR ball
|K| <= 2 alpha sqrt(ln 1/eps_LR).  Literal K=0 Taylor moments of M fail
(sec 12.2: 58-70% model error, Brunin hierarchy inverts) because Taylor
diverges from the truth by |K| ~ alpha; the right object may be a WEIGHTED
LSQ FIT of a compact analytic basis over the whole ball (the pasted
response's own untested sec-4 'fit-based extraction').

KEY STRUCTURAL FACT (verified in lr_singlevalued stage G): with q on the
coarse NxN grid (q_z = 0) and G in the fixed Miller superset, the in-plane
sample points K_par = B^T (n/N) tile a REGULAR fine lattice (spacing |b|/N)
per exact discrete K_z = G_z channel, and every fine point comes from
exactly ONE (q, G) pair.  So "is M single-valued in K?" cannot be posed as
a coincidence test; it becomes three measurable statements:
  (1) seam parity: adjacent fine-lattice pairs that cross a BZ boundary
      (Miller label changes) look statistically identical to pairs that do
      not — no residual seam beyond the analytically-carried phase;
  (2) no q-fiber: after fitting a rich smooth basis of K, the residuals do
      NOT cluster by source q (each coarse fit contributes no coherent
      per-q offset beyond smooth K-variation);
  (3) plateau: the weighted fit residual keeps falling with basis richness
      to a floor consistent with (2), not with a per-q obstruction.

FIT (per G_z channel, in-plane basis, weight = physical tile weight):
    min_c sum_s v_LR(K_s) | sum_b Phi_b(K_par,s) c_b[mu] - M_s[mu] |^2
The weight makes the objective exactly ||Delta A||_F^2 of the LR tile
factor A = zt sqrt(v_LR) (V = conj(A) A^T), i.e. the fit minimizes what the
physical contraction sees.  The design matrix depends on K only, so ONE
weighted normal solve per (G_z, LOO-target) serves all n_mu rows; normal
blocks are accumulated PER COARSE q so leave-one-out refits are honest
(target q's samples excluded) and O(n_b^2).

BASES (real functions of Cartesian K_par, scaled by 1/(2 alpha) for
conditioning; evaluation at ANY Q is closed-form; the winding phase
e^{-iK.s_mu} is NEVER fit through — it is re-applied analytically, exactly
as in tile_prep T2'):
    poly-d : K_x^a K_y^b, a+b <= d        ("renormalized moments": the LSQ
             analogue of the T2 moments, all orders fit against the ball)
    gto    : e^{-K^2/(4 sig_j^2)} x poly, even-tempered sig ladder
             (periodic-RI / Gaussian-density-fitting style radial freedom)
Per-G_z budgets follow the measured v_LR weight shares (0.55 / 0.19+0.19 /
0.028+0.028 / ... at alpha=0.3); channels outside the spec are model-zero.

SVD COMPRESSION ("learned multipoles"): fit a rich shared basis, transform
coefficients to the weighted-Gram-orthonormal basis metric (block-diag per
G_z), SVD the (n_btot x n_mu) coefficient matrix; rank-r truncation leaves
r mixing coefficients per mu against r shared analytic K-profiles.  The
same spectrum is measured model-free from the sqrt(w)-weighted sample
matrix itself (effective rank of the LR channel data).

READ-ONLY on all fixtures and on sources/lorrax_A. No Z_r object anywhere.
"""
import numpy as np


# ---------------------------------------------------------------------------
# basis construction
# ---------------------------------------------------------------------------
def poly_pairs(d):
    """[(a, b)] with a+b <= d, graded order."""
    return [(a, t - a) for t in range(d + 1) for a in range(t + 1)]


def spec_poly(d):
    """Pure polynomial spec: [(None, a, b)]."""
    return [(None, a, b) for a, b in poly_pairs(d)]


def spec_gto(d, sigmas):
    """Gaussian-radial x polynomial spec: [(sig, a, b)]."""
    return [(s, a, b) for s in sigmas for a, b in poly_pairs(d)]


def eval_basis(Kpar, spec, alpha):
    """Design matrix (n, nb) for in-plane Cartesian K rows Kpar (2, n).
    Terms: (K_x/2a)^p (K_y/2a)^q [* exp(-K^2/(4 sig^2))]."""
    s = 1.0 / (2.0 * alpha)
    x, y = Kpar[0] * s, Kpar[1] * s
    K2 = Kpar[0] ** 2 + Kpar[1] ** 2
    cols = []
    for sig, a, b in spec:
        t = (x ** a) * (y ** b) if (a or b) else np.ones_like(x)
        if sig is not None:
            t = t * np.exp(-K2 / (4.0 * sig ** 2))
        cols.append(t)
    return np.stack(cols, 1)


# ---------------------------------------------------------------------------
# sample container: K, weights, F-channel values, fine-lattice bookkeeping
# ---------------------------------------------------------------------------
class LRSamples:
    """All (q, G) samples of the cleaned phase-factored form factors on
    gset(alpha), with the physical weight w = v_LR(K) and the fine-lattice
    integer coordinates that expose the one-sample-per-K structure."""

    def __init__(self, ts, rc, alpha):
        fx = ts.fx
        self.ts, self.fx, self.alpha, self.rc = ts, fx, alpha, rc
        assert np.max(np.abs(fx.qfr[:, 2])) < 1e-12, "slab study needs q_z=0"
        self.GS = ts.gset(alpha)                       # (3, nG) Miller
        self.nG = self.GS.shape[1]
        self.Fch = ts.F_channels(rc, alpha)            # (nq, nmu, nG)
        nq = fx.nq
        self.K = np.empty((nq, 3, self.nG))
        self.W = np.empty((nq, self.nG))
        for q in range(nq):
            self.K[q] = fx.bvec.T @ (fx.qfr[q][:, None]
                                     + self.GS.astype(np.float64))
            self.W[q] = ts.v_on_set(fx.qfr[q], self.GS, kind="slab_lr",
                                    alpha=alpha)
        self.gz = self.GS[2].copy()                    # per-column int Gz
        self.gz_vals = [int(g) for g in np.unique(self.gz)]
        self.cols = {g: np.where(self.gz == g)[0] for g in self.gz_vals}
        # fine-lattice integer coords per sample: N*(q_frac + G)_par is an
        # exact integer on the coarse grid (gate: max deviation printed)
        kg = fx.kgrid
        assert kg[0] == kg[1], "in-plane fine lattice assumes Nx == Ny"
        self.Nf = int(kg[0])
        fin = (fx.qfr[:, None, :2] + self.GS.T[None, :, :2]) * self.Nf
        self.fine_dev = float(np.max(np.abs(fin - np.rint(fin))))
        self.fine = np.rint(fin).astype(np.int64)      # (nq, nG, 2)

    def wshare_gz(self):
        """v_LR weight share per Gz channel, summed over all coarse q."""
        tot = self.W.sum()
        return {g: float(self.W[:, self.cols[g]].sum() / tot)
                for g in self.gz_vals}


# ---------------------------------------------------------------------------
# per-Gz weighted LSQ with per-q normal blocks (honest LOO refits)
# ---------------------------------------------------------------------------
class ChannelFit:
    """specs: {gz: [(sig|None, a, b), ...]}; channels absent from specs are
    model-zero.  Specs are auto-clamped so nb <= 0.6 * n_inplane_samples."""

    RIDGE = 1e-11

    def __init__(self, lr, specs, tag=""):
        self.lr, self.tag = lr, tag
        fx = lr.fx
        self.specs = {}
        self.blocks = {}                    # gz -> (AtA (nq,nb,nb), AtY (nq,nb,nmu))
        self.resid_blocks = {}              # gz -> per-q (Phi, w, Y) closures
        for g, spec in specs.items():
            if g not in lr.cols:
                continue
            c = lr.cols[g]
            # clamp: distinct K_par samples per gz = len(c) * nq (fine
            # lattice, one per (q,G)); keep nb well below that
            nmax = max(1, min(len(spec), int(0.6 * len(c) * lr.fx.nq)))
            spec = list(spec)[:nmax]
            self.specs[int(g)] = spec
            nb = len(spec)
            AtA = np.empty((fx.nq, nb, nb))
            AtY = np.empty((fx.nq, nb, fx.n_mu), dtype=np.complex128)
            for q in range(fx.nq):
                Phi = eval_basis(lr.K[q][:2][:, c], spec, lr.alpha)
                w = lr.W[q][c]
                Y = lr.Fch[q][:, c].T                  # (m, nmu)
                Pw = Phi * w[:, None]
                AtA[q] = Phi.T @ Pw
                AtY[q] = Pw.T @ Y
            self.blocks[int(g)] = (AtA, AtY)

    def n_coeff(self):
        """complex coefficients PER MU (the storage headline)."""
        return sum(len(s) for s in self.specs.values())

    def coeffs(self, exclude=None):
        """{gz: C (nb, nmu)} from all q except `exclude` (honest LOO)."""
        out = {}
        for g, (AtA, AtY) in self.blocks.items():
            sel = [q for q in range(self.lr.fx.nq) if q != exclude]
            A = AtA[sel].sum(0)
            Y = AtY[sel].sum(0)
            A = A + self.RIDGE * (np.trace(A) / A.shape[0]) * np.eye(A.shape[0])
            out[g] = np.linalg.solve(A, Y)
        return out

    def model_F(self, C, qfrac):
        """Model M on gset at momentum qfrac: (nmu, nG); phase NOT applied
        (V_from_F applies e^{-iK.s_mu} analytically)."""
        lr = self.lr
        fx = lr.fx
        out = np.zeros((fx.n_mu, lr.nG), dtype=np.complex128)
        Kall = fx.bvec.T @ (np.asarray(qfrac)[:, None]
                            + lr.GS.astype(np.float64))
        for g, spec in self.specs.items():
            c = lr.cols[g]
            Phi = eval_basis(Kall[:2][:, c], spec, lr.alpha)   # (m, nb)
            out[:, c] = (Phi @ C[g]).T
        return out

    def resid_stats(self, C):
        """Weighted residual power per (q, gz) + totals, for the fiber test.
        Returns dict with rel (global weighted rel resid), per_qg arrays."""
        lr = self.lr
        fx = lr.fx
        num = 0.0
        den = 0.0
        rows = []                    # (q, gz, wsum, resid_pow, fiber_pow)
        for g, spec in self.specs.items():
            c = lr.cols[g]
            for q in range(fx.nq):
                Phi = eval_basis(lr.K[q][:2][:, c], spec, lr.alpha)
                w = lr.W[q][c]
                Y = lr.Fch[q][:, c].T                       # (m, nmu)
                R = Y - Phi @ C[g]
                wsum = float(w.sum())
                rp = float(np.sum(w[:, None] * np.abs(R) ** 2))
                yp = float(np.sum(w[:, None] * np.abs(Y) ** 2))
                if wsum > 0:
                    mq = (w[:, None] * R).sum(0) / wsum     # coherent per-q
                    fp = float(wsum * np.sum(np.abs(mq) ** 2))
                else:
                    fp = 0.0
                rows.append((q, g, wsum, rp, fp, yp))
                num += rp
                den += yp
        rows = np.array(rows)
        return {"rel": np.sqrt(num / den), "rows": rows,
                "fiber_frac": rows[:, 4].sum() / max(rows[:, 3].sum(), 1e-300)}


# ---------------------------------------------------------------------------
# SVD compression of a rich fit ("learned multipoles")
# ---------------------------------------------------------------------------
def svd_compress(cf, C, ranks):
    """Weighted-metric SVD of the coefficient matrix of ChannelFit cf.
    Gram per gz = sum_q AtA_q (the weighted basis Gram, block-diag across
    gz); Chat = L^T C stacked -> SVD -> rank-r truncations mapped back.
    Returns (svals, {r: {gz: C_r}})."""
    Ls, order = {}, []
    blocks = []
    for g, (AtA, _) in cf.blocks.items():
        Gm = AtA.sum(0)
        Gm = Gm + 1e-12 * (np.trace(Gm) / Gm.shape[0]) * np.eye(Gm.shape[0])
        L = np.linalg.cholesky(Gm)
        Ls[g] = L
        order.append(g)
        blocks.append(L.T @ C[g])
    Chat = np.vstack(blocks)
    U, s, Vh = np.linalg.svd(Chat, full_matrices=False)
    out = {}
    for r in ranks:
        Cr_hat = (U[:, :r] * s[:r]) @ Vh[:r]
        Cr, i0 = {}, 0
        for g in order:
            nb = Ls[g].shape[0]
            Cr[g] = np.linalg.solve(Ls[g].T, Cr_hat[i0:i0 + nb])
            i0 += nb
        out[r] = Cr
    return s, out


def sample_matrix_svals(lr, nsv=64):
    """Model-free effective rank: SVD spectrum of the sqrt(w)-weighted
    sample matrix [sqrt(w_s) F_s[mu]] over ALL samples."""
    fx = lr.fx
    cols = []
    for q in range(fx.nq):
        cols.append(lr.Fch[q] * np.sqrt(lr.W[q])[None, :])
    Y = np.concatenate(cols, axis=1)          # (nmu, nq*nG)
    s = np.linalg.svd(Y, compute_uv=False)
    return s[:nsv]
