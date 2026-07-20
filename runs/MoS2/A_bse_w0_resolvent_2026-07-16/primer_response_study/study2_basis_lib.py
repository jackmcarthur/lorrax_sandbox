"""study2_basis_lib — principled LR-basis library + generic weighted-LSQ
fit machinery for STUDY 1 (is the b26p polynomial ladder fundamental or
empirical?).

The b26p reference (arbitrary_q_bse.md sec 13, REFERENCE_arbitrary_q_vq.py)
fits the phase-factored LR form factor M_mu(K) per exact discrete K_z = G_z
channel with in-plane Cartesian MONOMIALS K_x^a K_y^b, a+b <= d, degrees
{|G_z|=0:3, 1:2, 2:0, 3:0} -> 26 complex coeffs per mu.  The owner's
question: monomials were picked by measured v_LR weight share, NOT from
completeness/symmetry.  Does a PRINCIPLED bandlimited/orthogonal disk basis
beat or match them?

This module supplies three principled per-channel bases, all fit with the
SAME v_LR-weighted per-q-normal-block LSQ (honest LOO) as b26p — only the
design matrix Phi(K_par) changes:

  monomial  : the b26p baseline (continuity anchor; K/(2a) scaling verbatim).
  bessel    : Fourier-Bessel disk harmonics J_m(k_{m,l} r/R) {cos,sin}(m th),
              Neumann radial roots k_{m,l} = roots of J_m' (so the m=0,l=0
              mode is the CONSTANT — reduces to b26p's monopole at 1 coeff).
              The bandlimited-optimal orthogonal basis on a disk of physical
              radius R = 2a sqrt(ln 1/eps_LR) (the "spherical harmonics of
              the disk").  Modes truncated by bandlimit k_{m,l}/R.
  bessel3   : the SAME, angular-restricted to m == 0 (mod 3) — MoS2 C3v/D3h
              symmetry-adapted (owner (b)).
  zernike   : Zernike disk polynomials R_n^m(r/R){cos,sin}(m th) up to radial
              degree n = d — SAME polynomial function space as monomial
              degree d, but the ORTHOGONAL basis of the disk.  Matched-degree
              Zernike vs monomial isolates conditioning/coeff-stability from
              accuracy (owner (iii): is it the space or the basis?).

Radius per signed-G_z channel is data-driven: R_gz = 1.02 * max_samples
|K_par| in that channel (the in-plane ball shrinks as |K_z| grows).

READ-ONLY on fixtures and sources/lorrax_A.  Complex-capable normal
equations (AtA = Phi^H W Phi); real cos/sin angular parts keep AtA real, so
b26p conditioning is reproduced apples-to-apples.
"""
import numpy as np
from scipy.special import jv, jnp_zeros, eval_jacobi

import REFERENCE_arbitrary_q_vq as R

RIDGE = 1e-11          # normal-equation ridge, relative to trace(AtA)/nb


# ---------------------------------------------------------------------------
# per-channel design-matrix builders (each returns Phi (n_samples, nb) and a
# short human label list of the modes)
# ---------------------------------------------------------------------------
def _polar(Kpar):
    r = np.sqrt(Kpar[0] ** 2 + Kpar[1] ** 2)
    th = np.arctan2(Kpar[1], Kpar[0])
    return r, th


def design_monomial(Kpar, deg, alpha, R_gz):
    """b26p baseline: (K_x/2a)^a (K_y/2a)^b, a+b <= deg (graded order)."""
    s = 1.0 / (2.0 * alpha)
    x, y = Kpar[0] * s, Kpar[1] * s
    cols, lab = [], []
    for t in range(deg + 1):
        for a in range(t + 1):
            b = t - a
            cols.append((x ** a) * (y ** b) if (a or b) else np.ones_like(x))
            lab.append(f"x{a}y{b}")
    return np.stack(cols, 1), lab


def _bessel_modes(nb, symm3=False):
    """Ordered (m, l, parity) mode list by Neumann bandlimit k_{m,l}, up to
    nb real functions.  parity 'c'/'s' for cos/sin (m>0); m=0 -> 'c' only.
    symm3: keep only m == 0 (mod 3)."""
    ms = range(0, 12)
    cand = []                      # (k_ml, m, l, parity)
    for m in ms:
        if symm3 and (m % 3 != 0):
            continue
        # Neumann radial roots of J_m'; prepend k=0 for the m=0 constant
        roots = list(jnp_zeros(m, nb + 2))
        if m == 0:
            roots = [0.0] + roots
        for l, k in enumerate(roots):
            if m == 0:
                cand.append((k, m, l, "c"))
            else:
                cand.append((k, m, l, "c"))
                cand.append((k, m, l, "s"))
    cand.sort(key=lambda t: (t[0], t[1]))
    return cand[:nb]


def design_bessel(Kpar, nb, alpha, R_gz, symm3=False):
    """Fourier-Bessel (Neumann) disk harmonics on radius R_gz, nb lowest
    bandlimit modes.  Real cos/sin angular parts."""
    r, th = _polar(Kpar)
    rho = np.clip(r / R_gz, 0.0, 1.0)
    modes = _bessel_modes(nb, symm3=symm3)
    cols, lab = [], []
    for k, m, l, par in modes:
        rad = jv(m, k * rho)
        if m == 0:
            cols.append(rad)
            lab.append(f"J0_{l}")
        elif par == "c":
            cols.append(rad * np.cos(m * th))
            lab.append(f"J{m}_{l}c")
        else:
            cols.append(rad * np.sin(m * th))
            lab.append(f"J{m}_{l}s")
    return np.stack(cols, 1), lab


def _zernike_radial(n, m, rho):
    """R_n^m(rho) = rho^m P_{(n-m)/2}^{(0,m)}(2 rho^2 - 1), n-m even, via
    Jacobi (numerically stable, matches the standard Zernike normalization
    up to an n-independent constant absorbed by the fit)."""
    k = (n - m) // 2
    return (rho ** m) * eval_jacobi(k, 0, m, 2.0 * rho ** 2 - 1.0)


def _zernike_modes(deg, symm3=False):
    """(n, m, parity) up to radial degree deg; n-|m| even, spans polynomials
    of total degree <= deg on the disk.  symm3: only m == 0 (mod 3)."""
    out = []
    for n in range(deg + 1):
        for m in range(n, -1, -1):
            if (n - m) % 2 != 0:
                continue
            if symm3 and (m % 3 != 0):
                continue
            if m == 0:
                out.append((n, 0, "c"))
            else:
                out.append((n, m, "c"))
                out.append((n, m, "s"))
    return out


def design_zernike(Kpar, deg, alpha, R_gz, symm3=False):
    """Zernike disk polynomials up to radial degree deg on radius R_gz."""
    r, th = _polar(Kpar)
    rho = np.clip(r / R_gz, 0.0, 1.0)
    cols, lab = [], []
    for n, m, par in _zernike_modes(deg, symm3=symm3):
        rad = _zernike_radial(n, m, rho)
        if m == 0:
            cols.append(rad)
            lab.append(f"Z{n}0")
        elif par == "c":
            cols.append(rad * np.cos(m * th))
            lab.append(f"Z{n}{m}c")
        else:
            cols.append(rad * np.sin(m * th))
            lab.append(f"Z{n}{m}s")
    return np.stack(cols, 1), lab


# ---------------------------------------------------------------------------
# a Basis bundles the per-channel design closures + the coarse-side samples
# ---------------------------------------------------------------------------
class Basis:
    """Generic per-signed-G_z basis over the LR superset.  `chan_design` is
    a dict {signed_gz: callable(Kpar_2xn) -> (Phi (n,nb), labels)} covering
    only the modeled channels (others are model-zero, like b26p)."""

    def __init__(self, fx, prep, chan_design, tag):
        self.fx, self.prep, self.tag = fx, prep, tag
        self.chan = chan_design
        self.gz_cols = prep["gz_cols"]
        self.alpha = prep["alpha"]
        # per-q design + weighted normal blocks per channel (honest LOO)
        self.blocks = {}       # gz -> (AtA (nq,nb,nb), AtY (nq,nb,nmu))
        self.nb = {}
        for g, design in self.chan.items():
            cols = self.gz_cols[g]
            nbq = None
            AtA = None
            AtY = None
            for q in range(fx["nq"]):
                qG = fx["qfr"][q][:, None] + prep["GS"][:, cols].astype(float)
                Kpar = (fx["bvec"].T @ qG)[:2]
                Phi, _ = design(Kpar)
                if AtA is None:
                    nbq = Phi.shape[1]
                    AtA = np.empty((fx["nq"], nbq, nbq), dtype=complex)
                    AtY = np.empty((fx["nq"], nbq, fx["n_mu"]), dtype=complex)
                w = prep["W"][q][cols]
                Pw = Phi * w[:, None]
                AtA[q] = Phi.conj().T @ Pw
                AtY[q] = Pw.conj().T @ prep["Fch"][q][:, cols].T
            self.blocks[g] = (AtA, AtY)
            self.nb[g] = nbq

    def n_coeff(self):
        return int(sum(self.nb.values()))

    def coeffs(self, exclude=None):
        sel = [q for q in range(self.fx["nq"]) if q != exclude]
        out = {}
        for g, (AtA, AtY) in self.blocks.items():
            A = AtA[sel].sum(0)
            Y = AtY[sel].sum(0)
            tr = np.trace(A).real / A.shape[0]
            A = A + RIDGE * tr * np.eye(A.shape[0])
            out[g] = np.linalg.solve(A, Y)
        return out

    def cond_blocks(self):
        """cond2 of the summed (ridged) normal block per channel."""
        out = {}
        for g, (AtA, _) in self.blocks.items():
            A = AtA.sum(0)
            tr = np.trace(A).real / A.shape[0]
            A = A + RIDGE * tr * np.eye(A.shape[0])
            s = np.linalg.svd(A, compute_uv=False)
            out[g] = float(s[0] / max(s[-1], 1e-300))
        return out

    def model_M(self, coeffs, qfrac):
        """M_mu(K) on the full superset at qfrac (nmu, nG); phase NOT applied
        (lr_model_tile applies e^{-iK.s_mu})."""
        fx, prep = self.fx, self.prep
        M = np.zeros((fx["n_mu"], prep["GS"].shape[1]), dtype=complex)
        Kall = fx["bvec"].T @ (np.asarray(qfrac)[:, None]
                               + prep["GS"].astype(float))
        for g, design in self.chan.items():
            cols = self.gz_cols[g]
            Phi, _ = design(Kall[:2][:, cols])
            M[:, cols] = (Phi @ coeffs[g]).T
        return M


def lr_tile_from_M(fx, prep, M, qfrac):
    """Closed-form LR tile from a model M on the superset (mirrors
    REFERENCE_arbitrary_q_vq.lr_model_tile but taking M directly)."""
    GS = prep["GS"]
    qf = np.asarray(qfrac, float)
    v = R.v_slab_on_set(fx, qf, GS, kind="slab_lr", alpha=prep["alpha"])
    qG = qf[None, :] + GS.T.astype(float)
    zt = np.exp(-2j * np.pi * (fx["rmu_frac"] @ qG.T)) * M
    A = zt * np.sqrt(v)[None, :]
    return np.conj(A) @ A.T


# ---------------------------------------------------------------------------
# basis factories matched to the b26p budget {0:3,1:2,2:0,3:0}
# ---------------------------------------------------------------------------
# b26p signed-channel coefficient counts: gz=0 ->10, |gz|=1 ->6, |gz|=2,3 ->1
DEG_B26P = {0: 3, 1: 2, 2: 0, 3: 0}
CNT_B26P = {0: 10, 1: 6, 2: 1, 3: 1}      # per SIGNED channel (monomials of deg)


def _channel_radii(fx, prep):
    """data-driven R_gz = 1.02 * max_samples |K_par| per signed channel."""
    Rg = {}
    for g, cols in prep["gz_cols"].items():
        rmax = 0.0
        for q in range(fx["nq"]):
            qG = fx["qfr"][q][:, None] + prep["GS"][:, cols].astype(float)
            Kpar = (fx["bvec"].T @ qG)[:2]
            rmax = max(rmax, float(np.max(np.hypot(Kpar[0], Kpar[1]))))
        Rg[g] = 1.02 * max(rmax, 1e-6)
    return Rg


def make_monomial(fx, prep, degrees=DEG_B26P):
    a = prep["alpha"]
    Rg = _channel_radii(fx, prep)
    chan = {}
    for g in prep["gz_cols"]:
        d = degrees.get(abs(g))
        if d is None:
            continue
        chan[g] = (lambda K, d=d, r=Rg[g]:
                   design_monomial(K, d, a, r))
    return Basis(fx, prep, chan, "monomial")


def make_bessel(fx, prep, counts=CNT_B26P, symm3=False):
    a = prep["alpha"]
    Rg = _channel_radii(fx, prep)
    chan = {}
    for g in prep["gz_cols"]:
        nb = counts.get(abs(g))
        if nb is None:
            continue
        chan[g] = (lambda K, nb=nb, r=Rg[g]:
                   design_bessel(K, nb, a, r, symm3=symm3))
    return Basis(fx, prep, chan, "bessel3" if symm3 else "bessel")


def make_zernike(fx, prep, degrees=DEG_B26P, symm3=False):
    a = prep["alpha"]
    Rg = _channel_radii(fx, prep)
    chan = {}
    for g in prep["gz_cols"]:
        d = degrees.get(abs(g))
        if d is None:
            continue
        chan[g] = (lambda K, d=d, r=Rg[g]:
                   design_zernike(K, d, a, r, symm3=symm3))
    return Basis(fx, prep, chan, "zernike3" if symm3 else "zernike")
