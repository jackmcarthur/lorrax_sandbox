"""Stage-0 ψ-unfold re-validation on CrI3 6×6 30Ry SOC bispinor.

Re-runs task #47's comprehensive_psi_unfold harness against the current
lorrax_B HEAD (post-{V_q IBZ cascade, signature refactors, sym
consolidation, ~360 lines cleanup}) to confirm
    ψ_n(Sk) = U_spinor(S) · ψ_n(k_IBZ)
still holds for every band n, every full-BZ k, and every sym op s
(spatial + TRS-augmented) on CrI3 6×6 30Ry SOC bispinor.

Metrics reported per (k_full, sym_idx) pair:

  1. **Subspace unit_err** = ||U U^H - I||_∞ where U_ij = ⟨ψ_unfold,i | ψ_nosym,j⟩
     within each degenerate band group (Ry tol 1e-5). This is the
     gauge-invariant subspace agreement.
  2. **Strict per-band overlap defect** = max_n (1 - |⟨ψ_unfold,n | ψ_nosym,n⟩|),
     where the comparison is between the LORRAX-unfolded ψ and the nosym ψ
     at the matching full-BZ k. For non-degenerate bands this should be ULP;
     for degenerate bands it can sit at the SCF-noise floor between the two
     independent SCFs.
  3. **Optimal-gauge raw element-wise residual**:
        min_φ max_{G,α} |ψ_unfold,n,α,G - e^{iφ} · ψ_nosym,n,α,G|
     where φ = arg(⟨ψ_unfold | ψ_nosym⟩). Per-band; reported as max over (n,k).

Gate (task spec): max strict per-band overlap defect < 1e-10.
Realistic floor on this system: ~3e-7 (independent-SCF noise on the
non-bit-equivalent k's, per task #47's CrI3_results.json).

Outputs
-------
data/CrI3_stage0_results.json   structured per-(k,sym) numbers
data/run.log                   tee'd stdout

Spinor index convention: WfnLoader returns ψ as (nb, ns, ngk) — spin axis
in the MIDDLE (verified from task #47 harness line 35 and live HEAD's
WfnLoader.load).
"""
import sys, os, json, time
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_B/src")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from file_io.wfn_loader import WfnLoader


SYM_WFN  = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5"
NOSYM_WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/qe_nosym/nscf/WFN.h5"
B_LO, B_HI = 0, 32   # same window as task #47
DEG_TOL = 1e-5       # Ry


def g_to_box(g, fft):
    Nx, Ny, Nz = fft
    g = np.asarray(g, dtype=np.int64)
    return ((g[..., 0] % Nx) * Ny + (g[..., 1] % Ny)) * Nz + (g[..., 2] % Nz)


def scatter(psi, gv, ngk_v, fft):
    """psi: (nb, ns, ngk) -> (nb, ns, Nx*Ny*Nz) FFT-box flat."""
    nb, ns, _ = psi.shape
    Nx, Ny, Nz = fft
    box = np.zeros((nb, ns, Nx * Ny * Nz), dtype=np.complex128)
    bidx = g_to_box(gv[:ngk_v], (Nx, Ny, Nz))
    box[:, :, bidx] = psi[:, :, :ngk_v]
    return box


def build_match_table(kpts_a, kpts_b, tol=1e-6):
    a = np.asarray(kpts_a, dtype=np.float64); b = np.asarray(kpts_b, dtype=np.float64)
    out = np.full(a.shape[0], -1, dtype=np.int64)
    for i in range(a.shape[0]):
        diff = b - a[i][None, :]
        diff_w = diff - np.round(diff)
        norms = np.max(np.abs(diff_w), axis=1)
        j = int(np.argmin(norms))
        if norms[j] < tol:
            out[i] = j
    return out


def degenerate_groups(energies, tol):
    nb = len(energies)
    groups = []
    i = 0
    while i < nb:
        j = i + 1
        while j < nb and abs(energies[j] - energies[i]) < tol:
            j += 1
        groups.append((i, j))
        i = j
    return groups


def main():
    out_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/bispinor_ibz_stage0_psi_check_2026-05-16/data"
    os.makedirs(out_dir, exist_ok=True)

    print(f"sym   WFN: {SYM_WFN}")
    print(f"nosym WFN: {NOSYM_WFN}")
    print(f"band window [{B_LO}, {B_HI}), deg_tol={DEG_TOL} Ry")
    print(f"jax devices: {jax.devices()}")

    t0 = time.time()
    loader_sym = WfnLoader(SYM_WFN)
    loader_nos = WfnLoader(NOSYM_WFN)
    sym = loader_sym._ensure_sym()
    fft_grid = tuple(int(x) for x in loader_sym.fft_grid)
    ntran = int(loader_sym.ntran)
    nk_full = int(sym.nk_tot)
    ns = int(loader_sym.nspinor)
    print(f"ntran={ntran}, nk_full={nk_full}, nspinor={ns}, fft_grid={fft_grid}")
    assert int(loader_nos.ntran) == 1, f"nosym WFN should have ntran=1, got {loader_nos.ntran}"

    unfolded_kpts = np.asarray(sym.unfolded_kpts)
    nosym_kpts = np.asarray(loader_nos.kpoints)
    nos_match = build_match_table(unfolded_kpts, nosym_kpts)
    assert (nos_match >= 0).all(), f"unmatched full-BZ k: {np.where(nos_match<0)[0].tolist()}"
    print(f"matched {(nos_match >= 0).sum()}/{nk_full} full-BZ k's to nosym")

    # LORRAX full-BZ load (production unfold_psi inside)
    t_load = time.time()
    print("loading LORRAX ψ at full_bz (this invokes unfold_psi for every full-BZ k)...", flush=True)
    psi_lor = np.asarray(loader_sym.load(bands=(B_LO, B_HI), k='full_bz'))
    g_lor = np.asarray(loader_sym.gvecs(k='full_bz'))
    ngk_lor = np.asarray(loader_sym.ngk_valid(k='full_bz'))
    print(f"ψ_LORRAX shape={psi_lor.shape}  load={time.time()-t_load:.1f}s", flush=True)

    ngk_nos = np.asarray(loader_nos.ngk, dtype=np.int64)
    starts_nos = np.cumsum(np.concatenate([[0], ngk_nos[:-1]])).astype(np.int64)
    coeffs_nos = loader_nos._coeffs_raw
    gv_nos = loader_nos._gvecs_raw

    irr_idx_k = np.asarray(sym.irr_idx_k)
    sym_idx_k = np.asarray(sym.sym_idx_k)

    nb_win = B_HI - B_LO

    # Aggregate stats
    results = []
    global_max_unit_err = 0.0
    global_max_perband_defect = 0.0    # max over (n,k) of (1 - |overlap|)
    global_max_perband_raw = 0.0       # max over (n,k,G,α) of |Δψ| at optimal φ
    worst_perband = None               # (k, n, defect, raw_max)

    print(f"\n{'k_f':>4s} {'k_irr':>5s} {'sym':>3s} {'TRS':>3s} "
          f"{'ngrp':>4s} {'unit_err':>10s} {'pb_defect':>10s} {'pb_raw':>10s}")
    for kf in range(nk_full):
        kirr = int(irr_idx_k[kf])
        s_idx = int(sym_idx_k[kf])
        is_trs = s_idx >= ntran
        ngk_a = int(ngk_lor[kf])

        psi_a = psi_lor[kf, :nb_win, :, :ngk_a]            # (nb, ns, ngk)
        gv_a = g_lor[kf, :ngk_a]

        kn = int(nos_match[kf])
        ngk_c = int(ngk_nos[kn])
        start = int(starts_nos[kn])
        raw = coeffs_nos[B_LO:B_HI, :, start:start+ngk_c, :]
        psi_c = raw[..., 0] + 1j*raw[..., 1]                # (nb, ns, ngk)
        gv_c = gv_nos[start:start+ngk_c]

        box_a = scatter(psi_a, gv_a, ngk_a, fft_grid)        # (nb, ns, N)
        box_c = scatter(psi_c, gv_c, ngk_c, fft_grid)        # (nb, ns, N)

        e_at_k = np.asarray(loader_nos.energies[0, kn, B_LO:B_HI])
        groups = degenerate_groups(e_at_k, DEG_TOL)

        # Flatten (ns, N) → 1D for per-band inner products
        Xf = box_a.reshape(nb_win, -1)  # ψ_unfold
        Yf = box_c.reshape(nb_win, -1)  # ψ_nosym
        Xn = np.linalg.norm(Xf, axis=1)
        Yn = np.linalg.norm(Yf, axis=1)
        # Drop zero-norm bands (shouldn't happen in [0,32) on CrI3 — sanity assert)
        if (Xn < 1e-12).any() or (Yn < 1e-12).any():
            raise RuntimeError(f"zero-norm band at k_f={kf}: Xn.min={Xn.min()}, Yn.min={Yn.min()}")

        # Per-band overlap defect (1 - |<X|Y>|), strict spec metric
        ov = np.einsum('nG,nG->n', np.conj(Xf), Yf) / (Xn * Yn)  # (nb,) complex
        pb_defect = 1.0 - np.abs(ov)  # (nb,)
        pb_defect_max = float(np.max(pb_defect))
        pb_defect_argmax = int(np.argmax(pb_defect))

        # Per-band optimal-gauge raw residual: rotate Yf by phase e^{iφ_n} where
        # φ_n = arg(<X|Y>_n), and report max |X - e^{iφ}·Y|. Bands are normalized
        # so this is dimensionless wrt ψ amplitude.
        phi = np.angle(ov)
        Xn_norm = Xf / Xn[:, None]
        Yn_norm = Yf / Yn[:, None]
        diff = Xn_norm - np.exp(1j * phi)[:, None] * Yn_norm
        pb_raw = np.max(np.abs(diff), axis=1)  # (nb,)
        pb_raw_max = float(np.max(pb_raw))
        pb_raw_argmax = int(np.argmax(pb_raw))

        # Subspace unitarity check (degenerate-group robust)
        max_unit_err = 0.0
        for (lo, hi) in groups:
            if hi - lo < 2:
                continue
            Xg = Xn_norm[lo:hi]
            Yg = Yn_norm[lo:hi]
            U = Xg @ np.conj(Yg).T
            err = float(np.linalg.norm(U @ np.conj(U.T) - np.eye(hi - lo), ord=np.inf))
            if err > max_unit_err:
                max_unit_err = err

        # Global aggregation
        if max_unit_err > global_max_unit_err:
            global_max_unit_err = max_unit_err
        if pb_defect_max > global_max_perband_defect:
            global_max_perband_defect = pb_defect_max
            worst_perband = dict(
                k_full=kf, k_irr=kirr, sym_idx=s_idx, is_trs=is_trs,
                band=pb_defect_argmax, defect=pb_defect_max,
                raw_at_band=float(pb_raw[pb_defect_argmax]),
                ov_re=float(ov[pb_defect_argmax].real),
                ov_im=float(ov[pb_defect_argmax].imag),
            )
        if pb_raw_max > global_max_perband_raw:
            global_max_perband_raw = pb_raw_max

        results.append(dict(
            k_full=kf, k_irr=kirr, sym_idx=s_idx, is_trs=is_trs,
            n_groups=len(groups),
            max_unit_err=max_unit_err,
            max_perband_defect=pb_defect_max,
            max_perband_raw=pb_raw_max,
            argmax_band_defect=pb_defect_argmax,
            argmax_band_raw=pb_raw_argmax,
        ))
        print(f"{kf:4d} {kirr:5d} {s_idx:3d} {'T' if is_trs else 'F':>3s} "
              f"{len(groups):4d} {max_unit_err:10.2e} "
              f"{pb_defect_max:10.2e} {pb_raw_max:10.2e}", flush=True)

    # Per-sym-idx aggregation
    by_sym = {}
    for r in results:
        by_sym.setdefault(r["sym_idx"], []).append(r)
    print(f"\nPer sym_idx aggregate:")
    print(f"{'sym':>3s} {'TRS':>3s} {'n_k':>3s} {'max_unit':>10s} "
          f"{'max_pb_def':>10s} {'max_pb_raw':>10s}")
    per_sym_stats = []
    for s_idx in sorted(by_sym.keys()):
        rows = by_sym[s_idx]
        mu = max(r["max_unit_err"] for r in rows)
        mpd = max(r["max_perband_defect"] for r in rows)
        mpr = max(r["max_perband_raw"] for r in rows)
        is_trs = s_idx >= ntran
        per_sym_stats.append(dict(sym_idx=s_idx, is_trs=is_trs, n_k=len(rows),
                                   max_unit=mu, max_perband_defect=mpd,
                                   max_perband_raw=mpr))
        print(f"{s_idx:3d} {'T' if is_trs else 'F':>3s} {len(rows):3d} "
              f"{mu:10.2e} {mpd:10.2e} {mpr:10.2e}")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"\n=== GLOBAL ===")
    print(f"  max subspace unit_err     : {global_max_unit_err:.3e}")
    print(f"  max per-band overlap defect (1-|<ψ_unfold|ψ_nosym>|): {global_max_perband_defect:.3e}")
    print(f"  max per-band raw |Δψ| at optimal gauge phase        : {global_max_perband_raw:.3e}")
    if worst_perband is not None:
        print(f"  worst per-band tuple: {worst_perband}")

    out_json = os.path.join(out_dir, "CrI3_stage0_results.json")
    with open(out_json, "w") as f:
        json.dump(dict(
            system="CrI3_6x6_30Ry_SOC_bispinor",
            sym_wfn=SYM_WFN, nosym_wfn=NOSYM_WFN,
            ntran=ntran, nk_full=nk_full, nspinor=ns,
            fft_grid=list(fft_grid),
            band_window=[B_LO, B_HI], deg_tol=DEG_TOL,
            global_max_unit_err=global_max_unit_err,
            global_max_perband_defect=global_max_perband_defect,
            global_max_perband_raw=global_max_perband_raw,
            worst_perband=worst_perband,
            results=results,
            per_sym_stats=per_sym_stats,
            elapsed_s=elapsed,
        ), f, indent=2)
    print(f"wrote {out_json}")
    loader_sym.close()
    loader_nos.close()


if __name__ == "__main__":
    main()
