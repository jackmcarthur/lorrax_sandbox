"""Re-emit a LORRAX GW run's eqp1 energies in the format htransform parses.

WHY THIS EXISTS (harness-side only; no src is touched).

``bandstructure.htransform --eqp-file`` calls ``read_eqp_energies``, which
needs BOTH of:

  * ``k-point <K>:`` block headers, one per FULL-BZ k-point, and
    ``nk == sym.nk_tot`` is enforced (16 here, not the 10-point IBZ);
  * a per-band line matching ``n=<B>`` and ``EQP=<value>``.

No file a current LORRAX GW run writes satisfies both:

  * ``eqp0.dat`` / ``eqp1.dat``  — BGW column format, and only the 10 IBZ
    k-points.  ``read_eqp_energies`` finds no ``k-point`` header at all and
    raises "No k-point blocks found".
  * ``eqp_g0w0.dat``            — full BZ and ``k-point <K>:`` headers, but
    its value tokens are ``E_DFT=`` / ``Re=`` / ``Im=``; neither ``EQP=``
    nor ``sigX=`` appears, so every band parses as missing.
  * ``sigma_diag.dat``          — full BZ and DOES carry ``sigX=``, so it
    parses, but the numbers it returns are bare exchange self-energies,
    not band energies.

``initialize_wfns`` SWALLOWS the resulting exception and logs
"EQP override skipped", so a run given ``--eqp-file eqp1.dat`` silently
produces the DFT bandstructure.  That is the failure this script avoids.

WHAT IT DOES

  1. reads ``eqp_g0w0.dat``  -> full-BZ order, E_DFT[k_full, n], Eqp0[k_full, n]
  2. reads ``eqp0.dat`` / ``eqp1.dat`` -> IBZ E_DFT, Eqp0, Eqp1
  3. matches each full-BZ k to its IBZ parent on BOTH the E_DFT and the
     Eqp0 vectors (a doubly-constrained match; refuses on ambiguity)
  4. writes ``k-point K:`` / ``n=B  EQP=<Ry>`` for the Z-linearised eqp1.

UNITS.  The eqp/sigma files are eV; ``get_enk_bandrange`` (the DFT path
this override replaces) returns ``wfn.energies``, which are Ry, and
htransform's own bandwidth gate is phrased in Ry.  This writer therefore
emits Ry, so the override and the DFT path are on one scale.

Usage:
    python make_eqp_htformat.py RUN_DIR OUT_FILE
"""
import sys

RYD_TO_EV = 13.605693122994
MATCH_MAX_EV = 1.0e-3      # best-candidate residual must be below this
MATCH_SEP = 20.0           # runner-up must be this many times worse


def read_g0w0(path):
    """-> (list of {n: (E_DFT, Re)}) indexed by full-BZ k."""
    blocks, cur = [], None
    for raw in open(path, encoding="utf8"):
        line = raw.strip()
        if line.startswith("k-point"):
            cur = {}
            blocks.append(cur)
            continue
        if cur is None or not line.startswith("n="):
            continue
        toks = line.replace("=", "= ").split()
        d = {}
        for i, t in enumerate(toks):
            if t.endswith("="):
                d[t[:-1]] = toks[i + 1]
        cur[int(d["n"])] = (float(d["E_DFT"]), float(d["Re"]))
    return blocks


def read_bgw(path):
    """BGW-format eqp file -> (list of k-labels, list of {n0: (E_DFT, E_QP)})."""
    ks, blocks, cur = [], [], None
    for raw in open(path, encoding="utf8"):
        if raw.lstrip().startswith("#"):
            continue
        f = raw.split()
        if len(f) == 4 and "." in f[0]:
            cur = {}
            ks.append(tuple(f[:3]))
            blocks.append(cur)
        elif len(f) == 4 and cur is not None:
            cur[int(f[1]) - 1] = (float(f[2]), float(f[3]))   # 1-based -> 0-based
    return ks, blocks


def main():
    run_dir, out_path = sys.argv[1], sys.argv[2]
    g = read_g0w0(f"{run_dir}/eqp_g0w0.dat")
    _, e0 = read_bgw(f"{run_dir}/eqp0.dat")
    _, e1 = read_bgw(f"{run_dir}/eqp1.dat")
    nk_full, nk_irr = len(g), len(e0)
    bands = sorted(g[0])
    print(f"full-BZ k blocks (eqp_g0w0.dat): {nk_full}")
    print(f"IBZ    k blocks (eqp0/eqp1.dat): {nk_irr} / {len(e1)}")
    print(f"bands per block: {len(bands)}  (n={bands[0]}..{bands[-1]})")
    if len(e1) != nk_irr:
        raise SystemExit("eqp0.dat and eqp1.dat disagree on k-point count")

    # full-BZ k -> IBZ parent: nearest on (E_DFT, Eqp0) jointly, with a
    # SEPARATION gate.  A fixed absolute tolerance is not usable here --
    # eqp_g0w0.dat prints 6 decimals and its Re column reaches ~1e-5 eV
    # away from eqp0.dat's own value -- but the correct parent is always
    # ~3 orders of magnitude closer than the runner-up (a k-star shares
    # its energies exactly), so require exactly that and refuse otherwise.
    parent = []
    for ik, blk in enumerate(g):
        scored = []
        for jk in range(nk_irr):
            d = max(max(abs(blk[n][0] - e0[jk][n][0]),
                        abs(blk[n][1] - e0[jk][n][1]))
                    for n in bands if n in e0[jk])
            scored.append((d, jk))
        scored.sort()
        best, runner = scored[0], scored[1]
        if best[0] > MATCH_MAX_EV or runner[0] < MATCH_SEP * max(best[0], 1e-12):
            raise SystemExit(
                f"full-BZ k {ik}: ambiguous IBZ parent — best {best[1]} at "
                f"{best[0]:.3e} eV, runner-up {runner[1]} at {runner[0]:.3e} eV "
                f"(need best < {MATCH_MAX_EV} eV and a {MATCH_SEP}x gap). "
                f"Refusing to guess.")
        parent.append(best[1])
        if ik < 4 or ik == nk_full - 1:
            print(f"  full k {ik} -> IBZ {best[1]}  (resid {best[0]:.2e} eV, "
                  f"runner-up {runner[0]:.2e} eV)")
    print(f"full-BZ -> IBZ map: {parent}")

    dmax = 0.0
    with open(out_path, "w", encoding="utf8") as fh:
        fh.write("# htransform --eqp-file input, generated by "
                 "make_eqp_htformat.py\n")
        fh.write("# values are Z-linearised G0W0 quasiparticle energies "
                 "(eqp1), in RYDBERG\n")
        for ik, jk in enumerate(parent):
            fh.write(f"k-point {ik}:\n")
            for n in bands:
                if n not in e1[jk]:
                    raise SystemExit(f"eqp1.dat lacks band {n} at IBZ k {jk}")
                ev = e1[jk][n][1]
                dmax = max(dmax, abs(ev - g[ik][n][1]))
                fh.write(f"n={n}  EQP= {ev / RYD_TO_EV:.12f}\n")
    print(f"max |eqp1 - eqp0| over the written window: {dmax:.6f} eV")
    print(f"WROTE {out_path}  ({nk_full} k-blocks x {len(bands)} bands, Ry)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
