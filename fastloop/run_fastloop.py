#!/usr/bin/env python3
"""fastloop runner — the full L1 driver chain on the checked-in mini-deck.

Chain (each driver a fresh subprocess, exactly like the certified b300
harnesses): kmeans -> dipole -> kin-ion -> gw -> eqp-convert ->
htransform(dft) -> htransform(qp).

Two legs:
  p1      single process, single device (JAX_PROCESS_COUNT=1, no XLA_FLAGS)
  shard4  single process, 4 HOST devices via
          XLA_FLAGS=--xla_force_host_platform_device_count=4 — every driver
          resolves a 2x2 ('x','y') mesh through common.collectives
          .resolve_mesh, so sharding semantics are exercised with no MPI
          and no multi-node job (the repo tools/probe_w_densifier_hlo.py
          trick, scaled to the whole chain).

Modes:
  check (default)  run leg(s), compare every stage output against
                   fastloop/reference/ + fastloop/pins_mini.json,
                   exit nonzero on ANY drift or stage failure.
  pin              run the p1 leg, WRITE reference/ + pins_mini.json.
                   Refuses if pins exist unless --allow-repin.

Runs IN-CONTAINER on a compute node only (the login node cannot import
jax: glibc 2.17 vs the wheels' 2.28 — repo docs/environment/overview.md
layer 2; verified 2026-07-31: the venv interpreter is
/usr/local/bin/python3.12, container-internal). Standard form is
fastloop/run_fastloop.sbatch; it also runs inside any existing
allocation with `bash fastloop/run_fastloop.sbatch`.

Exit codes: 0 pass | 1 parity drift | 2 stage rc!=0/timeout | 3 refusal.

The mini-deck proves plumbing and sharding semantics, not physics:
no physics conclusion may cite a number from this deck (fastloop/PLAN.md).
"""
import argparse
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

FASTLOOP = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(FASTLOOP)

DEFAULT_DECK = os.path.join(FASTLOOP, "deck_mini")
DEFAULT_PINS = os.path.join(FASTLOOP, "pins_mini.json")
DEFAULT_REF = os.path.join(FASTLOOP, "reference")
DEFAULT_WORK = os.path.join(FASTLOOP, "work")
DEFAULT_SRC = os.environ.get(
    "LORRAX_SRC", "/work2/08271/jackmc/frontera/lorrax/src")
DEFAULT_FFI_SO = os.environ.get(
    "LORRAX_FFI_HOST_SO",
    "/work2/08271/jackmc/frontera/lorrax_ffi_unified/build_host_ONE/"
    "liblorrax_ffi_host.so")
MAKE_EQP = os.path.join(
    SANDBOX, "runs", "mos2_4x4_b300", "make_eqp_htformat.py")

# host FFI .so deps (post_b300.sbatch's certified list, libfabric included)
LDLP_PREFIX = ":".join([
    "/opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/libfabric/lib",
    "/opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/lib",
    "/opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/lib/release",
    "/work2/08271/jackmc/frontera/slate_builds/cpu/install/lib64",
    "/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin",
    "/home1/apps/intel19/impi19_0/phdf5/1.14.6/lib",
    "/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin",
])

KMEANS_LADDER = (400, 320, 272)   # pin mode: first N passing the rank gate
DAT_FILES = ("eqp0.dat", "eqp1.dat", "eqp_g0w0.dat", "sigma_diag.dat")
H5_FILES = ("dipole.h5", "kin_ion.h5", "sigma_mnk.h5")
BS_FILES = (("ht_dft", "bandstructure_dft.dat"),
            ("ht_qp", "bandstructure_qp.dat"))
H5_SAMPLE_CAP = 65536  # elements/dataset in the .sig.npz point-wise sample

DEFAULT_TOLS = {
    # 1e-8 eV: the compare_valsmoke.py precedent — 5-6 orders above the
    # measured FFI-swap noise floor (2.5e-14 eV), 5 orders below physical
    # significance. Recalibrated from measured deltas at pin time.
    "dat_tol_ev": 1.0e-8,
    "h5_tol": 1.0e-8,
    "scalar_tol_ev": 1.0e-6,
    "coord_tol": 1.0e-9,       # centroid fractional coords (grid-snapped)
    "shard4_tol_scale": 10.0,  # shard leg tolerance = base * scale
}

STAGE_TIMEOUT = {"kmeans": 900, "dipole": 900, "kin_ion": 1200,
                 "gw": 1200, "eqp_convert": 300, "ht_dft": 1200,
                 "ht_qp": 1200}

# The gw stage runs bare (python -m gw.gw_jax), like every other stage.
# A GW_WRAPPER os._exit workaround lived here 2026-07-31..08-01 for the
# job-7884928 interpreter-teardown hang (CLAIMS 19); it was removed once
# the hang was closed repo-side — check mode exiting 0 without it is the
# acceptance test for that fix.


def log(msg):
    print("[fastloop] %s" % msg, flush=True)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def stage_env(threads, src, ffi_so, shard4, extra_xla="", cache_cold=False):
    env = dict(os.environ)
    env.update({
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "CUDA_VISIBLE_DEVICES": "",
        # BOTH belts (the job-7884642 lesson, deck_b300.sbatch header):
        # never let SLURM_NTASKS talk a driver into jax.distributed.
        "JAX_PROCESS_COUNT": "1",
        "JAX_PROCESS_INDEX": "0",
        "HDF5_USE_FILE_LOCKING": "FALSE",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": src,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    })
    # certified opt-in perf stack (GATES.md); overridable from outside
    env.setdefault("LORRAX_FFT_FFI", "1")
    env.setdefault("LORRAX_FFT_FFI_FUSED", "1")
    if ffi_so and os.path.exists(ffi_so):
        env["LORRAX_FFI_HOST_SO"] = ffi_so
    env["LD_LIBRARY_PATH"] = LDLP_PREFIX + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    xla = []
    if shard4:
        xla.append("--xla_force_host_platform_device_count=4")
    if extra_xla:
        xla.append(extra_xla)
    if xla:
        env["XLA_FLAGS"] = " ".join(xla)
    else:
        env.pop("XLA_FLAGS", None)
    if cache_cold:
        # collective/HLO evidence is only valid cache-cold (CLAIMS row 5)
        env["ISDF_JAX_CACHE_DIR"] = ""
    # else: leave the persistent compile cache at its default (resolves
    # under $SCRATCH; outputs measured byte-identical warm, jobs
    # 7884869/7884871) — the warm loop is the fast loop.
    return env


# ---------------------------------------------------------------------------
# leg setup + stage execution
# ---------------------------------------------------------------------------

def _link(target, name):
    if os.path.lexists(name):
        os.remove(name)
    os.symlink(target, name)


def setup_leg(leg_dir, deck):
    os.makedirs(os.path.join(leg_dir, "kmeans"), exist_ok=True)
    os.makedirs(os.path.join(leg_dir, "logs"), exist_ok=True)
    links = {
        "WFN.h5": os.path.join(deck, "WFN_mini.h5"),
        "RHO": os.path.join(deck, "RHO_mini"),
        "vxc.dat": os.path.join(deck, "vxc_mini.dat"),
        "kih.dat": os.path.join(deck, "kih_mini.dat"),
        "out": os.path.join(deck, "mini_out"),
        "Mo.upf": os.path.join(deck, "Mo.upf"),
        "S.upf": os.path.join(deck, "S.upf"),
    }
    for name, target in links.items():
        _link(target, os.path.join(leg_dir, name))
    for name in ("WFN.h5", "Mo.upf", "S.upf"):
        _link(links[name], os.path.join(leg_dir, "kmeans", name))
    deck_in = open(os.path.join(deck, "deck_mini.in")).read()
    with open(os.path.join(leg_dir, "deck.in"), "w") as fh:
        fh.write(deck_in)
    for tag in ("ht_dft", "ht_qp"):
        d = os.path.join(leg_dir, tag)
        os.makedirs(d, exist_ok=True)
        _link(links["WFN.h5"], os.path.join(d, "WFN.h5"))
        with open(os.path.join(d, "ht.in"), "w") as fh:
            fh.write(deck_in.replace(
                "centroids_file = centroids_mini.txt",
                "centroids_file = ../centroids_mini.txt"))
    _link(os.path.join("..", "eqp_ht.dat"),
          os.path.join(leg_dir, "ht_qp", "eqp_ht.dat"))


def run_stage(name, cmd, cwd, env, leg_dir):
    logf = os.path.join(leg_dir, "logs", name + ".log")
    t0 = time.time()
    try:
        with open(logf, "w") as fh:
            rc = subprocess.run(
                cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                timeout=STAGE_TIMEOUT.get(name, 1800)).returncode
    except subprocess.TimeoutExpired:
        rc = 124
    wall = time.time() - t0
    log("stage %-12s rc=%-3d wall=%6.1fs" % (name, rc, wall))
    if rc != 0:
        try:
            tail = open(logf, errors="replace").readlines()[-25:]
            sys.stdout.write("".join("    | " + ln for ln in tail))
        except OSError:
            pass
    return rc, wall


def deck_nband(leg_dir):
    txt = open(os.path.join(leg_dir, "deck.in")).read()
    m = re.search(r"^nband\s*=\s*(\d+)", txt, re.M)
    if not m:
        raise SystemExit("REFUSING: no nband key in deck.in")
    return int(m.group(1))


def run_leg(leg_dir, deck, env, ladder):
    """Run the full chain in leg_dir. Returns (results dict, ok bool)."""
    res = {"stages": {}, "kmeans": {}}
    py = sys.executable

    # --- kmeans (hard-reads WFN.h5 from cwd: own subdir, b1024 pattern) ----
    kdir = os.path.join(leg_dir, "kmeans")
    accepted = None
    for n_req in ladder:
        rc, wall = run_stage(
            "kmeans",
            [py, "-u", "-m", "centroid.kmeans_cli", str(n_req), "--orbit",
             "--qe-save", os.path.join(deck, "mini_out", "MoS2.save"),
             "--out-suffix", "_mini_c%d" % n_req],
            kdir, env, leg_dir)
        res["stages"]["kmeans"] = {"rc": rc, "wall": wall}
        files = sorted(glob.glob(
            os.path.join(kdir, "centroids_frac_*_mini_c%d.txt" % n_req)))
        if rc == 0 and files:
            accepted = (n_req, files[-1])
            break
        log("kmeans N_c=%d rejected (rc=%d) — stepping down the ladder"
            % (n_req, rc))
    if accepted is None:
        return res, False
    n_req, cfile = accepted
    shutil.copy2(cfile, os.path.join(leg_dir, "centroids_mini.txt"))
    klog = open(os.path.join(leg_dir, "logs", "kmeans.log"),
                errors="replace").read()
    m = re.search(r"After pruning: (\d+) centroids \(rank=(\d+)\)", klog)
    res["kmeans"] = {
        "n_request": n_req,
        "n_centroids": sum(1 for ln in open(cfile)
                           if not ln.startswith("#")),
        "rank": int(m.group(2)) if m else -1,
        "gate_pass": bool(re.search(r"\[rank gate\].*PASS", klog)),
    }
    log("kmeans accepted: N_req=%d rows=%d rank=%d gate_pass=%s"
        % (n_req, res["kmeans"]["n_centroids"], res["kmeans"]["rank"],
           res["kmeans"]["gate_pass"]))
    if not res["kmeans"]["gate_pass"]:
        return res, False

    # --- dipole / kin-ion / gw (flat leg dir, deck_b300 pattern) -----------
    nb = deck_nband(leg_dir)
    plan = [
        ("dipole", [py, "-u", "-m", "psp.get_dipole_mtxels",
                    "-i", "deck.in", "--out", "dipole.h5"], leg_dir),
        ("kin_ion", [py, "-u", "-m", "gw.kin_ion_io", "-i", "deck.in",
                     "-o", "kin_ion.h5", "-n", str(nb), "--hartree"],
         leg_dir),
        ("gw", [py, "-u", "-m", "gw.gw_jax", "-i", "deck.in"], leg_dir),
        ("eqp_convert", [py, "-u", MAKE_EQP, leg_dir,
                         os.path.join(leg_dir, "eqp_ht.dat")], leg_dir),
        ("ht_dft", [py, "-u", "-m", "bandstructure.htransform",
                    "-i", "ht.in", "--verbose"],
         os.path.join(leg_dir, "ht_dft")),
        ("ht_qp", [py, "-u", "-m", "bandstructure.htransform",
                   "-i", "ht.in", "--verbose", "--eqp-file", "eqp_ht.dat"],
         os.path.join(leg_dir, "ht_qp")),
    ]
    for name, cmd, cwd in plan:
        rc, wall = run_stage(name, cmd, cwd, env, leg_dir)
        res["stages"][name] = {"rc": rc, "wall": wall}
        if rc != 0:
            return res, False

    # EQP override must be TAKEN, not swallowed (post_b300 90/91 gate)
    qlog = open(os.path.join(leg_dir, "logs", "ht_qp.log"),
                errors="replace").read()
    if "EQP override skipped" in qlog or "Using EQP energies" not in qlog:
        log("FAIL: EQP override not proven in ht_qp log — the 'QP' curve "
            "would be DFT")
        res["stages"]["ht_qp"]["rc"] = 90
        return res, False
    return res, True


# ---------------------------------------------------------------------------
# parity: numeric loaders, h5 signatures, comparisons
# ---------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+|nan|inf)(?:[eEdD][-+]?\d+)?",
                  re.IGNORECASE)


def load_numbers(path):
    """compare_valsmoke.py loader: rows of floats from .dat text
    (key=value tokens included; nan/inf tokens preserved)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            vals = []
            for t in _NUM.findall(line):
                vals.append(float(t.replace("D", "e").replace("d", "e")))
            if vals:
                rows.append(vals)
    return rows


def max_delta_rows(a, b):
    """max |a-b| with nan==nan counting as 0; shape mismatch -> None."""
    import math
    if len(a) != len(b):
        return None
    worst = 0.0
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return None
        for x, y in zip(ra, rb):
            if math.isnan(x) and math.isnan(y):
                continue
            if math.isnan(x) != math.isnan(y):
                return float("inf")
            worst = max(worst, abs(x - y))
    return worst


def _h5_sig_arrays(path):
    """{dataset name: complex128 sample (deterministic stride)} +
    {name: (shape, max_abs)} for every numeric dataset (names containing
    'time' skipped — walltime-style provenance)."""
    import h5py
    import numpy as np
    samples, meta = {}, {}
    with h5py.File(path, "r") as f:
        names = []

        def _collect(n, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(n)
        f.visititems(_collect)
        for n in names:
            d = f[n]
            if d.size == 0 or not np.issubdtype(d.dtype, np.number):
                continue
            if "time" in n.lower():
                continue
            arr = np.asarray(d[...], dtype=np.complex128).ravel()
            stride = max(1, arr.size // H5_SAMPLE_CAP)
            samples[n] = arr[::stride][:H5_SAMPLE_CAP]
            mags = np.abs(arr)
            mags = mags[~np.isnan(mags)]  # nans compared via the samples
            meta[n] = (list(d.shape),
                       float(mags.max()) if mags.size else 0.0)
    return samples, meta


def h5_sig_write(path, ref_dir, base):
    import numpy as np
    samples, meta = _h5_sig_arrays(path)
    np.savez_compressed(
        os.path.join(ref_dir, base + ".sig.npz"),
        **{n.replace("/", "__SL__"): s for n, s in samples.items()})
    with open(os.path.join(ref_dir, base + ".sig.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)


def h5_sig_compare(path, ref_dir, base):
    """-> (max_delta, detail str). inf on shape/name mismatch."""
    import numpy as np
    try:
        samples, meta = _h5_sig_arrays(path)
    except OSError as e:
        return float("inf"), "unreadable: %r" % (e,)
    ref = np.load(os.path.join(ref_dir, base + ".sig.npz"))
    ref_meta = json.load(open(os.path.join(ref_dir, base + ".sig.json")))
    if sorted(meta) != sorted(ref_meta):
        return float("inf"), "dataset set changed (%d vs %d)" % (
            len(meta), len(ref_meta))
    worst, where = 0.0, "-"
    for n in meta:
        if list(meta[n][0]) != list(ref_meta[n][0]):
            return float("inf"), "%s shape %s vs %s" % (
                n, meta[n][0], ref_meta[n][0])
        r = ref[n.replace("/", "__SL__")]
        s = samples[n]
        a, b = np.nan_to_num(s, nan=1e300), np.nan_to_num(r, nan=1e300)
        d = float(np.max(np.abs(a - b))) if s.size else 0.0
        d = max(d, abs(meta[n][1] - ref_meta[n][1]))
        if d > worst:
            worst, where = d, n
    return worst, where


def bs_scalars(path, nval):
    """(E_vbm, E_cbm, gap) from a bandstructure.dat (idx_k idx_b ... energy;
    energies in the file's own unit — pins are unit-agnostic)."""
    vb, cb = nval - 1, nval
    e_v, e_c = [], []
    for ln in open(path):
        if ln.lstrip().startswith("#"):
            continue
        p = ln.split()
        if len(p) < 7:
            continue
        b, e = int(p[1]), float(p[6])
        if b == vb:
            e_v.append(e)
        elif b == cb:
            e_c.append(e)
    if not e_v or not e_c:
        return None
    return max(e_v), min(e_c), min(e_c) - max(e_v)


def deck_nval(deck):
    txt = open(os.path.join(deck, "deck_mini.in")).read()
    return int(re.search(r"^nval\s*=\s*(\d+)", txt, re.M).group(1))


class Parity(object):
    def __init__(self):
        self.rows = []
        self.failed = False

    def add(self, leg, item, delta, tol, note=""):
        ok = (delta is not None) and (delta <= tol)
        self.rows.append((leg, item, delta, tol, ok, note))
        if not ok:
            self.failed = True

    def report(self):
        print("\n[fastloop] ---- parity report ----")
        for leg, item, delta, tol, ok, note in self.rows:
            ds = ("%.3e" % delta) if delta is not None else "SHAPE-MISMATCH"
            print("[fastloop] %-7s %-28s max|d|=%-14s tol=%-8.1e %s %s"
                  % (leg, item, ds, tol, "OK  " if ok else "FAIL", note))
        print("[fastloop] parity verdict: %s"
              % ("FAIL" if self.failed else "PASS"))


def compare_leg(leg, leg_dir, res, pins, ref_dir, deck, par):
    scale = pins["shard4_tol_scale"] if leg == "shard4" else 1.0
    km, pkm = res.get("kmeans", {}), pins["kmeans"]
    for key in ("n_request", "n_centroids", "rank"):
        d = abs(km.get(key, -10**9) - pkm[key])
        par.add(leg, "kmeans." + key, float(d), 0.0)
    d = max_delta_rows(
        load_numbers(os.path.join(leg_dir, "centroids_mini.txt")),
        load_numbers(os.path.join(ref_dir, "centroids_mini.txt")))
    par.add(leg, "centroids_mini.txt", d, pins["coord_tol"] * scale)
    for f in DAT_FILES:
        try:
            d = max_delta_rows(load_numbers(os.path.join(leg_dir, f)),
                               load_numbers(os.path.join(ref_dir, f)))
        except OSError:
            d = None
        par.add(leg, f, d, pins["dat_tol_ev"] * scale)
    for f in H5_FILES:
        d, where = h5_sig_compare(os.path.join(leg_dir, f), ref_dir, f)
        par.add(leg, f, d, pins["h5_tol"] * scale, note=where)
    nval = deck_nval(deck)
    for sub, refname in BS_FILES:
        try:
            d = max_delta_rows(
                load_numbers(os.path.join(leg_dir, sub, "bandstructure.dat")),
                load_numbers(os.path.join(ref_dir, refname)))
        except OSError:
            d = None
        par.add(leg, refname, d, pins["dat_tol_ev"] * scale)
        s = bs_scalars(os.path.join(leg_dir, sub, "bandstructure.dat"), nval)
        pin = pins["scalars"][refname]
        d = (max(abs(a - b) for a, b in zip(s, pin))
             if s is not None else None)
        par.add(leg, refname + ".vbm/cbm/gap", d,
                pins["scalar_tol_ev"] * scale)


def write_pins(leg_dir, res, deck, ref_dir, pins_path, walls):
    os.makedirs(ref_dir, exist_ok=True)
    shutil.copy2(os.path.join(leg_dir, "centroids_mini.txt"), ref_dir)
    for f in DAT_FILES + ("eqp_ht.dat",):
        shutil.copy2(os.path.join(leg_dir, f), ref_dir)
    for sub, refname in BS_FILES:
        shutil.copy2(os.path.join(leg_dir, sub, "bandstructure.dat"),
                     os.path.join(ref_dir, refname))
    for f in H5_FILES:
        h5_sig_write(os.path.join(leg_dir, f), ref_dir, f)
    nval = deck_nval(deck)
    scalars = {}
    for sub, refname in BS_FILES:
        scalars[refname] = list(
            bs_scalars(os.path.join(leg_dir, sub, "bandstructure.dat"),
                       nval))
    pins = dict(DEFAULT_TOLS)
    pins.update({
        "kmeans": res["kmeans"],
        "scalars": scalars,
        "chain": [s for s in ("kmeans", "dipole", "kin_ion", "gw",
                              "eqp_convert", "ht_dft", "ht_qp")],
        "pinned_by": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "jobid": os.environ.get("SLURM_JOB_ID", "interactive"),
            "host": socket.gethostname(),
            "src_commit": os.environ.get("LORRAX_SRC_COMMIT", "unknown"),
            "walls_s": walls,
        },
    })
    with open(pins_path, "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
    log("pins written: %s" % pins_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("check", "pin"), default="check")
    ap.add_argument("--legs", default=None,
                    help="comma list from {p1,shard4}; default: "
                         "check=p1,shard4  pin=p1")
    ap.add_argument("--deck", default=DEFAULT_DECK)
    ap.add_argument("--pins", default=DEFAULT_PINS)
    ap.add_argument("--reference", default=DEFAULT_REF)
    ap.add_argument("--work", default=None,
                    help="work dir (default fastloop/work/<mode>.<jobid>)")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="LORRAX src tree for PYTHONPATH (the LIVE tree by "
                         "design: this is a pre-commit semantic check of "
                         "the working tree, unlike production jobs which "
                         "read frozen bundles)")
    ap.add_argument("--ffi-so", default=DEFAULT_FFI_SO)
    ap.add_argument("--threads", type=int, default=int(os.environ.get(
        "FASTLOOP_THREADS", min(56, os.cpu_count() or 28))))
    ap.add_argument("--allow-repin", action="store_true")
    ap.add_argument("--hlo-diag", action="store_true",
                    help="shard4 gw stage: cache-cold + --xla_dump_to, then "
                         "run tools/hlo/analyze_hlo_dump.py (DIAGNOSTIC "
                         "ONLY — known-open gathers exist, CLAIMS row 16, "
                         "so this never fails the run)")
    args = ap.parse_args()

    if socket.gethostname().startswith("login"):
        log("REFUSING: login node cannot import jax (glibc; "
            "docs/environment/overview.md layer 2). Use "
            "fastloop/run_fastloop.sbatch.")
        return 3
    for path, what in ((args.deck, "deck"), (args.src, "src"),
                       (MAKE_EQP, "make_eqp_htformat.py")):
        if not os.path.exists(path):
            log("REFUSING: missing %s: %s" % (what, path))
            return 3
    if not os.path.exists(os.path.join(args.deck, "WFN_mini.h5")):
        log("REFUSING: deck has no WFN_mini.h5 — run "
            "fastloop/build_minideck.sbatch first")
        return 3

    legs = (args.legs.split(",") if args.legs
            else (["p1"] if args.mode == "pin" else ["p1", "shard4"]))
    jobid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    work = args.work or os.path.join(
        DEFAULT_WORK, "%s.%s" % (args.mode, jobid))
    log("mode=%s legs=%s work=%s threads=%d src=%s"
        % (args.mode, ",".join(legs), work, args.threads, args.src))

    if args.mode == "pin":
        if os.path.exists(args.pins) and not args.allow_repin:
            log("REFUSING: %s exists — repinning erases the certified "
                "baseline; pass --allow-repin if that is intended"
                % args.pins)
            return 3

    pins = None
    if args.mode == "check":
        if not os.path.exists(args.pins):
            log("REFUSING: no pins at %s — run --mode pin first" % args.pins)
            return 3
        pins = json.load(open(args.pins))

    par = Parity()
    t_all = time.time()
    stage_fail = False
    for leg in legs:
        leg_dir = os.path.join(work, leg)
        setup_leg(leg_dir, os.path.abspath(args.deck))
        env = stage_env(args.threads, args.src, args.ffi_so,
                        shard4=(leg == "shard4"))
        ladder = (KMEANS_LADDER if pins is None
                  else (pins["kmeans"]["n_request"],))
        log("---- leg %s ----" % leg)
        t0 = time.time()
        res, ok = run_leg(leg_dir, os.path.abspath(args.deck), env, ladder)
        wall = time.time() - t0
        log("leg %s: %s in %.1fs" % (leg, "ok" if ok else "STAGE FAILURE",
                                     wall))
        if not ok:
            stage_fail = True
            continue
        if args.mode == "pin":
            walls = {k: round(v["wall"], 1)
                     for k, v in res["stages"].items()}
            walls["leg_total"] = round(wall, 1)
            write_pins(leg_dir, res, os.path.abspath(args.deck),
                       args.reference, args.pins, walls)
        else:
            compare_leg(leg, leg_dir, res, pins, args.reference,
                        os.path.abspath(args.deck), par)
        if leg == "shard4" and args.hlo_diag:
            _hlo_diag(leg_dir, args)

    log("total wall %.1fs (target: each leg's chain under ~90 s warm)"
        % (time.time() - t_all))
    if args.mode == "check":
        par.report()
    if stage_fail:
        return 2
    if args.mode == "check" and par.failed:
        return 1
    return 0


def _hlo_diag(leg_dir, args):
    """Re-run ONLY gw with cache-cold HLO dump on 4 host devices, then
    summarize with the sandbox analyzer. Diagnostic: never fails the run."""
    dump = os.path.join(leg_dir, "hlo_dump")
    env = stage_env(args.threads, args.src, args.ffi_so, shard4=True,
                    extra_xla="--xla_dump_to=%s" % dump, cache_cold=True)
    rc, _ = run_stage("gw_hlo", [sys.executable, "-u", "-m", "gw.gw_jax",
                                 "-i", "deck.in"],
                      leg_dir, env, leg_dir)
    if rc != 0:
        log("hlo-diag: gw re-run failed rc=%d (diagnostic only)" % rc)
        return
    ana = os.path.join(SANDBOX, "tools", "hlo", "analyze_hlo_dump.py")
    rc = subprocess.run([sys.executable, ana, dump, "--top", "10"]).returncode
    log("hlo-diag: analyzer rc=%d -> %s/hlo_summary.md (tables valid: this "
        "dump was cache-cold)" % (rc, dump))


if __name__ == "__main__":
    raise SystemExit(main())
