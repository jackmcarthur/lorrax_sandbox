"""Doctored-stamp reader matrix against the four ζ this job just wrote."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ['LORRAX_FROZEN_SRC'])
from file_io.isdf_header import read_isdf_header
from gw import gw_init

TMP = sys.argv[1]
PATHS = {0: os.path.join(TMP, 'zeta_q.h5')}
PATHS.update({L: os.path.join(TMP, 'zeta_q_mu%d.h5' % L) for L in (1, 2, 3)})

fails = []


def check(name, cond):
    print(('  PASS ' if cond else '  FAIL ') + name)
    if not cond:
        fails.append(name)


print("== every ζ carries a fit_provenance ==")
stamps = {}
for L, p in sorted(PATHS.items()):
    h = read_isdf_header(p)
    stamps[L] = h.fit_provenance
    check("mu_L=%d stamped and complete (%s)" % (L, os.path.basename(p)),
          h.fit_provenance is not None and bool(h.zeta_is_done))

print("== the four stamps differ in exactly n_rmu + vertex_mu_L ==")
d0 = json.loads(stamps[0])
for L in (1, 2, 3):
    dL = json.loads(stamps[L])
    diff = sorted(k for k in set(d0) | set(dL) if d0.get(k) != dL.get(k))
    check("mu_L=%d diff vs charge == ['n_rmu', 'vertex_mu_L'] (got %s)"
          % (L, diff), diff == ['n_rmu', 'vertex_mu_L'])
    check("mu_L=%d vertex recorded" % L, dL['vertex_mu_L'] == L)
print("  charge stamp: " + json.dumps(d0, sort_keys=True))

print("== transverse identity is present on every stamp ==")
for L in sorted(PATHS):
    d = json.loads(stamps[L])
    check("mu_L=%d names n_rmu_transverse/md5/lu/kind" % L,
          d.get('n_rmu_transverse') is not None
          and d.get('centroids_transverse_md5') is not None
          and d.get('distributed_lu') is not None
          and d.get('transverse_solver_kind') is not None)

print("== the true stamps are REUSABLE against their own files ==")
for L, p in sorted(PATHS.items()):
    h = read_isdf_header(p)
    ok = gw_init._zeta_reuse_ok(p, stamps[L], h.r_mu_fft_idx,
                                print_fn=lambda *a: None,
                                n_rmu_expected=h.r_mu_fft_idx.shape[0])
    check("mu_L=%d reuse OK against itself" % L, ok)

print("== each new key, doctored, forces a REFIT ==")
DOCTOR = {
    'n_rmu_transverse': 999,
    'centroids_transverse_md5': '0' * 32,
    'distributed_lu': 'scalapack',
    'transverse_solver_kind': 'cusolvermp_lu',
    'vertex_mu_L': 7,
}
for key, val in DOCTOR.items():
    for L, p in sorted(PATHS.items()):
        d = json.loads(stamps[L])
        d[key] = val
        h = read_isdf_header(p)
        msgs = []
        ok = gw_init._zeta_reuse_ok(p, json.dumps(d, sort_keys=True),
                                    h.r_mu_fft_idx, print_fn=msgs.append,
                                    n_rmu_expected=h.r_mu_fft_idx.shape[0])
        named = any(key in m for m in msgs)
        check("mu_L=%d %s changed -> refit, key named" % (L, key),
              (not ok) and named)

print("== a stamp with the four keys STRIPPED is refused for bispinor ==")
NEW = ('n_rmu_transverse', 'centroids_transverse_md5', 'distributed_lu',
       'transverse_solver_kind')
for L, p in sorted(PATHS.items()):
    d = json.loads(stamps[L])
    for k in NEW:
        d.pop(k, None)
    h = read_isdf_header(p)
    msgs = []
    # The RUN still requests the real (bispinor) values -> legacy-implied
    # None mismatches every one of them -> refit.
    ok = gw_init._zeta_reuse_ok(p, stamps[L], h.r_mu_fft_idx,
                                print_fn=msgs.append,
                                n_rmu_expected=h.r_mu_fft_idx.shape[0])
    check("mu_L=%d (control) true stamp still reuses" % L, ok)
    # And with the file's stamp doctored to the legacy shape it must not.
    import file_io.isdf_header as _ih
    _real = _ih.read_isdf_header

    class _H:
        pass

    def _stub(_p, _hdr=h, _prov=json.dumps(d, sort_keys=True)):
        o = _H()
        o.zeta_is_done = True
        o.fit_provenance = _prov
        o.r_mu_fft_idx = _hdr.r_mu_fft_idx
        return o

    _ih.read_isdf_header = _stub
    try:
        msgs = []
        ok = gw_init._zeta_reuse_ok(p, stamps[L], h.r_mu_fft_idx,
                                    print_fn=msgs.append,
                                    n_rmu_expected=h.r_mu_fft_idx.shape[0])
    finally:
        _ih.read_isdf_header = _real
    check("mu_L=%d legacy-shaped stamp refused for bispinor" % L,
          (not ok) and any('n_rmu_transverse' in m for m in msgs))

print("[reader matrix] %d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
