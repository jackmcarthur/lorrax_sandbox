"""Audit PR2 — R1 regression: PR2 IBZ vs reference full-BZ.

Per audit spec: max |Δx_bare| must be 0 (literal bit-equality; PR2 only
refactored the call path). sex_0 + coh_0 also compared.

Note: SKILL parser only returns k=0; expand here to all k for full-grid coverage.
"""
import numpy as np

def parse_all_k(path):
    """Return list of dicts, one per row, with k,n_phys,x_bare,sex_0,coh_0."""
    rows = []
    for line in open(path):
        s = line.strip()
        if s.startswith('#') or s.startswith('k') or not s:
            continue
        p = s.split()
        if len(p) >= 13:
            try:
                rows.append({
                    'k': int(p[0]),
                    'n': int(p[1]),
                    'sex_0': float(p[5]),
                    'coh_0': float(p[6]),
                    'x_bare': float(p[7]),
                })
            except (ValueError, IndexError):
                pass
    return rows


pr2  = parse_all_k('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_pr2/sigma_freq_debug.dat')
ref  = parse_all_k('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_B_fullbz/sigma_freq_debug.dat')
post = parse_all_k('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz/sigma_freq_debug.dat')

print(f"# rows PR2 IBZ        : {len(pr2)}")
print(f"# rows REF run_B_fullbz: {len(ref)}")
print(f"# rows PRIOR run_A_ibz : {len(post)}")

def key(r):
    return (r['k'], r['n'])

pr2_map  = {key(r): r for r in pr2}
ref_map  = {key(r): r for r in ref}
post_map = {key(r): r for r in post}

# Compare PR2 vs run_B_fullbz (full-BZ reference — same physics)
print("\n=== PR2 IBZ vs run_B_fullbz (TRS-aware unfold should match) ===")
maxdiff = {'x_bare': 0.0, 'sex_0+coh_0': 0.0, 'sex_0': 0.0, 'coh_0': 0.0}
worst = {'x_bare': None, 'sex_0+coh_0': None}
for k, r1 in pr2_map.items():
    if k not in ref_map: continue
    r2 = ref_map[k]
    dx  = abs(r1['x_bare'] - r2['x_bare'])
    dsx = abs(r1['sex_0'] - r2['sex_0'])
    dch = abs(r1['coh_0'] - r2['coh_0'])
    dsxch = abs((r1['sex_0']+r1['coh_0']) - (r2['sex_0']+r2['coh_0']))
    if dx > maxdiff['x_bare']: maxdiff['x_bare']=dx; worst['x_bare']=k
    if dsx > maxdiff['sex_0']: maxdiff['sex_0']=dsx
    if dch > maxdiff['coh_0']: maxdiff['coh_0']=dch
    if dsxch > maxdiff['sex_0+coh_0']: maxdiff['sex_0+coh_0']=dsxch; worst['sex_0+coh_0']=k
print(f"  max |Δx_bare|         = {maxdiff['x_bare']:.6e}  at (k,n)={worst['x_bare']}")
print(f"  max |Δ(sex_0+coh_0)|  = {maxdiff['sex_0+coh_0']:.6e}  at (k,n)={worst['sex_0+coh_0']}")
print(f"  max |Δsex_0|          = {maxdiff['sex_0']:.6e}")
print(f"  max |Δcoh_0|          = {maxdiff['coh_0']:.6e}")

# Compare PR2 vs run_A_ibz (Phase 1 IBZ-cascade prior) — should be byte-exact (only refactor)
print("\n=== PR2 IBZ vs PRIOR run_A_ibz (Phase 1 path — should be bit-equal refactor) ===")
maxdiff2 = {'x_bare': 0.0, 'sex_0+coh_0': 0.0}
worst2 = {'x_bare': None, 'sex_0+coh_0': None}
for k, r1 in pr2_map.items():
    if k not in post_map: continue
    r2 = post_map[k]
    dx  = abs(r1['x_bare'] - r2['x_bare'])
    dsxch = abs((r1['sex_0']+r1['coh_0']) - (r2['sex_0']+r2['coh_0']))
    if dx > maxdiff2['x_bare']: maxdiff2['x_bare']=dx; worst2['x_bare']=k
    if dsxch > maxdiff2['sex_0+coh_0']: maxdiff2['sex_0+coh_0']=dsxch; worst2['sex_0+coh_0']=k
print(f"  max |Δx_bare|         = {maxdiff2['x_bare']:.6e}  at (k,n)={worst2['x_bare']}")
print(f"  max |Δ(sex_0+coh_0)|  = {maxdiff2['sex_0+coh_0']:.6e}  at (k,n)={worst2['sex_0+coh_0']}")

# Verdict
print("\n=== AUDIT VERDICT ===")
if maxdiff['x_bare'] < 1e-10:
    print(f"PASS: max |Δx_bare| (PR2 vs full-BZ ref) = {maxdiff['x_bare']:.3e} eV < 1e-10")
else:
    print(f"FAIL: max |Δx_bare| = {maxdiff['x_bare']:.3e} eV exceeds 1e-10")
if maxdiff2['x_bare'] < 1e-10:
    print(f"PASS: PR2 vs prior Phase1 IBZ match — refactor is bit-equal as intended")
else:
    print(f"FAIL: PR2 vs Phase1 IBZ diverge — refactor changed the math: max|Δ|={maxdiff2['x_bare']:.3e}")


# Additional checks: PR2 vs post-Phase1 fix (run_A_ibz_postfix) and PR1 (run_A_ibz_pr1)
print("\n=== Additional: PR2 vs run_A_ibz_postfix (Phase 1 fix landed) ===")
pf = parse_all_k('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_postfix/sigma_freq_debug.dat')
pf_map = {(r['k'], r['n']): r for r in pf}
max_dx = 0.0; max_dsxch = 0.0; worst_dx = None; worst_dsxch = None
for k, r1 in pr2_map.items():
    if k not in pf_map: continue
    r2 = pf_map[k]
    dx = abs(r1['x_bare'] - r2['x_bare'])
    dsxch = abs((r1['sex_0']+r1['coh_0']) - (r2['sex_0']+r2['coh_0']))
    if dx > max_dx: max_dx = dx; worst_dx = k
    if dsxch > max_dsxch: max_dsxch = dsxch; worst_dsxch = k
print(f"  max |Δx_bare|        = {max_dx:.6e}  at (k,n)={worst_dx}")
print(f"  max |Δ(sex_0+coh_0)| = {max_dsxch:.6e}  at (k,n)={worst_dsxch}")

print("\n=== Additional: PR2 vs run_A_ibz_pr1 (PR1 landed) ===")
p1 = parse_all_k('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_pr1/sigma_freq_debug.dat')
p1_map = {(r['k'], r['n']): r for r in p1}
max_dx = 0.0; max_dsxch = 0.0; worst_dx = None; worst_dsxch = None
for k, r1 in pr2_map.items():
    if k not in p1_map: continue
    r2 = p1_map[k]
    dx = abs(r1['x_bare'] - r2['x_bare'])
    dsxch = abs((r1['sex_0']+r1['coh_0']) - (r2['sex_0']+r2['coh_0']))
    if dx > max_dx: max_dx = dx; worst_dx = k
    if dsxch > max_dsxch: max_dsxch = dsxch; worst_dsxch = k
print(f"  max |Δx_bare|        = {max_dx:.6e}  at (k,n)={worst_dx}")
print(f"  max |Δ(sex_0+coh_0)| = {max_dsxch:.6e}  at (k,n)={worst_dsxch}")
