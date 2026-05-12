"""Sum sig_c over degenerate subspaces at Γ — invariant under gauge."""
import h5py, numpy as np, sys, re
from pathlib import Path
base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul')
ryd2ev = 13.6056980659

with h5py.File(base/'sigma_mnk_ppm.h5','r') as f:
    omega_ev=f['omega_ev'][...]; sigma_c=f['sigma_c_kij_ev'][...]
ol,oh=float(omega_ev.min()),float(omega_ev.max())
with h5py.File(base/'WFN.h5','r') as f:
    el_ry=f['mf_header/kpoints/el'][0]; nelec=int(f['mf_header/kpoints/ifmax'][0,0])
el_ev=el_ry*ryd2ev
ef=0.5*(max(np.max(el_ev[k,:nelec]) for k in range(el_ev.shape[0])) +
        min(np.min(el_ev[k,nelec:]) for k in range(el_ev.shape[0])))

# BGW PPM Cor' at Γ
bgw_corp = {}
cur = None
for raw in open('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/01_bgw_gn_ppm/sigma_hp.log'):
    s = raw.strip()
    m = re.match(r'k\s*=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+).*ik\s*=\s*(\d+)', s)
    if m: cur = tuple(round(float(x),6) for x in m.groups()[:3]); continue
    if cur is None: continue
    p = s.split()
    if len(p) >= 15 and p[0].isdigit():
        bgw_corp[(cur, int(p[0])-1)] = float(p[4]) + float(p[10])

# Group bands at Γ by exact degenerate energies
kfull = 0  # Γ
irrk = 0   # Γ in IRR
energies = el_ev[irrk, :16]
groups = []
i = 0
while i < 16:
    j = i + 1
    while j < 16 and abs(energies[j] - energies[i]) < 1e-6:
        j += 1
    groups.append(list(range(i, j)))
    i = j

print(f'Γ degenerate groups (n_0idx, energy):')
for g in groups:
    print(f'  bands {g[0]}..{g[-1]} ({len(g)} bands) at E={energies[g[0]]:+.3f} eV, ω_rel={energies[g[0]]-ef:+.3f} eV')

# Sum over each group
print(f'\nGroup    n_bands  E(eV)    LORRAX_sig_c_sum  BGW_Cor\'_sum   diff   diff/n_bands')
for g in groups:
    edft = energies[g[0]]
    omr = edft - ef
    if not (ol <= omr <= oh):
        print(f'  {g[0]+1}..{g[-1]+1} ({len(g)}b)  {edft:+.3f}  out-of-grid')
        continue
    lor_sum = sum(float(np.interp(omr, omega_ev, sigma_c[:, kfull, n, n].real)) for n in g)
    bgw_sum = sum(bgw_corp.get(((0.0, 0.0, 0.0), n), float('nan')) for n in g)
    d = lor_sum - bgw_sum
    print(f'  {g[0]+1}..{g[-1]+1} ({len(g)}b)  {edft:+.3f}  {lor_sum:+9.3f}        {bgw_sum:+9.3f}     {d:+8.3f}   {d/len(g):+.3f}')
