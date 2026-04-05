# Comparing BGW and LORRAX GW Self-Energies

How to parse, compare, and plot GW output from BerkeleyGW and LORRAX.

## 1. Physics: BGW vs LORRAX decomposition

The GN-PPM correlation self-energy is:

```
Σ^c(ω) = Σ_{m,q,μν} M²_{m,q,μν} · 2B_q(μ,ν)Ω_q(μ,ν) / ((ω - E_m)² - Ω²)
```

Using partial fractions `2BΩ/((ω-E)²-Ω²) = B/(ω-E-Ω) - B/(ω-E+Ω)`:

**LORRAX decomposes by band occupation:**
```
Σ⁺(ω) = Σ_{v∈occ}  (-B) / (ω + h_v + Ω)     [occupied/valence branch]
Σ⁻(ω) = ��_{c∈unocc}  B  / (ω - E_c - Ω)     [unoccupied/conduction branch]
```
where `h_v = E_F - E_v ≥ 0` and `E_c = E_c - E_F ≥ 0`. Each band contributes
through only ONE pole (contour integration selects which based on occupation).

**BGW decomposes as SX-X + CH:**
```
SX-X(��) = Σ_{v∈occ}  M² · [W^c(ω-E_v) - v]     [screened exchange, occupied only]
CH(ω)   = ½ Σ_{m∈all} M² · W^c(ω-E_m)            [Coulomb hole, ALL bands]
```

**The total is the same**: `Σ⁺ + Σ⁻ = SX-X + CH = Cor`. But the individual
pieces are NOT equivalent:
- Σ⁺ ≠ SX-X (Σ⁺ uses one PPM pole; SX-X uses the full W^c for occupied bands)
- Σ⁻ ≠ CH (Σ⁻ sums unoccupied only; CH sums all bands)

**Only the sum Cor is a valid cross-code comparison target.** Comparing
Σ⁺ vs SX-X or Σ⁻ vs CH individually will show large, meaningless differences.

The ISDF SoS script (`isdf_sos_debug.py`) can reproduce the BGW-style SX-X and CH
decomposition in the ISDF basis for detailed debugging — use it when you need to
isolate whether a discrepancy is in the ISDF projection or the frequency integration.

## 2. Output formats and parsers

**Use these parsers for all output extraction.** If you write a new parser for a
new output format, add it to this file.

### 2a. BGW `sigma_hp.log`

Machine-readable BGW sigma output. One block per k-point.

```
  n     Emf      Eo       X      SX-X      CH       Sig      Vxc     Eqp0     Eqp1      CH'      Sig'    Eqp0'    Eqp1'     Znk
  0      1        2       3        4        5         6        7        8        9        10        11       12       13       14
```

Key comparison quantity: **`Cor' = SX-X + CH'`** (cols 4 + 10). For `frequency_dependence 3`, primed = unprimed.

```python
import re
import numpy as np

def parse_sigma_hp(path):
    """Parse sigma_hp.log → list of dicts per k-point block.
    Each: {'kcrys': (kx,ky,kz), 'ik': int,
           'bands': {n: {'X','SXmX','CH','CHp','Cor','Corp','Sig','Vxc','Eqp0','Eqp1','Znk'}}}
    """
    blocks = []
    ik = None
    kcrys = None
    for line in open(path):
        s = line.strip()
        m = re.match(r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)', s)
        if m:
            kcrys = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            ik = int(m.group(4))
            continue
        if ik is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            if not any(b.get('ik') == ik for b in blocks):
                blocks.append({'kcrys': kcrys, 'ik': ik, 'bands': {}})
            blocks[-1]['bands'][n] = {
                'X': float(p[3]), 'SXmX': float(p[4]), 'CH': float(p[5]),
                'Sig': float(p[6]), 'Vxc': float(p[7]),
                'Eqp0': float(p[8]), 'Eqp1': float(p[9]),
                'CHp': float(p[10]), 'Sigp': float(p[11]),
                'Znk': float(p[14]),
                'Cor': float(p[4]) + float(p[5]),
                'Corp': float(p[4]) + float(p[10]),
            }
    return blocks
```

### 2b. BGW `ch_converge.dat`

Partial CH sum vs number of bands, for VBM and CBM only.

```
#    Nb          CH(vbm)          CH(cbm)             diff
      1   3.44801763E-02   1.59148440E-02  -1.85653323E-02
     80  -1.03131965E+01  -1.25103018E+01  -2.19710529E+00
```

```python
def parse_ch_converge(path):
    """Returns (nb, ch_vbm, ch_cbm) arrays."""
    nb, vbm, cbm = [], [], []
    for line in open(path):
        s = line.strip()
        if s.startswith('#') or not s:
            continue
        p = s.split()
        if len(p) >= 3:
            nb.append(int(p[0]))
            vbm.append(float(p[1]))
            cbm.append(float(p[2]))
    return np.array(nb), np.array(vbm), np.array(cbm)
```

### 2c. LORRAX `sigma_freq_debug.dat`

Per-band diagonal sigma decomposition. Written when `sigma_freq_debug_output = true`.

```
k    n    Edft-Ef  E_dft  kin_ion  sex_0  coh_0  x_bare  sig_c(0)  sig_c+(w)  sig_c-(w)  sig_c_invld(0)  sig_c(Edft)  [sig_c_head]
0    1      2        3      4       5      6       7        8          9          10           11              12            13
```

Key comparison quantity: **`sig_c(Edft)`** (col 12) vs BGW `Cor'`.
Band mapping: LORRAX `n` is 0-indexed; BGW is 1-indexed. Physical band = n + 1.
Optional col 13 (`sig_c_head`) is the scalar head correction, present when head
correction module is active. Check whether sig_c(Edft) includes the head by
reading the file header comment.

```python
def parse_sigma_freq_debug(path):
    """Returns dict of {physical_band: {field: value}} for k=0."""
    bands = {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#') or s.startswith('k') or not s:
            continue
        p = s.split()
        if len(p) >= 13:
            try:
                n_phys = int(p[1]) + 1
                entry = {
                    'k': int(p[0]),
                    'Edft_rel': float(p[2]),
                    'Edft': float(p[3]),
                    'kin_ion': float(p[4]),
                    'sex_0': float(p[5]),
                    'coh_0': float(p[6]),
                    'x_bare': float(p[7]),
                    'sigc_0': float(p[8]),
                    'sigc_plus': float(p[9]) if p[9] != 'nan' else np.nan,
                    'sigc_minus': float(p[10]) if p[10] != 'nan' else np.nan,
                    'sigc_invld': float(p[11]),
                    'sigc_edft': float(p[12]) if p[12] != 'nan' else np.nan,
                }
                if len(p) >= 14:
                    entry['sigc_head'] = float(p[13]) if p[13] != 'nan' else 0.0
                bands[n_phys] = entry
            except (ValueError, IndexError):
                pass
    return bands
```

### 2d. ISDF SoS `sos_cor_gn.dat`

Produced by `isdf_sos_debug.py`. **Requires `eps0mat.h5`** (BGW output).

```
  n    E_dft    SEX-X_GN    COH_GN    Cor_GN    Sig_GN
  0     1         2           3         4         5
```

Band indexing auto-detected: 0-indexed in stdout, 1-indexed in `--out-cor-dat` file.

```python
def parse_sos_cor_gn(path):
    """Returns {physical_band: {'Edft','SXmX','COH','Cor'}}."""
    raw = []
    for line in open(path):
        if line.startswith('#'):
            continue
        p = line.split()
        if len(p) >= 5:
            try:
                raw.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            except (ValueError, IndexError):
                pass
    if not raw:
        return {}
    offset = 1 if raw[0][0] == 0 else 0
    return {n + offset: {'Edft': e, 'SXmX': sx, 'COH': ch, 'Cor': cor}
            for n, e, sx, ch, cor in raw}
```

### Running the SoS code

```bash
uv run -- python isdf_sos_debug.py \
  -g cohsex.in \
  --n-valence 26 --n-cond-solve 6 --n-sum-states 80 \
  --transpose-eps 1 --patch-head 1 --zero-q0-wings 1 \
  --gn-sx-form bgw --gpp-broadening-ev 0.5 --gpp-sexcutoff 4.0 \
  2>&1 | tee sos_dynamic_run.log
```

Key flags: `--n-valence` = occupied bands (= nelec for spinor), `--n-sum-states` =
total (match BGW `number_bands`), `--transpose-eps 1` (BGW epsilon orientation),
`--patch-head 1` (BGW wcoul0/vcoul0 convention), `--zero-q0-wings 1` (BGW slab convention).

## 3. Comparison procedure

### Quick: single k-point, sigma_freq_debug vs sigma_hp

For a gamma-only or single-k comparison where both files are available:

1. Parse BGW with `parse_sigma_hp` → extract `Corp` per band.
2. Parse LORRAX with `parse_sigma_freq_debug` → extract `sigc_edft` per band.
3. Match bands by physical index (BGW 1-indexed = LORRAX n+1).
4. Compute `diff = sigc_edft - Corp` per band, then MAE and max|Δ|.

### Full: multi-k, eqp0.dat vs sigma_hp

Use `compare_bgw_gwjax.py` for multi-k comparisons. It matches k-points by
crystal coordinates via WFN.h5 and handles the BGW symmetry-reduced grid:

```bash
MPLBACKEND=Agg uv run --project /home/jackm/projects/lorrax python \
  compare_bgw_gwjax.py \
  --bgw-hp /path/to/sigma_hp.log \
  --gw-eqp /path/to/eqp0.dat \
  --wfn /path/to/WFN.h5 \
  --out-dat compare_cor.dat \
  --out-png compare_cor.png
```

### Three-way: BGW vs LORRAX vs ISDF SoS

When debugging whether a discrepancy is in ISDF projection or frequency integration:

1. Run `isdf_sos_debug.py` (requires `eps0mat.h5`).
2. Parse all three with the parsers above.
3. Compare `Cor` columns. If ISDF SoS agrees with LORRAX but not BGW, the issue is
   in the ISDF projection of W. If ISDF SoS agrees with BGW, the issue is in the
   LORRAX frequency integration.

## 4. Plotting

Always use `MPLBACKEND=Agg` for headless plot generation. Standard comparison plot:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# After computing bgw_cor, lorrax_sigc, diff arrays for matched bands:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(bands, bgw_cor, 'o-', label="BGW Cor'")
ax1.plot(bands, lorrax_sigc, 's--', label='LORRAX sig_c(Edft)')
ax1.set_xlabel('Band index')
ax1.set_ylabel('Cor (eV)')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.bar(range(len(diff)), diff)
ax2.axhline(0, color='black', lw=0.5)
ax2.set_xlabel('Band index')
ax2.set_ylabel('LORRAX - BGW (eV)')
ax2.set_title(f'MAE = {mae:.3f} eV')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
```
