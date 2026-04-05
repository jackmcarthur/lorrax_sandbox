# Output Parsing Reference

Prewritten parsers and column definitions for extracting comparison data from
BerkeleyGW and GWJAX output files. **Use these consistently** — do not write
ad-hoc parsing logic for the same quantities.

---

## BerkeleyGW: `sigma_hp.log`

The high-precision sigma output. One block per k-point. Columns (1-indexed,
whitespace-delimited):

| Col | Name | Description |
|-----|------|-------------|
| 1 | n | Band index (1-indexed) |
| 2 | Emf | Mean-field energy (eV) |
| 3 | Eo | DFT eigenvalue (eV) |
| 4 | X | Bare exchange Σ_x (eV) |
| 5 | SX-X | Screened exchange minus bare exchange (eV) |
| 6 | CH | Coulomb hole, partial-sum (eV) |
| 7 | Sig | Total self-energy X + SX-X + CH (eV) |
| 8 | Vxc | Exchange-correlation potential (eV) |
| 9 | Eqp0 | QP energy, linearized (eV) |
| 10 | Eqp1 | QP energy, iterated (eV) |
| 11 | CH' | Coulomb hole, with static remainder correction (eV) |
| 12 | Sig' | Total self-energy using CH' (eV) |
| 13 | Eqp0' | QP energy using Sig' (eV) |
| 14 | Eqp1' | QP energy using Sig', iterated (eV) |
| 15 | Znk | Renormalization factor |

**Key derived quantity**: `Cor' = SX-X + CH' = col5 + col11`. This is the
correlation self-energy to compare against GWJAX `sigC_EDFT`. For static
COHSEX, CH ≠ CH' because CH uses a partial band sum while CH' includes the
static remainder correction. Always compare against CH' (primed).

```python
def parse_sigma_hp(path):
    """Parse sigma_hp.log -> list of dicts per k-point, each with bands dict."""
    import re
    blocks = []
    ik, kcrys = None, None
    for line in open(path):
        s = line.strip()
        m = re.match(r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)', s)
        if m:
            kcrys = tuple(float(m.group(i)) for i in (1, 2, 3))
            ik = int(m.group(4))
            continue
        if ik is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            if not any(b.get('ik') == ik for b in blocks):
                blocks.append({'kcrys': kcrys, 'ik': ik, 'bands': {}})
            n = int(p[0])
            blocks[-1]['bands'][n] = {
                'X': float(p[3]), 'SX_X': float(p[4]),
                'CH': float(p[5]), 'CHp': float(p[10]),
                'Corp': float(p[4]) + float(p[10]),
            }
    return blocks
```

---

## BerkeleyGW: `epsilon.out` — Mini-BZ Coulomb Head Values

For `frequency_dependence 3` (GN-GPP), epsilon prints the mini-BZ averaged
Coulomb head values needed for GWJAX's finite-size correction. These are
**critical** for accurate GWJAX results — the S-tensor default can be
significantly wrong, especially for coarse k-grids where the mini-BZ is large.

Lines to grep:

```
q-pt  1: Vcoul head (MiniBZ)       = 3.150136999989064E+002
q-pt  1: Wcoul head (MiniBZ iw=1)  = 5.561971536649861E+001   # ω = 0 (static)
q-pt  1: Wcoul head (MiniBZ iw=2)  = 2.461099166562844E+002   # ω = iωp (GN probe)
```

These correspond to GWJAX cohsex.in overrides:

| epsilon.out line | cohsex.in key | Description |
|---|---|---|
| `Vcoul head (MiniBZ)` | `vhead` | Mini-BZ avg bare Coulomb head (a.u.) |
| `Wcoul head (MiniBZ iw=1)` | `whead_0freq` | Mini-BZ avg screened Coulomb head at ω=0 (a.u.) |
| `Wcoul head (MiniBZ iw=2)` | `whead_imfreq` | Mini-BZ avg screened Coulomb head at ω=iωp (a.u.) |

**When to use**: Always extract these from the `frequency_dependence 3` BGW
epsilon.out and pass them as overrides in cohsex.in when running GWJAX
comparisons. The GWJAX S-tensor computation (`wcoul0_source = s_tensor`) uses
a macroscopic dielectric model that breaks down for coarse k-grids where the
mini-BZ extends far from Γ. For MoS2 1×1 the S-tensor gives wcoul0 = 42.7
vs BGW's 55.6 (30% error), causing 165 meV MAE that drops to 46 meV with
the BGW values.

**Note**: Only `frequency_dependence 3` prints these lines. Static COHSEX
(`frequency_dependence 0`) does NOT print them. Run a `frequency_dependence 3`
epsilon once to extract the values, then use them for both GN and COHSEX GWJAX
runs.

```python
def parse_epsilon_heads(path):
    """Extract mini-BZ Vcoul/Wcoul heads from BGW epsilon.out.

    Returns dict with keys: vhead, whead_0freq, whead_imfreq (floats, a.u.).
    Only works for frequency_dependence 3 output.
    """
    import re
    result = {}
    for line in open(path):
        m = re.search(r'Vcoul head \(MiniBZ\)\s*=\s*([\d.Ee+-]+)', line)
        if m:
            result['vhead'] = float(m.group(1))
        m = re.search(r'Wcoul head \(MiniBZ iw=1\)\s*=\s*([\d.Ee+-]+)', line)
        if m:
            result['whead_0freq'] = float(m.group(1))
        m = re.search(r'Wcoul head \(MiniBZ iw=2\)\s*=\s*([\d.Ee+-]+)', line)
        if m:
            result['whead_imfreq'] = float(m.group(1))
    return result
```

---

## GWJAX: `eqp0.dat`

Quasiparticle self-energy output. One block per k-point. Format depends on
the calculation type.

### GN-PPM (`use_ppm_sigma = true`)

```
n=18  sigX=  -33.432605  sigC_EDFT=  9.099091+ -0.012866i  sigXC_EDFT=  -24.333514+ -0.012866i  VH=  319.080251+ -0.000002i
```

| Field | Description |
|-------|-------------|
| `sigX` | Bare exchange (real, eV). Compare to BGW col 4 (X). |
| `sigC_EDFT` | Correlation at DFT energy (complex, eV). Real part compares to BGW Cor'. |
| `sigXC_EDFT` | sigX + sigC_EDFT (complex, eV). |
| `VH` | Hartree potential diagonal (complex, eV). |

`sigC_EDFT` may be `nan` for bands whose DFT energy falls outside the
`sigma_omega_min_ev / sigma_omega_max_ev` grid.

### Static COHSEX (`use_ppm_sigma = false`)

```
n=18  sigSX=  -9.357814  sigCOH=  -15.457215  sigTOT=  -24.815029  VH=  319.080251+ -0.000002i
```

| Field | Description |
|-------|-------------|
| `sigSX` | Screened exchange (real, eV). Corresponds to BGW SX = X + (SX-X). |
| `sigCOH` | Coulomb hole (real, eV). Compare to BGW CH' (col 11), NOT CH (col 6). |
| `sigTOT` | sigSX + sigCOH = total Σ_xc (real, eV). |

To compare COHSEX correlation with BGW Cor', compute
`(sigSX - sigX) + sigCOH` where `sigX` comes from a separate GN-PPM run
(bare exchange is independent of screening method).

**Band indexing**: GWJAX is 0-indexed, BGW is 1-indexed.
GWJAX `n=18` = BGW `band 19`.

```python
def parse_eqp0_gn(path):
    """Parse GN-PPM eqp0.dat -> {band_1idx: {'sigX': float, 'sigC': float}}."""
    import re, numpy as np
    bands = {}
    for line in open(path):
        m = re.match(r'n=(\d+)\s+', line.strip())
        if not m: continue
        n1 = int(m.group(1)) + 1
        mx = re.search(r'sigX=\s*([-\d.Ee]+)', line)
        mc = re.search(r'sigC_EDFT=\s*([-\d.Ee]+)', line)
        if mx and mc:
            sigc = float(mc.group(1))
            if not np.isnan(sigc):
                bands[n1] = {'sigX': float(mx.group(1)), 'sigC': sigc}
    return bands


def parse_eqp0_cohsex(path):
    """Parse COHSEX eqp0.dat -> {band_1idx: {'sigSX': float, 'sigCOH': float}}."""
    import re
    bands = {}
    for line in open(path):
        m = re.match(r'n=(\d+)\s+', line.strip())
        if not m: continue
        n1 = int(m.group(1)) + 1
        ms = re.search(r'sigSX=\s*([-\d.Ee]+)', line)
        mc = re.search(r'sigCOH=\s*([-\d.Ee]+)', line)
        if ms and mc:
            bands[n1] = {'sigSX': float(ms.group(1)), 'sigCOH': float(mc.group(1))}
    return bands
```

---

## GWJAX: `sigma_freq_debug.dat`

Detailed frequency decomposition (PPM runs only, when
`sigma_freq_debug_output = true`). Tab-separated columns:

| Col | Name | Description |
|-----|------|-------------|
| 0 | k | k-point index |
| 1 | n | Band index (0-indexed) |
| 2 | Edft-Ef | DFT energy relative to Fermi (eV) |
| 3 | E_dft | Absolute DFT energy (eV) |
| 4 | kin_ion | Kinetic + ionic matrix element (eV) |
| 5 | sex_0 | Static screened exchange (eV) |
| 6 | coh_0 | Static Coulomb hole (eV) |
| 7 | x_bare | Bare exchange (eV) |
| 8 | sig_c(0) | Static correlation (eV) |
| 9 | sig_c+(w) | Positive-pole PPM contribution (eV) |
| 10 | sig_c-(w) | Negative-pole PPM contribution (eV) |
| 11 | sig_c_invld(0) | Invalid-pole replacement value (eV) |
| 12 | sig_c(Edft) | Correlation at DFT energy (eV) — compare to BGW Cor' |
| 13 | sig_c_head | Head correction (eV, diagnostic only unless `apply_head_diagonal=true`) |

**Note**: The header comment "sig_c(Edft) includes head" is misleading —
sig_c(Edft) does NOT include the head unless `apply_head_diagonal = true`.
See `KNOWN_SANDBOX_ERRORS.md`.
