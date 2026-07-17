"""Static analysis of a compiled BSE matvec HLO dump (login-node, no GPU).

Extracts, from `compiled.as_text()` output:
  * collective schedule: all-gather / collective-permute / reduce-scatter /
    all-reduce ops, in program order, with shapes + channel ids
  * every copy / transpose / bitcast op (layout mismatches XLA papered over)
  * gemm/dot ops with operand layouts ({1,0} row-major vs {0,1} col-major)
  * fft ops
  * fusion count + rough boundaries

usage: hlo_analyze.py <hlo_text_file>
"""
import re, sys
from collections import Counter

path = sys.argv[1]
txt = open(path).read()
lines = txt.splitlines()

# match a shape+layout token like f64[400,400]{1,0} or c128[9,2,2,400]{3,2,1,0}
SHAPE = r'[a-z0-9]+\[[0-9,]*\](?:\{[0-9,]*\})?'

def op_lines(kw):
    out = []
    for i, ln in enumerate(lines):
        # the op name appears after '= <shape> <opname>('
        m = re.search(r'=\s*(' + SHAPE + r')\s+' + kw + r'\(', ln)
        if m:
            out.append((i + 1, ln.strip()))
    return out

collectives = {}
for kw in ["all-gather", "all-gather-start", "all-gather-done",
           "collective-permute", "collective-permute-start", "collective-permute-done",
           "reduce-scatter", "all-reduce", "all-reduce-start", "all-reduce-done",
           "all-to-all"]:
    ls = op_lines(kw)
    if ls:
        collectives[kw] = ls

copies = {}
for kw in ["copy", "copy-start", "copy-done", "transpose", "bitcast"]:
    ls = op_lines(kw)
    if ls:
        copies[kw] = ls

fusions = op_lines("fusion")
ffts = op_lines("fft")

# gemm: custom-call to cublas OR dot
gemms = []
for i, ln in enumerate(lines):
    if "__cublas$gemm" in ln or "__cublas$lt$matmul" in ln:
        gemms.append((i + 1, ln.strip()))
dots = op_lines("dot")

# ---- report ----
print(f"### HLO analysis: {path}")
print(f"total HLO lines: {len(lines)}")
print()

print("## Collective schedule (program order)")
allc = []
for kw, ls in collectives.items():
    for (ln_no, ln) in ls:
        allc.append((ln_no, kw, ln))
allc.sort()
for ln_no, kw, ln in allc:
    # extract shape + channel_id + replica/source-target
    sh = re.search(r'=\s*(' + SHAPE + r')', ln)
    ch = re.search(r'channel_id=(\d+)', ln)
    dims = re.search(r'(dimensions=\{[0-9,]*\})', ln)
    st = re.search(r'(source_target_pairs=\{[^}]*\})', ln)
    rg = re.search(r'(replica_groups=\{[^}]*\})', ln)
    extra = " ".join(x.group(1) for x in [dims, st, rg] if x)
    print(f"  L{ln_no:5d} {kw:26s} {sh.group(1) if sh else '?':40s} "
          f"ch={ch.group(1) if ch else '-':>3} {extra}")
print(f"  [counts] " + ", ".join(f"{k}={len(v)}" for k, v in collectives.items()))
print()

print("## Layout-fixup ops (copy / transpose / bitcast)")
for kw, ls in copies.items():
    print(f"  {kw}: {len(ls)}")
# print the non-bitcast copies + transposes (these are the real data movements)
real = []
for kw in ("copy", "copy-start", "transpose"):
    for (ln_no, ln) in copies.get(kw, []):
        sh = re.search(r'=\s*(' + SHAPE + r')', ln)
        real.append((ln_no, kw, sh.group(1) if sh else '?'))
real.sort()
print(f"  -- real copies/transposes (non-bitcast), {len(real)} total:")
for ln_no, kw, sh in real[:60]:
    print(f"     L{ln_no:5d} {kw:12s} {sh}")
if len(real) > 60:
    print(f"     ... (+{len(real)-60} more)")
print()

print("## GEMM / dot operand layouts")
print(f"  cublas gemm custom-calls: {len(gemms)}")
print(f"  dot ops (non-cublas):     {len(dots)}")
def layouts_in(ln):
    return re.findall(r'\{[0-9,]+\}', ln)
for (ln_no, ln) in (gemms + dots):
    # operand shapes are inside the parens
    args = ln[ln.find("(")+1:]
    shapes = re.findall(SHAPE, ln)
    lays = re.findall(r'([a-z0-9]+\[[0-9,]*\]\{[0-9,]*\})', ln)
    print(f"  L{ln_no:5d} " + (" ; ".join(lays[:6])))
print()

print("## FFT ops")
for (ln_no, ln) in ffts:
    sh = re.search(r'=\s*(' + SHAPE + r')', ln)
    ft = re.search(r'fft_type=(\w+)', ln)
    fl = re.search(r'fft_length=\{[0-9,]*\}', ln)
    print(f"  L{ln_no:5d} {sh.group(1) if sh else '?':40s} "
          f"{ft.group(0) if ft else ''} {fl.group(0) if fl else ''}")
print(f"  [count] fft={len(ffts)}")
print()

print("## Fusions")
print(f"  fusion op count: {len(fusions)}")
kinds = Counter()
for (ln_no, ln) in fusions:
    k = re.search(r'kind=(\w+)', ln)
    kinds[k.group(1) if k else "?"] += 1
print(f"  fusion kinds: {dict(kinds)}")
