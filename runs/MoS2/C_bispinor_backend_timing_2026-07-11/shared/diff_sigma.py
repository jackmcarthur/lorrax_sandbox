#!/usr/bin/env python3
"""Max |delta| between two sigma_diag.dat files (all numeric fields).

sigma_diag.dat is not a plain numeric table (``k-point N:`` section lines,
``key= value`` fields, ``a+ bi`` complex pairs), so extract every decimal
float from non-comment lines instead of np.loadtxt.
"""
import re
import sys

import numpy as np

FLOAT = re.compile(r"[-+]?[0-9]+\.[0-9]+")


def load(path):
    vals = []
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            vals.extend(float(tok) for tok in FLOAT.findall(line))
    return np.asarray(vals)


a = load(sys.argv[1])
b = load(sys.argv[2])
assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
print(f"max|delta| = {np.max(np.abs(a - b)):.3e}  ({sys.argv[1]} vs {sys.argv[2]})")
