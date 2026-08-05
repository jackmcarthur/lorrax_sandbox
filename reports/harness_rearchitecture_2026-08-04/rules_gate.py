#!/usr/bin/env python3
"""rules_gate.py -- banned-pattern gates with ratcheting allowlists.

DRAFT (2026-08-05, harness re-architecture report). Runs on python 3.7+
with stdlib only, so it works on Frontera login nodes like the AST gate
suites. Intended home after consolidation: lorrax tests/ (run from its
__main__ like the other gates); lives in the sandbox report dir until
then.

Usage:
    python3 rules_gate.py --src /path/to/lorrax/src            # check
    python3 rules_gate.py --src /path/to/lorrax/src --regen    # rewrite allowlist

Semantics (the ratchet):
  - A violation in a file NOT in the allowlist fails the gate.
  - A count INCREASE in an allowlisted file fails the gate.
  - Count decreases are reported and require --regen to lock in (so the
    allowlist only ever shrinks deliberately, with a diff in review).
  - Every failure message teaches the rule: what was matched, the rule
    id, the helper to use instead, and the doc anchor (RULES_v2 ids).

Adding a rule = one entry in RULES below + --regen in the same commit.
"""
import argparse
import json
import os
import re
import sys

# rule_id -> dict(pattern, exempt_paths (prefixes, relative to src root),
#                 message: what to do instead)
RULES = {
    "D1-no-raw-jnp-fft": {
        "pattern": r"jnp\.fft\.|jax\.numpy\.fft\b",
        "exempt": ["common/fft_helpers.py"],
        "message": (
            "raw jnp.fft in stage code. Use the factories in "
            "common/fft_helpers.py (one FFT path; keeps sharding, box "
            "placement, Bloch phases consistent and NUFFT drop-in). "
            "RULES_v2 D1."
        ),
    },
    "B4-no-device-put": {
        "pattern": r"\bjax\.device_put\b|(?<![\w.])device_put\(",
        "exempt": [],
        "message": (
            "jax.device_put is banned: device data ingress goes through "
            "the FFI I/O layer / sharded loaders; large host data streams "
            "via io_callback. RULES_v2 B4 + owner rule 2026-08-05."
        ),
    },
}

ALLOWLIST_NAME = "rules_gate_allowlist.json"


def scan(src_root):
    """Return {rule_id: {relpath: count}} for all .py under src_root."""
    hits = {rid: {} for rid in RULES}
    compiled = {rid: re.compile(r["pattern"]) for rid, r in RULES.items()}
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, src_root).replace(os.sep, "/")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError as e:
                print("rules_gate: unreadable %s (%s)" % (rel, e))
                continue
            for rid, rx in compiled.items():
                if any(rel.startswith(p) for p in RULES[rid]["exempt"]):
                    continue
                n = len(rx.findall(text))
                if n:
                    hits[rid][rel] = n
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="lorrax src/ directory")
    ap.add_argument("--regen", action="store_true",
                    help="rewrite the allowlist from the current tree")
    ap.add_argument("--allowlist", default=None,
                    help="allowlist path (default: next to this script)")
    args = ap.parse_args()

    allowlist_path = args.allowlist or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ALLOWLIST_NAME)
    hits = scan(args.src)

    if args.regen:
        with open(allowlist_path, "w") as f:
            json.dump(hits, f, indent=2, sort_keys=True)
            f.write("\n")
        total = sum(sum(v.values()) for v in hits.values())
        print("rules_gate: allowlist regenerated (%d sites across %d rules) -> %s"
              % (total, len(RULES), allowlist_path))
        for rid in sorted(hits):
            n_files = len(hits[rid])
            n_sites = sum(hits[rid].values())
            print("  %-24s %3d sites in %d files" % (rid, n_sites, n_files))
        return 0

    try:
        with open(allowlist_path) as f:
            allow = json.load(f)
    except OSError:
        print("rules_gate: no allowlist at %s -- run with --regen first"
              % allowlist_path)
        return 2

    failures, shrunk = [], []
    for rid in sorted(RULES):
        allowed = allow.get(rid, {})
        for rel, n in sorted(hits[rid].items()):
            a = allowed.get(rel, 0)
            if n > a:
                failures.append(
                    "%s: %s has %d match(es), allowlist permits %d.\n    %s"
                    % (rid, rel, n, a, RULES[rid]["message"]))
        for rel, a in sorted(allowed.items()):
            n = hits[rid].get(rel, 0)
            if n < a:
                shrunk.append("%s: %s %d -> %d" % (rid, rel, a, n))

    for line in shrunk:
        print("rules_gate: IMPROVED (run --regen to ratchet down): " + line)
    if failures:
        print("rules_gate: FAIL (%d violation(s))" % len(failures))
        for msg in failures:
            print("  " + msg)
        return 1
    print("rules_gate: PASS (%d rules, allowlist %s)"
          % (len(RULES), os.path.basename(allowlist_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
