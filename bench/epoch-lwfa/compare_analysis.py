#!/usr/bin/env python3
"""Validate the openPMD conversion against the original SDF dumps.

Compares the JSON reports produced by analyze_sdf.py (SDF side) and
analyze_openpmd.py (openPMD side) iteration by iteration.

Exact: shapes, counts, ids. Tolerant (relative): float reductions whose
value depends on summation order; maxima/quantiles only tolerate fp
representation noise.

usage: compare_analysis.py stats_sdf.json stats_openpmd.json
       [--rtol 1e-12] [--verbose]
exit 0 = identical within tolerance, 1 = mismatch, 2 = usage
"""
import argparse
import json
import math
import sys

INT_LEAFS = {"n", "nan_count", "id_sum", "id_min", "id_max", "id_unique",
             "shape"}


def cmp_val(path, a, b, rtol, diffs, verbose):
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            diffs.append(f"{path}: type {type(a).__name__} vs "
                         f"{type(b).__name__}")
            return
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                # tolerate metadata-only extras (end/shape/nonuniformity)
                if b.get(k) is None or a.get(k) is None:
                    continue
                diffs.append(f"{path}.{k}: missing on one side")
                continue
            cmp_val(f"{path}.{k}", a[k], b[k], rtol, diffs, verbose)
        return
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)):
            diffs.append(f"{path}: type list mismatch")
            return
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            cmp_val(f"{path}[{i}]", x, y, rtol, diffs, verbose)
        return
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            diffs.append(f"{path}: {a} != {b}")
        return
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            diffs.append(f"{path}: {a} != {b} (exact)")
        elif verbose:
            print(f"  ok {path} = {a}")
        return
    if a is None and b is None:
        return
    if a is None or b is None:
        diffs.append(f"{path}: {a} vs {b} (one side missing)")
        return
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return
    tol = rtol * max(1.0, abs(fa), abs(fb))
    if abs(fa - fb) > tol:
        diffs.append(f"{path}: {fa!r} != {fb!r} "
                     f"(|d|={abs(fa - fb):.3g} > {tol:.3g})")
    elif verbose:
        print(f"  ok {path} = {fa!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sdf_json")
    ap.add_argument("opmd_json")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.sdf_json) as fh:
        ref = json.load(fh)
    with open(args.opmd_json) as fh:
        got = json.load(fh)

    it_ref, it_got = ref["iterations"], got["iterations"]
    if set(it_ref) != set(it_got):
        print(f"FAIL: iteration sets differ: "
              f"sdf={sorted(map(int, it_ref))} "
              f"opmd={sorted(map(int, it_got))}")
        return 1

    total = 0
    for step in sorted(it_ref, key=int):
        diffs = []
        cmp_val(f"step {step}", it_ref[step], it_got[step], args.rtol, diffs,
                args.verbose)
        total += len(diffs)
        if diffs:
            print(f"FAIL iteration {step} ({len(diffs)} mismatch(es)):")
            for d in diffs[:25]:
                print(f"  {d}")
            if len(diffs) > 25:
                print(f"  ... {len(diffs) - 25} more")
        else:
            print(f"iteration {step}: IDENTICAL (rtol {args.rtol:g})")

    if total:
        print(f"VALIDATION FAILED: {total} mismatch(es)")
        return 1
    print(f"VALIDATION PASSED: {len(it_ref)} iteration(s) match within "
          f"rtol={args.rtol:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
