#!/usr/bin/env python3
"""Automatic validation: original EPOCH SDF files vs converted openPMD series.

No hand-maintained field/species lists: the converter stamps an `sdf_id`
attribute on every record component and `sdf_axis_*` on meshes, so the
script discovers the mapping from the converted data itself and compares
element-by-element against the `sdfr` view of the original.

The conversion is designed lossless (datatypes preserved, no recompression),
so the default check is exact elementwise equality INCLUDING dtype;
`--rtol` loosens only for deliberate dtype experiments.

usage:
    validate_conversion.py 0000.sdf 0001.sdf --series /path/to/opmd
    validate_conversion.py --sdf-dir data --pattern 'normal_*.sdf' \
                           --series /path/to/opmd
exit 0 = all identical, 1 = mismatch, 2 = usage/IO problem
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import sdfr
except ImportError:
    sys.exit("the 'sdfr' module is required: pip install sdfr")
try:
    import openpmd_api as opmd
except ImportError:
    sys.exit("the 'openpmd_api' module is required: pip install openpmd-api")

SCALAR = opmd.Record_Component.SCALAR
# openPMD (record, component) -> SDF block-id suffix per species
PART_MAP = [
    (("position", "x"), "x"), (("position", "y"), "y"),
    (("position", "z"), "z"),
    (("momentum", "x"), "px"), (("momentum", "y"), "py"),
    (("momentum", "z"), "pz"),
    (("weighting", SCALAR), "w"), (("id", SCALAR), "id"),
]


def access_read_only():
    for name in ("READ_ONLY", "read_only"):
        acc = getattr(opmd.Access, name, None)
        if acc is not None:
            return acc
    raise RuntimeError("openpmd_api too old?")


def open_series(path):
    """Series dir -> rebuild the file-based %T pattern automatically."""
    if os.path.isdir(path):
        cands = sorted(f for f in os.listdir(path) if f.endswith(".bp"))
        if not cands:
            sys.exit(f"no .bp found under {path}")
        first = cands[0]
        if "%T" not in first:
            m = re.search(r"(\d+)", first)
            if m:
                first = (first[:m.start()] + f"%0{len(m.group(1))}T"
                         + first[m.end():])
        return opmd.Series(os.path.join(path, first), access_read_only())
    return opmd.Series(path, access_read_only())


def stagger_to_position(stagger, nd):
    if stagger == 0:
        return [0.5] * nd
    if stagger == (1 << nd) - 1:
        return [0.0] * nd
    return [0.5 if (stagger >> d) & 1 else 0.0 for d in range(nd)]


def c_items(c):
    """(name, object) pairs across openpmd-api container versions."""
    if hasattr(c, "items"):
        yield from c.items()
        return
    for item in c:
        if isinstance(item, tuple) and len(item) == 2:
            yield item
        else:
            yield item, c[item]


class Cmp:
    def __init__(self, rtol, verbose):
        self.rtol = rtol
        self.verbose = verbose
        self.n_ok = 0
        self.fails = []

    def eq(self, name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            self.fails.append(f"{name}: shape SDF {a.shape} vs "
                              f"openPMD {b.shape}")
            return
        if a.dtype != b.dtype and self.rtol <= 0.0:
            self.fails.append(f"{name}: dtype SDF {a.dtype} vs "
                              f"openPMD {b.dtype} (losslessness broken)")
            return
        if self.rtol <= 0.0:
            ok = np.array_equal(a, b, equal_nan=(a.dtype.kind == "f"))
        else:
            ok = np.allclose(a, b, rtol=self.rtol, equal_nan=True)
        if not ok:
            diff = np.flatnonzero(a != b)
            ia = np.unravel_index(int(diff[0]), a.shape) if diff.size else ()
            self.fails.append(f"{name}: values differ, "
                              f"{diff.size}/{a.size} elements, first at "
                              f"{ia}: {a[ia]!r} vs {b[ia]!r}")
            return
        self.n_ok += 1
        if self.verbose:
            print(f"    ok {name} {a.shape} {a.dtype}")


def validate_file(sdf_path, series, cmp):
    tag = os.path.basename(sdf_path)
    f = sdfr.read(sdf_path, derived=False)
    ids = f.id_dict
    step = int(f.Header["step"])
    if step not in series.iterations:
        cmp.fails.append(f"{tag}: step {step} absent from series")
        return
    it = series.iterations[step]
    cmp.n_ok += 0  # label helper below

    def scal(name, x, y):
        cmp.eq(name, np.asarray([x]), np.asarray([y]))

    scal(f"step {step} time", float(f.Header["time"]), float(it.time))
    if "dt" in ids:
        scal(f"step {step} dt",
             float(np.asarray(ids["dt"].data).ravel()[0]),
             float(it.dt) if it.contains_attribute("dt") else 1.0)

    # grid axes: sdfr node arrays vs the sdf_axis_* audit attributes
    grid = ids.get("general_grid")
    if grid is not None:
        mesh = next((m for _, m in c_items(it.meshes)
                     if any(m.contains_attribute(f"sdf_axis_{lbl}")
                            for lbl in m.axis_labels)), None)
        if mesh is None:
            cmp.fails.append(f"step {step}: no mesh carries sdf_axis_*")
        else:
            for lbl, axis in zip(mesh.axis_labels, grid.data):
                key = f"sdf_axis_{lbl}"
                if not mesh.contains_attribute(key):
                    cmp.fails.append(f"step {step}: mesh lacks {key}")
                    continue
                cmp.eq(f"step {step} grid axis {lbl}", axis,
                       mesh.get_attribute(key))

    # fields: discovered via the converter's own sdf_id stamps
    futures = []
    for _, rec in c_items(it.meshes):
        ndim = len(rec.grid_global_offset)
        for _, rc in c_items(rec):
            if not rc.contains_attribute("sdf_id"):
                continue
            sdf_id = str(rc.get_attribute("sdf_id"))
            blk = ids.get(sdf_id)
            if blk is None:
                cmp.fails.append(f"step {step}: openPMD claims sdf_id "
                                 f"'{sdf_id}', absent from SDF file")
                continue
            if rec.contains_attribute("position") and \
                    rc.contains_attribute("sdf_stagger"):
                st = int(rc.get_attribute("sdf_stagger"))
                pos = [float(v) for v in rec.get_attribute("position")]
                exp = stagger_to_position(st, ndim)
                if pos != exp:
                    cmp.fails.append(f"step {step} {sdf_id}: position {pos} "
                                     f"!= stagger {st} -> {exp}")
                else:
                    cmp.n_ok += 1
            ext = list(rc.shape)
            futures.append((f"step {step} field {sdf_id}", blk.data,
                            rc.load_chunk([0] * len(ext), ext)))

    # particles: BOTH sides authoritative - anything present in one and
    # unmappable to the other is a FAIL, never a silent skip
    sdf_species = {m.group(1) for bid in ids
                   if (m := re.match(r"^x_(.+)$", bid))}
    pmap = dict(c_items(it.particles))
    for sp in sorted(sdf_species | set(pmap)):
        if sp not in pmap:
            cmp.fails.append(f"step {step}: species '{sp}' in SDF, absent "
                             "from openPMD")
            continue
        if sp not in sdf_species:
            cmp.fails.append(f"step {step}: species '{sp}' in openPMD, no "
                             f"'x_{sp}' block in SDF")
            continue
        e = pmap[sp]
        for (rec_name, comp), suffix in PART_MAP:
            bid = f"{suffix}_{sp}"
            in_sdf = bid in ids
            in_opmd = rec_name in e and comp in e[rec_name]
            if not in_sdf and not in_opmd:
                continue          # legitimately not stored (e.g. no id)
            if in_sdf and not in_opmd:
                cmp.fails.append(f"step {step}: SDF '{bid}' has no openPMD "
                                 f"{rec_name}/{comp}")
                continue
            if in_opmd and not in_sdf:
                cmp.fails.append(f"step {step}: openPMD {rec_name}/{comp} "
                                 f"of '{sp}' has no SDF '{bid}'")
                continue
            n = int(np.asarray(ids[bid].data).size)
            rc = e[rec_name][comp]
            futures.append((f"step {step} particle {bid}", ids[bid].data,
                            rc.load_chunk([0], [n])))

    series.flush()
    for name, sdf_arr, fut in futures:
        cmp.eq(name, sdf_arr, np.asarray(fut))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sdf", nargs="*", help="explicit .sdf files")
    ap.add_argument("--sdf-dir", help="directory of .sdf files")
    ap.add_argument("--pattern", default="*.sdf")
    ap.add_argument("--series", required=True,
                    help="openPMD series dir or %%T pattern")
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="0 (default) = exact incl. dtype; >0 relaxes")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = list(args.sdf)
    if args.sdf_dir:
        files += sorted(glob.glob(os.path.join(args.sdf_dir, args.pattern)))
    if not files:
        ap.error("no .sdf inputs (give files or --sdf-dir)")
    for fpath in files:
        if not os.path.exists(fpath):
            sys.exit(f"no such file: {fpath}")

    series = open_series(args.series)
    cmp = Cmp(args.rtol, args.verbose)
    for fpath in files:
        print(f"validating {fpath}")
        validate_file(fpath, series, cmp)

    print(f"\n{cmp.n_ok} comparisons passed, "
          f"{len(cmp.fails)} failed (rtol {cmp.rtol:g})")
    for d in cmp.fails[:50]:
        print(f"  FAIL {d}")
    if len(cmp.fails) > 50:
        print(f"  ... {len(cmp.fails) - 50} more")
    print("VALIDATION PASSED" if not cmp.fails else "VALIDATION FAILED")
    return 0 if not cmp.fails else 1


if __name__ == "__main__":
    sys.exit(main())
