#!/usr/bin/env python3
"""LWFA benchmark analysis reading the CONVERTED openPMD series.

Uses `openpmd_api` (pip install openpmd-api). Produces the same JSON
structure as analyze_sdf.py from the original SDF dumps, so
compare_analysis.py can validate the conversion.

usage:
    analyze_openpmd.py SERIES_DIR -o stats_openpmd.json \
        [--fields rho,ex,ey,bz] [--species electron,electront]

SERIES_DIR is the output directory passed to sdf2opmd -o (contains
simData_*.bp) or a single *.bp directory.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

from analysis_core import array_stats, grid_stats, particle_stats

try:
    import openpmd_api as opmd
except ImportError:  # pragma: no cover
    sys.exit("the 'openpmd_api' module is required: pip install openpmd-api")


def access_read_only():
    for name in ("READ_ONLY", "read_only"):
        acc = getattr(opmd.Access, name, None)
        if acc is not None:
            return acc
    raise RuntimeError("cannot find read-only access enum in openpmd_api")


def open_series(path):
    """Accept a series directory (auto-detect pattern) or exact pattern."""
    if os.path.isdir(path):
        cands = sorted(f for f in os.listdir(path) if f.endswith(".bp"))
        if not cands:
            sys.exit(f"no .bp found under {path}")
        first = cands[0]
        if "%T" in first or "%0" in first:  # already a pattern
            base = path
        else:  # rebuild file-based pattern: simData_0001.bp -> %04T
            m = re.search(r"(\d+)", first)
            if m:
                digits = m.group(1)
                pat = first[:m.start()] + f"%0{len(digits)}T" + \
                    first[m.end():]
                base = os.path.join(path, pat)
            else:
                base = os.path.join(path, first)
        return opmd.Series(base, access_read_only())
    return opmd.Series(path, access_read_only())


def load_array(rc):
    ext = list(rc.shape)
    fut = rc.load_chunk([0] * len(ext), ext)
    return fut


def iterate_particles(it):
    """Yield (name, species) across openpmd-api binding versions."""
    cont = it.particles
    if hasattr(cont, "items"):
        yield from cont.items()
        return
    for item in cont:
        if isinstance(item, tuple) and len(item) == 2:
            yield item
        else:  # iteration yields names only
            yield item, cont[item]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out", default="stats_openpmd.json")
    ap.add_argument("--fields", default="rho,ex,ey,bz")
    ap.add_argument("--species", default="")
    args = ap.parse_args()

    series = open_series(args.path)
    want_fields = [f for f in args.fields.split(",") if f]
    want_species = [s for s in args.species.split(",") if s]

    # EPOCH field id -> openPMD (record, component); see src/mapping.cpp
    field_map = {"ex": ("E", "x"), "ey": ("E", "y"), "ez": ("E", "z"),
                 "bx": ("B", "x"), "by": ("B", "y"), "bz": ("B", "z"),
                 "jx": ("J", "x"), "jy": ("J", "y"), "jz": ("J", "z")}
    scalar = opmd.Record_Component.SCALAR

    report = {"backend": "openpmd_api", "source": os.path.abspath(args.path),
              "iterations": {}}
    for step, it in series.iterations.items():
        entry = {"time": float(it.time), "dt": float(it.dt)
                 if it.contains_attribute("dt") else None,
                 "grid": None, "fields": {}, "species": {}}

        # grid geometry from any mesh (all share the SDF general_grid)
        for name, mesh in it.meshes.items():
            if mesh.contains_attribute("gridGlobalOffset"):
                entry["grid"] = {
                    "origin": [float(v) for v in mesh.grid_global_offset],
                    "spacing": [float(v) for v in mesh.grid_spacing],
                    "end": None, "shape": None,
                    "nonuniformity": None,
                }
                # exact node arrays were kept as sdf_axis_* attributes
                axes = []
                for lbl in mesh.axis_labels:
                    key = f"sdf_axis_{lbl}"
                    if mesh.contains_attribute(key):
                        axes.append(np.asarray(
                            mesh.get_attribute(key), dtype=np.float64))
                if axes:
                    entry["grid"] = grid_stats(axes)
                break

        # enqueue all loads, flush once per iteration
        futures = {}
        for f in want_fields:
            rec, comp = field_map.get(f, (f, None))
            rc = None
            if rec in it.meshes:
                mrec = it.meshes[rec]
                if comp is not None and comp in mrec:
                    rc = mrec[comp]
                elif scalar in mrec:
                    rc = mrec[scalar]
            elif f in it.meshes and scalar in it.meshes[f]:
                rc = it.meshes[f][scalar]  # scalar record named by SDF id
            if rc is None:
                continue
            futures[f] = (rc, load_array(rc))

        species_data = {}
        for sp, e in iterate_particles(it):
            if want_species and sp not in want_species:
                continue
            needed = [("position", "x"), ("position", "y"), ("position", "z"),
                      ("momentum", "x"), ("momentum", "y"), ("momentum", "z"),
                      ("weighting", scalar)]
            if not all(r in e and c in e[r] for r, c in needed):
                print(f"  note: species '{sp}' incomplete, skipped")
                continue
            arrays = [load_array(e[r][c]) for r, c in needed]
            ids = None
            if "id" in e and scalar in e["id"]:
                ids = load_array(e["id"][scalar])
            species_data[sp] = (arrays, ids)

        series.flush()

        for f, (rc, fut) in futures.items():
            entry["fields"][f] = array_stats(np.asarray(fut))
        for sp, (arrays, ids) in species_data.items():
            pos = [np.asarray(a) for a in arrays[0:3]]
            mom = [np.asarray(a) for a in arrays[3:6]]
            w = np.asarray(arrays[6])
            entry["species"][sp] = particle_stats(
                pos, mom, w, np.asarray(ids) if ids is not None else None)

        report["iterations"][str(step)] = entry
        print(f"step {step}: t={entry['time']:.6e} "
              f"fields={len(entry['fields'])} "
              f"species={ {s: e['n'] for s, e in entry['species'].items()} }")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
