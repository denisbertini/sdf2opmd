#!/usr/bin/env python3
"""LWFA benchmark analysis reading the ORIGINAL EPOCH SDF dumps.

Uses the pure-python EPOCH SDF reader `sdfr` (pip install sdfr). Emits a
JSON report of deterministic per-iteration statistics (grid geometry,
field stats, particle distributions per species) directly comparable with
analyze_openpmd.py output.

usage:
    analyze_sdf.py SDF_DIR -o stats_sdf.json [--pattern 'normal_*.sdf']
                     [--fields rho,ex,ey,bz] [--species electron,electront]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from analysis_core import array_stats, grid_stats, particle_stats

try:
    import sdfr
except ImportError:  # pragma: no cover
    sys.exit("the 'sdfr' module is required: pip install sdfr")

PARTICLE_VARS = {"x", "y", "z", "px", "py", "pz", "w", "id"}


def discover_species(id_dict):
    species = {}
    for bid in id_dict:
        m = re.match(r"^(?P<var>\w+)_(?P<sp>[a-zA-Z]\w*)$", bid)
        if m and m.group("var") in PARTICLE_VARS:
            species.setdefault(m.group("sp"), set()).add(m.group("var"))
    return {sp: vs for sp, vs in species.items() if {"x", "w"} <= vs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="directory containing .sdf files")
    ap.add_argument("-o", "--out", default="stats_sdf.json")
    ap.add_argument("--pattern", default="*.sdf")
    ap.add_argument("--fields", default="rho,ex,ey,bz")
    ap.add_argument("--species", default="",
                    help="limit species (default: all discovered)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.path, args.pattern)))
    if not files:
        sys.exit(f"no files matching {args.path}/{args.pattern}")

    want_fields = [f for f in args.fields.split(",") if f]
    want_species = [s for s in args.species.split(",") if s]

    report = {"backend": "sdfr", "source": os.path.abspath(args.path),
              "iterations": {}}
    for fname in files:
        f = sdfr.read(fname, derived=False)
        ids = f.id_dict
        step = int(f.Header["step"])
        entry = {"time": float(f.Header["time"]), "dt": None, "grid": None,
                 "fields": {}, "species": {}}

        grid = ids.get("general_grid")
        if grid is not None:
            axes = [np.asarray(a, dtype=np.float64) for a in grid.data]
            entry["grid"] = grid_stats(axes)

        for fld in want_fields:
            if fld in ids:
                entry["fields"][fld] = array_stats(ids[fld].data)

        if "dt" in ids:
            entry["dt"] = float(np.asarray(ids["dt"].data).ravel()[0])

        species = discover_species(ids)
        for sp in sorted(species):
            if want_species and sp not in want_species:
                continue
            vs = species[sp]

            def var(v):
                return np.asarray(ids[f"{v}_{sp}"].data)

            entry["species"][sp] = particle_stats(
                [var("x"), var("y"), var("z")],
                [var("px"), var("py"), var("pz")] if "px" in vs else None,
                var("w"), var("id") if "id" in vs else None)

        if str(step) in report["iterations"]:
            sys.exit(f"duplicate step {step} across files "
                     f"({args.pattern}); convert output-block files "
                     f"separately")
        report["iterations"][str(step)] = entry
        print(f"step {step}: t={entry['time']:.6e} "
              f"fields={len(entry['fields'])} "
              f"species={ {s: e['n'] for s, e in entry['species'].items()} }")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
