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
SDF_NP = {1: "<i4", 2: "<i8", 3: "<f4", 4: "<f8", 6: "|S1", 7: "|i1"}


def _sdfdump_argv(cand):
    """Resolve a candidate into an argv prefix, or None. Accepts a plain
    binary path, or a command that gets there (e.g. when the helper's MPI
    lives in a container: 'apptainer exec pp.sif /path/to/sdfdump')."""
    import shlex
    import shutil
    if not cand:
        return None
    argv = shlex.split(cand) if any(c.isspace() for c in cand.strip()) \
        else [cand]
    if os.path.exists(argv[0]) or shutil.which(argv[0]):
        return argv
    return None


def find_sdfdump(explicit):
    """Helper (or command prefix) used to read blocks too large for sdfr
    (>2 GiB). Must actually be loadable in this environment: a binary built
    against a container's MPI silently fails elsewhere, so probe it once."""
    import subprocess
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "build", "sdfdump")
    cands = [explicit, os.environ.get("SDFDUMP"), here, "sdfdump"]
    for cand in cands:
        argv = _sdfdump_argv(cand)
        if argv is None:
            continue
        try:
            probe = subprocess.run(argv, capture_output=True, text=True,
                                   timeout=120)
        except Exception as e:                      # not runnable at all
            sys.stderr.write(f"note: sdfdump candidate {cand!r} not usable "
                             f"({e})\n")
            continue
        err = (probe.stderr or "") + (probe.stdout or "")
        if "cannot open shared object file" in err \
                or "error while loading shared libraries" in err:
            sys.stderr.write(
                "note: sdfdump candidate not runnable here "
                f"({cand!r}): {err.strip().splitlines()[-1]}\n"
                "      run the validation inside the environment it was "
                "built in, or set SDFDUMP to a command that works, e.g.\n"
                '      SDFDUMP="apptainer exec $CONT /path/to/build/'
                'sdfdump"\n')
            continue
        return argv                                 # usage/argc exit is fine
    return None


def sdf_array(blk, bid, sdf_path, dump):
    """sdfr's view of a block, with a raw-MPI-IO fallback for blocks whose
    data_length exceeds int32 (sdfr limitation, files with >2 GiB blocks)."""
    try:
        return np.asarray(blk.data)
    except ValueError:
        if dump is None:
            raise RuntimeError(
                f"block '{bid}' exceeds sdfr's 2 GiB limit and no sdfdump "
                "helper found (build sdf2opmd and pass --sdfdump or set "
                "SDFDUMP)")
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".sdfblk", delete=False) as t:
        tmp = t.name
    try:
        r = subprocess.run(list(dump) + [sdf_path, bid, tmp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"sdfdump failed for '{bid}': "
                               f"{(r.stderr or r.stdout).strip()}")
        dtype_id, nbytes = r.stdout.split()
        dtype_id, nbytes = int(dtype_id), int(nbytes)
        if dtype_id not in SDF_NP:
            raise RuntimeError(f"sdfdump: SDF datatype {dtype_id} of "
                               f"'{bid}' has no numpy mapping")
        dt = np.dtype(SDF_NP[dtype_id])
        arr = np.fromfile(tmp, dtype=dt, count=nbytes // dt.itemsize)
        dims = getattr(blk, "dims", None)
        if dims is not None and len(dims) > 1 \
                and int(np.prod([int(d) for d in dims])) == arr.size:
            arr = arr.reshape([int(d) for d in dims], order="F")
        return arr
    finally:
        os.unlink(tmp)


def grid_axes(blk, bid, sdf_path, dump):
    """Per-axis node arrays of a plain-mesh (grid) block. sdfr keeps these in
    separate buffers (contents.grids[i]), so its view is a tuple of arrays;
    the raw fallback reads the concatenated block and splits by dims."""
    try:
        data = blk.data
        if isinstance(data, (tuple, list)):
            return [np.asarray(a) for a in data]
        return [np.asarray(data)]
    except ValueError:
        if dump is None:
            raise RuntimeError(
                f"grid '{bid}' exceeds sdfr's 2 GiB limit and no sdfdump "
                "helper found (build sdf2opmd and pass --sdfdump)")
        raw = sdf_array(blk, bid, sdf_path, dump).ravel()
        out, off = [], 0
        for d in blk.dims:
            d = int(d)
            out.append(raw[off:off + d])
            off += d
        return out


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
        self.skips = []

    def skip(self, name, why):
        """Unchecked data is announced, never silently passed."""
        self.skips.append(f"{name}: {why}")

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


def validate_file(sdf_path, series, cmp, dump=None):
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

    # fields: discovered via the converter's own sdf_id stamps; mapping them
    # also tells us which SDF grid each openPMD mesh was built from
    futures = []
    claimed = set()
    mesh_grid = {}                      # mesh name -> SDF grid block id

    def read_block(blk, bid, kind, axes=False):
        """Block data, or None + FAIL if the reader(s) cannot deliver it."""
        try:
            return grid_axes(blk, bid, sdf_path, dump) if axes \
                else sdf_array(blk, bid, sdf_path, dump)
        except (RuntimeError, ValueError, OSError) as e:
            cmp.fails.append(f"step {step} {kind} '{bid}': {e}")
            return None

    for mname, rec in c_items(it.meshes):
        ndim = len(rec.grid_global_offset)
        for _, rc in c_items(rec):
            if not rc.contains_attribute("sdf_id"):
                cmp.fails.append(f"step {step}: component '{mname}' lacks "
                                 "sdf_id (reconvert with current converter?)")
                continue
            sdf_id = str(rc.get_attribute("sdf_id"))
            blk = ids.get(sdf_id)
            if blk is None:
                cmp.fails.append(f"step {step}: openPMD claims sdf_id "
                                 f"'{sdf_id}', absent from SDF file")
                continue
            claimed.add(sdf_id)
            gid = getattr(blk, "grid_id", None)
            if gid:
                prev = mesh_grid.get(mname)
                if prev is not None and prev != gid:
                    cmp.fails.append(f"step {step}: mesh '{mname}' mixes SDF "
                                     f"grids '{prev}' and '{gid}'")
                mesh_grid[mname] = gid
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
            arr = read_block(blk, sdf_id, "field")
            if arr is None:
                continue
            futures.append((f"step {step} field {sdf_id}", arr,
                            rc.load_chunk([0] * len(ext), ext)))

    def is_plain_mesh(b):
        raw = getattr(b, "_contents", None)
        return raw is not None and int(raw.blocktype) == 1

    def blocktype(b):
        raw = getattr(b, "_contents", None)
        return None if raw is None else int(raw.blocktype)

    # SDF-side completeness: every grid field must have been converted
    for bid, blk in sorted(ids.items()):
        if blocktype(blk) == 3 and bid not in claimed:   # PLAIN_VARIABLE
            cmp.fails.append(f"step {step}: SDF field '{bid}' was not "
                             "converted")

    # grid axes: the SDF node arrays vs the sdf_axis_* audit attributes of
    # the mesh the converter built on that same grid
    grids = {bid for bid, blk in ids.items() if is_plain_mesh(blk)}
    for mname, rec in c_items(it.meshes):
        gid = mesh_grid.get(mname)
        if gid is None and len(grids) == 1:
            gid = next(iter(grids))
        elif gid is None and rec.contains_attribute("sdf_mesh_id"):
            gid = str(rec.get_attribute("sdf_mesh_id"))
        if gid is None:
            cmp.skip(f"step {step} mesh '{mname}' axes",
                     "cannot tell which SDF grid it came from")
            continue
        if gid not in grids:
            cmp.fails.append(f"step {step}: mesh '{mname}' grid '{gid}' is "
                             "not a plain-mesh block in the SDF file")
            continue
        have = [lbl for lbl in rec.axis_labels
                if rec.contains_attribute(f"sdf_axis_{lbl}")]
        if not have:
            cmp.skip(f"step {step} mesh '{mname}' axes",
                     "no sdf_axis_* stamps (grid too large or older "
                     "converter)")
            continue
        axes = read_block(ids[gid], gid, "grid", axes=True)
        if axes is None:
            continue
        for lbl, axis in zip(rec.axis_labels, axes):
            if lbl not in have:
                cmp.fails.append(f"step {step}: mesh '{mname}' lacks "
                                 f"sdf_axis_{lbl}")
                continue
            cmp.eq(f"step {step} grid '{gid}' axis {lbl}", axis,
                   rec.get_attribute(f"sdf_axis_{lbl}"))
    for bid in sorted(grids - set(mesh_grid.values())):
        cmp.skip(f"step {step} grid '{bid}' axes",
                 "no converted field lives on it")

    # particles: driven by the converter's own sdf_id stamps on datasets
    # (immune to EPOCH's truncated species names and subset duplications),
    # plus a completeness check from the SDF side
    claimed_parts = set()
    for sp, e in sorted(c_items(it.particles)):
        for rec_name, rec in c_items(e):
            for comp, rc in c_items(rec):
                if not rc.contains_attribute("sdf_id"):
                    cmp.fails.append(f"step {step}: particle dataset "
                                     f"'{sp}/{rec_name}/{comp}' lacks "
                                     "sdf_id (reconvert with current "
                                     "converter?)")
                    continue
                bid = str(rc.get_attribute("sdf_id"))
                blk = ids.get(bid)
                if blk is None:
                    cmp.fails.append(f"step {step}: openPMD particle "
                                     f"'{sp}/{rec_name}/{comp}' claims "
                                     f"sdf_id '{bid}', absent from SDF")
                    continue
                claimed_parts.add(bid)
                n = int(list(rc.shape)[0])
                arr = read_block(blk, bid, "particle")
                if arr is None:
                    continue
                futures.append((f"step {step} particle {sp}/{bid}", arr,
                                rc.load_chunk([0], [n])))
    for bid in sorted(ids):
        if re.match(r"^(x|y|z|px|py|pz|w|id)_.+$", bid) \
                and bid not in claimed_parts:
            cmp.fails.append(f"step {step}: SDF particle block '{bid}' was "
                             "not converted")

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
    ap.add_argument("--sdfdump",
                    help="path to the sdf2opmd 'sdfdump' helper, used to "
                         "read blocks over sdfr's 2 GiB limit "
                         "(also $SDFDUMP)")
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
    dump = find_sdfdump(args.sdfdump)
    for fpath in files:
        print(f"validating {fpath}")
        validate_file(fpath, series, cmp, dump)

    print(f"\n{cmp.n_ok} comparisons passed, "
          f"{len(cmp.fails)} failed, {len(cmp.skips)} skipped "
          f"(rtol {cmp.rtol:g})")
    for d in cmp.skips[:20]:
        print(f"  SKIP {d}")
    for d in cmp.fails[:50]:
        print(f"  FAIL {d}")
    if len(cmp.fails) > 50:
        print(f"  ... {len(cmp.fails) - 50} more")
    print("VALIDATION PASSED" if not cmp.fails else "VALIDATION FAILED")
    return 0 if not cmp.fails else 1


if __name__ == "__main__":
    sys.exit(main())
