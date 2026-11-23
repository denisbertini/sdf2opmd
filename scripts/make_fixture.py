#!/usr/bin/env python3
"""Generate synthetic SDF files for sdf2opmd testing.

An independent, from-scratch writer for the Self-Describing Format (version
1.4, as implemented by epochpic/SDF_C and written by EPOCH), used here as a
test fixture so the converter can be validated without an EPOCH run.

Usage:
    make_fixture.py <outdir>            # writes 0001.sdf 0002.sdf (3D)
                                        # and 1001.sdf (2D)

Block metadata byte layouts match the SDF documentation and the SDF_C
reader: fixed 200-byte block headers, then per-type inline metadata.
"""
import os
import struct
import sys

IDLEN = 32
NAMELEN = 128
BLOCKHDR = 200

# blocktype enum (SDF 1.4)
PLAIN_MESH = 1
POINT_MESH = 2
PLAIN_VARIABLE = 3
POINT_VARIABLE = 4
CONSTANT = 5
NAMEVALUE = 9 + 5  # see below

# actual enum order from sdf.h:
#  MESH_PLAIN=1 ... we use the values of enum sdf_blocktype:
PLAIN_MESH = 1
POINT_MESH = 2
PLAIN_VARIABLE = 3
POINT_VARIABLE = 4
CONSTANT = 5
ARRAY = 6
RUN_INFO = 7
SOURCE = 8
STITCHED_TENSOR = 9
STITCHED_MATERIAL = 10
STITCHED_MATVAR = 11
STITCHED_SPECIES = 12
SPECIES = 13
PLAIN_DERIVED = 14
POINT_DERIVED = 15
CONTIGUOUS_TENSOR = 16
CONTIGUOUS_MATERIAL = 17
CONTIGUOUS_MATVAR = 18
CONTIGUOUS_SPECIES = 19
CPU_SPLIT = 20
STITCHED_OBSTACLE_GROUP = 21
UNSTRUCTURED_MESH = 22
STITCHED = 23
CONTIGUOUS = 24
LAGRANGIAN_MESH = 25
STATION = 26
STATION_DERIVED = 27
DATABLOCK = 28
NAMEVALUE = 29

# datatype enum
DT_INT4, DT_INT8, DT_REAL4, DT_REAL8 = 1, 2, 3, 4
GEOM_CART = 2
SIZES = {DT_INT4: 4, DT_INT8: 8, DT_REAL4: 4, DT_REAL8: 8}


def pad(s, n):
    b = s.encode("ascii")
    assert len(b) <= n, f"name too long: {s}"
    return b + b" " * (n - len(b))


def f8(x):
    return struct.pack("<d", x)


def i4(x):
    return struct.pack("<i", x)


def i8(x):
    return struct.pack("<q", x)


class Block:
    def __init__(self, bid, name, blocktype, datatype, ndims, meta, data=b""):
        self.id = bid
        self.name = name
        self.blocktype = blocktype
        self.datatype = datatype
        self.ndims = ndims
        self.meta = meta
        self.data = data
        self.loc = 0          # location of the metadata copy
        self.data_loc = 0
        self.next = 0

    def fixed(self, next_loc):
        dloc = self.data_loc if self.data else 0
        return (
            i8(next_loc)
            + i8(dloc)
            + pad(self.id, IDLEN)
            + i8(len(self.data))
            + i4(self.blocktype)
            + i4(self.datatype)
            + i4(self.ndims)
            + pad(self.name, NAMELEN)
            + i4(len(self.meta))
        )

    def inline(self, next_loc):
        assert len(self.fixed(next_loc)) == BLOCKHDR
        return self.fixed(next_loc) + self.meta


# ---------------------------------------------------------------------------
# per-type metadata builders
# ---------------------------------------------------------------------------

def plain_mesh_meta(dims, labels=("x", "y", "z"), units=("m", "m", "m"),
                    extents=None):
    nd = len(dims)
    if extents is None:
        extents = [0.0] * nd + [1.0] * nd
    m = b"".join(f8(1.0) for _ in range(nd))            # mults
    m += b"".join(pad(l, IDLEN) for l in labels[:nd])
    m += b"".join(pad(u, IDLEN) for u in units[:nd])
    m += i4(GEOM_CART)
    m += b"".join(f8(e) for e in extents[: 2 * nd])
    m += b"".join(i4(d) for d in dims)
    return m


def plain_var_meta(dims, mesh, units="", stagger=0):
    nd = len(dims)
    m = f8(1.0) + pad(units, IDLEN) + pad(mesh, IDLEN)
    m += b"".join(i4(d) for d in dims)
    m += i4(stagger)
    return m


def point_mesh_meta(npoints, species, ndims=1):
    m = b"".join(f8(1.0) for _ in range(ndims))
    m += b"".join(pad("x", IDLEN) for _ in range(ndims))
    m += b"".join(pad("", IDLEN) for _ in range(ndims))
    m += i4(GEOM_CART)
    m += b"".join(f8(0.0) for _ in range(2 * ndims))
    m += i8(npoints)
    m += pad(species, IDLEN)
    return m


def point_var_meta(npoints, mesh, species, units=""):
    m = f8(1.0) + pad(units, IDLEN) + pad(mesh, IDLEN)
    m += i8(npoints)
    m += pad(species, IDLEN)
    return m


def constant_meta(value, datatype):
    if datatype == DT_REAL8:
        return f8(value)
    if datatype == DT_INT4:
        return i4(value)
    if datatype == DT_INT8:
        return i8(value)
    raise ValueError(datatype)


def namevalue_meta(names, values, datatype):
    m = b"".join(pad(n, NAMELEN) for n in names)
    if datatype == DT_REAL8:
        m += b"".join(f8(v) for v in values)
    elif datatype == DT_INT4:
        m += b"".join(i4(v) for v in values)
    elif datatype == DT_REAL4:
        m += b"".join(struct.pack("<f", v) for v in values)
    else:
        raise ValueError(datatype)
    return m


# ---------------------------------------------------------------------------
# file writer
# ---------------------------------------------------------------------------

def write_sdf(path, step, time, blocks, with_summary=True):
    header_len = 112
    # --- assign metadata region offsets (all blocks back to back) ---
    meta_off = header_len
    for b in blocks:
        b.loc = meta_off
        meta_off += BLOCKHDR + len(b.meta)
    data_off = meta_off
    for b in blocks:
        if b.data:
            b.data_loc = data_off
            data_off += len(b.data)

    summary_off = data_off if with_summary else 0
    summary_len = 0
    if with_summary:
        sum_off = summary_off
        for b in blocks:
            summary_len += BLOCKHDR + len(b.meta)
            b_s_next = sum_off + BLOCKHDR + len(b.meta)
            sum_off = b_s_next
        # last summary block's next points just past the summary
        summary_total = summary_len

    hdr = (
        b"SDF1"
        + b"\x0f\x0e\x02\x01"        # endianness marker (fixed byte order)
        + i4(1)                      # format version
        + i4(4)                      # format revision (1.4: species fields)
        + pad("epoch_fixture", IDLEN)
        + i8(header_len)             # first_block_location
        + i8(summary_off)            # summary_location
        + i4(summary_len)            # summary_size
        + i4(len(blocks))            # nblocks
        + i4(BLOCKHDR)               # block_header_length
        + i4(step)
        + f8(time)
        + i4(42) + i4(0)             # jobid1, jobid2
        + i4(NAMELEN)                # string_length
        + i4(100)                    # code_io_version
        + b"\x00" * 3                # restart/other_domains/station flags
    )
    hdr += b"\x00" * (header_len - len(hdr))
    assert len(hdr) == header_len

    out = bytearray(hdr)

    # inline metadata chain
    for idx, b in enumerate(blocks):
        nxt = blocks[idx + 1].loc if idx + 1 < len(blocks) else header_len
        out += b.inline(nxt)
    # data
    for b in blocks:
        if b.data:
            assert b.data_loc == len(out)
            out += b.data
    # summary copy
    if with_summary:
        sum_pos = summary_off
        for b in blocks:
            nxt = sum_pos + BLOCKHDR + len(b.meta)
            out += b.inline(nxt)
            sum_pos = nxt
        assert sum_pos == summary_off + summary_len

    with open(path, "wb") as fh:
        fh.write(out)


# ---------------------------------------------------------------------------
# fixture content
# ---------------------------------------------------------------------------

def field_value(step, i, j, k):
    return step * 1000.0 + i + 2.0 * j + 3.0 * k


def make_3d(step):
    gx = [0.1 * i for i in range(9)]
    gy = [1.0 + 0.2 * j for j in range(7)]
    gz = [-2.0 + 0.3 * k for k in range(6)]

    blocks = []
    # grid (node arrays: 9x7x6 nodes), mesh dims = node counts
    mesh_data = (
        b"".join(f8(v) for v in gx)
        + b"".join(f8(v) for v in gy)
        + b"".join(f8(v) for v in gz)
    )
    blocks.append(
        Block("general_grid", "general mesh", PLAIN_MESH, DT_REAL8, 3,
              plain_mesh_meta([9, 7, 6]), mesh_data))

    for bid, stagger, off in (("ex", 1, 0), ("ey", 2, 1000), ("bz", 4, 100)):
        vals = bytearray()
        # Fortran order: k outer, j, i fastest -- but dims are (8,6,5) cells
        for k in range(5):
            for j in range(6):
                for i in range(8):
                    vals += f8(field_value(step, i, j, k) + off)
        blocks.append(
            Block(bid, bid.upper() + " field", PLAIN_VARIABLE, DT_REAL8, 3,
                  plain_var_meta([8, 6, 5], "general_grid", "V/m", stagger),
                  bytes(vals)))

    blocks.append(Block("dt", "timestep", CONSTANT, DT_REAL8, 0,
                        constant_meta(5e-17 / step, DT_REAL8)))
    blocks.append(Block("nsteps", "number of steps", CONSTANT, DT_INT4, 0,
                        constant_meta(10 * step, DT_INT4)))
    blocks.append(Block("control", "run controls", NAMEVALUE, DT_REAL8, 2,
                        namevalue_meta(["cfl", "mov_maxfrac"],
                                       [0.95, 0.01], DT_REAL8)))

    # particles: species "el", npart = 103 (odd on purpose)
    npart = 103
    grid_idx = []
    n = 0
    g = 0
    while n < npart:
        c = min(11, npart - n)
        grid_idx += [g] * c
        n += c
        g += 1
    assert len(grid_idx) == npart
    blocks.append(
        Block("grid_el", "grid of electrons", POINT_MESH, DT_INT8, 1,
              point_mesh_meta(npart, "el"),
              b"".join(i8(v) for v in grid_idx)))

    def pvar(suffix, fn):
        vals = b"".join(f8(fn(p)) for p in range(npart))
        blocks.append(
            Block(f"{suffix}_el", f"electron {suffix}", POINT_VARIABLE,
                  DT_REAL8, 1,
                  point_var_meta(npart, "grid_el", "el"), vals))

    pvar("x", lambda p: 0.5 * p + step)
    pvar("y", lambda p: -1.0 * p)
    pvar("z", lambda p: p / 8.0)
    pvar("px", lambda p: p + 1.0)
    pvar("py", lambda p: p + 2.0)
    pvar("pz", lambda p: p + 3.0)
    pvar("w", lambda p: 4.0)
    return blocks


def make_2d(step):
    gx = [0.25 * i for i in range(5)]
    gy = [10.0 + 1.0 * j for j in range(4)]
    blocks = []
    blocks.append(
        Block("general_grid", "general mesh", PLAIN_MESH, DT_REAL8, 2,
              plain_mesh_meta([5, 4]),
              b"".join(f8(v) for v in gx) + b"".join(f8(v) for v in gy)))
    for bid in ("ex", "ey"):
        vals = b"".join(f8(step * 10.0 + i + 2.0 * j)
                        for j in range(3) for i in range(4))
        blocks.append(
            Block(bid, bid.upper(), PLAIN_VARIABLE, DT_REAL8, 2,
                  plain_var_meta([4, 3], "general_grid"), vals))
    blocks.append(Block("dt", "timestep", CONSTANT, DT_REAL8, 0,
                        constant_meta(1e-18, DT_REAL8)))
    npart = 7
    blocks.append(
        Block("grid_ion", "grid of ions", POINT_MESH, DT_INT8, 1,
              point_mesh_meta(npart, "ion"), b"".join(i8(0) for _ in range(npart))))
    for suffix, fn in (("x", lambda p: 1.5 * p), ("w", lambda p: 1.0)):
        blocks.append(
            Block(f"{suffix}_ion", f"ion {suffix}", POINT_VARIABLE, DT_REAL8,
                  1, point_var_meta(npart, "grid_ion", "ion"),
                  b"".join(f8(fn(p)) for p in range(npart))))
    return blocks


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    # 3D: two steps with summary; plus one no-summary file to test the
    # inline-metadata fallback.
    write_sdf(os.path.join(out, "0001.sdf"), 1, 1.5e-15, make_3d(1))
    write_sdf(os.path.join(out, "0002.sdf"), 2, 3.0e-15, make_3d(2),
              with_summary=False)
    write_sdf(os.path.join(out, "1001.sdf"), 1001, 0.25, make_2d(1001))
    print(f"fixtures written to {out}")


if __name__ == "__main__":
    main()
