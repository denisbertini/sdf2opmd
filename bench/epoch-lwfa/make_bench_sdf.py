#!/usr/bin/env python3
"""EPOCH-like synthetic SDF dump set for offline benchmark/analysis testing.

Mimics the structure of the LWFA benchmark deck (see ../README.md):
  - named output blocks -> per timestep two files:
        normal_NNNN.sdf  (grid, rho, ex, ey, bz  + electron species)
        test_NNNN.sdf    (grid, rho              + electront tracer species)
  - Yee-style staggering: rho cell-centred (0), ex/ey/bz face-centred (1/2/4)
  - moving window: grid x-origin shifts +0.05 per step
  - electron id is integer8 (exercises int particle variables)

Values are deterministic functions of (step, index); the analysis scripts
never assume formulas - they compare the SDF side against the converted
openPMD side.

usage: make_bench_sdf.py <outdir> [nsteps]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts"))
from make_fixture import (Block, PLAIN_MESH, PLAIN_VARIABLE, POINT_MESH,
                          POINT_VARIABLE, CONSTANT, DT_INT8, DT_REAL8,
                          NAMEVALUE, plain_mesh_meta, plain_var_meta,
                          point_mesh_meta, point_var_meta, constant_meta,
                          namevalue_meta, write_sdf, f8, i8)

NX, NY, NZ = 17, 13, 11          # grid nodes (cells: 16 x 12 x 10)
NE = 257                         # electrons (odd on purpose)
NT = 9                           # tracer particles


def field(step, i, j, k, off):
    return step * 100.0 + 3.0 * i - 1.7 * j + 0.25 * k + off


def make_grid(step):
    """general mesh, x shifted by the moving window."""
    gx = [0.05 * step + 0.01 * i for i in range(NX)]
    gy = [-0.3 + 0.05 * j for j in range(NY)]
    gz = [-0.2 + 0.04 * k for k in range(NZ)]
    data = (b"".join(f8(v) for v in gx) + b"".join(f8(v) for v in gy)
            + b"".join(f8(v) for v in gz))
    return Block("general_grid", "general mesh", PLAIN_MESH, DT_REAL8, 3,
                 plain_mesh_meta([NX, NY, NZ]), data)


def add_field(blocks, bid, dims, stagger, off, step):
    nx, ny, nz = dims
    vals = b"".join(f8(field(step, i, j, k, off))
                    for k in range(nz) for j in range(ny) for i in range(nx))
    blocks.append(Block(bid, bid, PLAIN_VARIABLE, DT_REAL8, 3,
                        plain_var_meta(list(dims), "general_grid", "",
                                       stagger), vals))


def add_species(blocks, species, npart, step, mass):
    blocks.append(Block(f"grid_{species}", f"grid of {species}", POINT_MESH,
                        DT_INT8, 1, point_mesh_meta(npart, species),
                        b"".join(i8(p % 7) for p in range(npart))))
    params = {
        "x": lambda p: 0.002 * p + 0.01 * step,
        "y": lambda p: -0.001 * p,
        "z": lambda p: 0.0005 * p + 1e-4 * step,
        "px": lambda p: 0.5 + 0.01 * p * step,
        "py": lambda p: -0.25 + 0.002 * p,
        "pz": lambda p: 0.125 + 0.001 * p * step,
        "w": lambda p: 1234.0 + (p % 13),
    }
    for suffix, fn in params.items():
        vals = b"".join(f8(fn(p)) for p in range(npart))
        blocks.append(Block(f"{suffix}_{species}", f"{species} {suffix}",
                            POINT_VARIABLE, DT_REAL8, 1,
                            point_var_meta(npart, f"grid_{species}", species),
                            vals))
    ids = b"".join(i8(100000 + 2 * p) for p in range(npart))
    blocks.append(Block(f"id_{species}", f"{species} id", POINT_VARIABLE,
                        DT_INT8, 1,
                        point_var_meta(npart, f"grid_{species}", species),
                        ids))
    blocks.append(Block(f"mass_{species}", f"{species} mass", CONSTANT,
                        DT_REAL8, 0, constant_meta(mass, DT_REAL8)))


def make_normal(step):
    blocks = [make_grid(step)]
    cells = (NX - 1, NY - 1, NZ - 1)
    add_field(blocks, "rho", cells, 0, 7.0, step)
    add_field(blocks, "ex", (NX, NY - 1, NZ - 1), 1, 0.0, step)
    add_field(blocks, "ey", (NX - 1, NY, NZ - 1), 2, 11.0, step)
    add_field(blocks, "bz", (NX - 1, NY - 1, NZ), 4, -3.0, step)
    blocks.append(Block("dt", "timestep", CONSTANT, DT_REAL8, 0,
                        constant_meta(1e-16 * step, DT_REAL8)))
    blocks.append(Block("nsteps", "steps", CONSTANT, DT_INT8, 0,
                        constant_meta(10 * step, DT_INT8)))
    blocks.append(Block("control", "controls", NAMEVALUE, DT_REAL8, 2,
                        namevalue_meta(["cfl", "mov_maxfrac"], [0.9, 0.02],
                                       DT_REAL8)))
    add_species(blocks, "electron", NE, step, 1.0)
    return blocks


def make_test(step):
    blocks = [make_grid(step)]
    cells = (NX - 1, NY - 1, NZ - 1)
    add_field(blocks, "rho", cells, 0, 7.0, step)
    blocks.append(Block("dt", "timestep", CONSTANT, DT_REAL8, 0,
                        constant_meta(1e-16 * step, DT_REAL8)))
    add_species(blocks, "electront", NT, step, 1.0)
    return blocks


def main():
    out = sys.argv[1]
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs(out, exist_ok=True)
    for s in range(1, nsteps + 1):
        write_sdf(os.path.join(out, f"normal_{s:04d}.sdf"), s,
                  1e-15 * s, make_normal(s))
        write_sdf(os.path.join(out, f"test_{s:04d}.sdf"), s, 1e-15 * s,
                  make_test(s))
    print(f"epoch-like fixtures ({nsteps} steps) written to {out}")


if __name__ == "__main__":
    main()
