# sdf2opmd

Fast, dimension-agnostic converter from **SDF** ([Self-Describing Format],
written by [EPOCH]/FliteCode) to **openPMD** (ADIOS2 BP5), built on the
[SDF C library] for reading and [openPMD-api] for writing.

Successor of [mpi-sdf2opmd]: no 1d/2d/3d code trees, no Fortran reader,
no merger step, no scalar transpose loops.

## Design

```
sdf_file.hpp   RAII over sdf_file_t + raw MPI-IO read primitives
blockmap.*     one-pass blocklist classification (fields/meshes/species/meta)
mapping.*      EPOCH/SDF naming -> openPMD object model + units/stagger
convert.*      per-file pipeline: SDF iteration -> openPMD iteration
main.cpp       CLI, file queue, timings
```

Fast path per input file (all MPI-IO collective over `MPI_COMM_WORLD`):

1. `sdf_open` + `sdf_read_blocklist`: header + summary parsed **once**,
   blocks hash-indexed by the SDF_C library.
2. Field data is read through Fortran-ordered MPI-IO **subarray views**
   (derived filetype, contiguous base-type buffer read: the monotonic,
   well-trodden ROMIO fast path) and reordered to the C order openPMD
   expects by a **cache-blocked per-rank transpose**. A "free transpose"
   by reading C-order through a reversed filetype is impossible: MPI-IO
   requires filetype displacements to be monotonically non-decreasing
   (ROMIO's `iobadoverlap` check).
3. Read buffers are moved into openPMD (`storeChunk(unique_ptr)`), which
   defers serialization to the BP5 engine; flushes are batched: one
   collective flush after all fields, one per species.
4. Datatypes are preserved (r4 stays r4, int4/int8 preserved).
5. Each input file becomes one file-based openPMD iteration
   (`simData_%04T.bp`); nothing to merge.

## SDF coverage (v1)

| SDF block | openPMD output |
|---|---|
| `plain_mesh` | mesh axes: `gridSpacing`, `gridGlobalOffset`, axis labels, full axes as `sdf_axis_*` attributes |
| `plain_variable` | vector meshes `E/B/J` (from `ex..jz`), scalar meshes otherwise; `sdf_id`, `sdf_units`, `sdf_stagger` attributes; standard `position` derived from stagger when consistent |
| `point_mesh` / `point_variable` | particle records `position`, `momentum`, `velocity`, `weighting`, custom scalar records otherwise; species via SDF species id |
| `constant` / `namevalue` | iteration attributes (`sdf_*`); the `dt` constant becomes the standard iteration `dt`; header `step`/`time` become iteration index/`time` |
| stitched/contiguous tensors, materials, species, run_info, cpu_split, stations, datablocks, source | reported and skipped (v1 scope), never fatal |

Particle data is written as one contiguous chunk per rank (valid openPMD);
grid-level `particlePatches` reconstruction from the SDF point-mesh indices
is on the roadmap below.

## Build

Dependencies: MPI (Intel/OpenMPI), a pre-built MPI-enabled ADIOS2, CMake >=
3.22, Python 3 for the fixtures. SDF_C comes in as a git submodule.

```sh
git submodule update --init          # fetch extern/sdf (epochpic/SDF_C)
./scripts/build-openpmd.sh           # builds openPMD-api (BP5-only) into deps/
                                     #   ADIOS2_DIR=<adios2-build-tree> ...
cmake -B build -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
      -DopenPMD_DIR=deps/openpmd/lib/cmake/openPMD
cmake --build build -j
```

## Usage

```sh
# convert every SDF dump in a directory, 48 ranks
mpirun -n 48 ./build/sdf2opmd -p /lustre/epoch_run -o /eos/archive/run \
    --name 'simData_%04T.bp'

# specific files, subset of fields/species
mpirun -n 8 ./build/sdf2opmd -p run -f 0100.sdf:0200.sdf \
    -m ex:ey:bz -s el --quiet

# what is in these files?
./build/sdf2opmd --info -p run -f 0100.sdf
```

Output is a file-based openPMD series; browse with `openpmd-ls`,
`openpmd_api` Python, or (for EOS-resident data) the bbstage/bbview
ADIOS2 staging engines.

## Tests

```sh
ctest --test-dir build            # generates SDF fixtures with an
                                  # independent pure-Python SDF writer,
                                  # converts (3 ranks), reads back and
                                  # validates fields, particles, geometry
```

## Roadmap

- Reconstruct grid-level `particlePatches` from point-mesh grid indices.
- `cpu_split`-aware read decomposition; multi-group mode (N ranks per
  file) with disjoint output directories for many-file batches.
- Stitched/contiguous tensor and material blocks.
- Optional `--dtype f8` upconversion and `--downsample` filters.

[Self-Describing Format]: https://wwarwick.github.io/www-sdf/
[SDF C library]: https://github.com/epochpic/SDF_C
[EPOCH]: https://github.com/epochpic/epoch
[openPMD-api]: https://github.com/openPMD/openPMD-api
[mpi-sdf2opmd]: https://github.com/denisbertini/mpi-sdf2opmd
