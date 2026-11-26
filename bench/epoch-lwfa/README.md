# EPOCH LWFA benchmark: run -> convert -> validate

Full-workflow benchmark for `sdf2opmd` on a real EPOCH 3D LWFA deck: run
the simulation on SLURM (containerized EPOCH), convert the SDF dumps to
openPMD/BP5, then **validate the conversion by running the same physics
analysis twice** - once directly on the SDF files (`sdfr`), once on the
converted data (`openpmd_api`) - and comparing the results.

## Contents

| file | role |
|---|---|
| `decks/lwfa.deck` | the original LWFA deck (Stephan Kuschel) |
| `decks/lwfa_bench.deck` | I/O-benchmark scaling: `t_end=200 fs`, 1 npc, snapshots every 50 fs |
| `epoch_job.sh` | sbatch wrapper (step 1), reproduces your Orion submit pattern |
| `run_epoch.sh` | the run-file: `apptainer exec $CONT epoch3d` |
| `convert_job.sh` | step 2: parallel `sdf2opmd` conversion + timings |
| `run_analysis.sh` | step 3: both analyses + strict comparison (sbatchable) |
| `analyze_sdf.py` | analysis straight on SDF via the EPOCH `sdfr` reader |
| `analyze_openpmd.py` | identical analysis on the converted openPMD series |
| `compare_analysis.py` | iteration-by-iteration JSON comparison, rtol 1e-12 |
| `analysis_core.py` | the shared statistic definitions both readers feed |
| `make_bench_sdf.py` | synthetic EPOCH-like dump set for offline testing |
| `run_local_smoke.sh` | offline end-to-end: fixture -> convert -> compare |
| `build_sdf2opmd_cluster.sh` | build converter + ADIOS2/openPMD on the cluster |

## Workflow (cluster)

```sh
# once: build the converter (inside the EPOCH container for MPI matching)
srun --exclusive --pty apptainer exec $CONT ./build_sdf2opmd_cluster.sh

# 1. physics run (Dumps -> ./data/)
./epoch_job.sh                       # or DECK=... NODES=8 TIME=08:00:00 ...

# 2. conversion benchmark -> $OUT/{simData,simDataTest}/simData_%04T.bp
sbatch --export=ALL,OUT_DIR=/lustre/.../out ./convert_job.sh ./data /lustre/.../out

# 3. validation: same analysis on SDF vs openPMD, must match
sbatch ./run_analysis.sh ./data /lustre/.../out

# 3b. and/or the element-wise gate (bit-exact, no field lists to maintain):
python3 validate_conversion.py --sdf-dir ./data --pattern 'normal_*.sdf' \
    --series /lustre/.../out/simData
```

## validate_conversion.py — element-wise gate

One command, exit code 0/1, no hand-maintained field/species lists: it reads
the mapping from the converter's own audit stamps (`sdf_id` on every record
component, `sdf_axis_*`/`sdf_mesh_id` on meshes) and then compares, per file,
`time`/`dt`, the grid node axes, the stagger -> `position` invariant, every
field and every particle variable, element by element. The conversion is
lossless, so the default is **exact equality including dtype**; `--rtol`
loosens only for deliberate dtype experiments. Anything the script cannot
check is reported as `SKIP <what>: <why>` in the summary, never passed
silently.

Blocks larger than 2 GiB (EPOCH dumps with >250M particles per species)
overflow a 32-bit length inside the pure-python `sdfr` reader, which is a
reader limitation, not a conversion problem. For those files point the
validator at the `sdfdump` helper of this repo (same SDF_C + MPI-IO path as
the converter):

```sh
cmake --build build --target sdfdump
export SDFDUMP=$PWD/build/sdfdump        # or --sdfdump path/to/sdfdump
```

Each step's log records the numbers to graph: per-step conversion
`[metadata | data reads | engine flush]` and wall time from `time`, output
volume from `du`; the analysis logs `VALIDATION PASSED/FAILED`.

## Why two series

The deck has two named output blocks (`normal`, `testparticle`); EPOCH
writes them as separate files (`normal_0001.sdf`, `test_0001.sdf`) that
share step numbers. `sdf2opmd` therefore converts them into two openPMD
series via `--pattern` (`-g`).

## What the validation compares (per iteration)

- header/iteration metadata: `time`, `dt`
- grid: node-array origin/spacing/uniformity per axis (moving-window
  shifts included; the full axes are also carried as `sdf_axis_*` attrs)
- fields `rho, ex, ey, bz`: shape, sum, absmax(+index), sum of squares,
  NaN count - on the stored doubles, bit-lossless through BP5
- every particle species: count, weight sum, weighted centroid, mean/max/
  p95 gamma, momentum-derived stats, id sum/min/max/uniqueness (covers the
  int8 `id` records), spatial extent

Exact match is expected for integers and shapes; float reductions allow
rtol 1e-12 only because Fortran- vs C-order summation can differ in the
last ulp. Anything coarser = conversion bug, please report with the two
JSONs.

## Record model note (Yee staggering)

openPMD records share extent/position over components. EPOCH's Yee
components (`ex` on x-faces, `ey` on y-faces, `bz` on z-faces...) have
different extents *and* positions, so they become scalar records named by
their SDF id (`ex`, `ey`, `bz`), each with the exact grid `position`
derived from the SDF stagger. Components that *do* agree (dims + stagger)
are still combined into vector records (`E/x, E/y, ...`).

## Offline smoke test (no cluster needed)

```sh
VENV=/path/to/venv ./run_local_smoke.sh
# venv needs: numpy sdfr openpmd-api
```

Generates an EPOCH-like 3-step dump set (Yee staggering, moving window,
int8 particle ids, both output-block files), converts it with 3 ranks and
requires the two analyses to compare identical.

## Memory sizing (ranks and mem-per-cpu)

One file is in flight at a time, and every rank holds its ~`1/R` share of
that file's data until the batched engine flush (fields batched, particles
batched per species). So the peak is bounded by the **largest single file**,
never by the series length:

```
peak RSS/rank  ~=  C  +  k * S_max / R
   C     ~ 0.1-0.5 GiB fixed floor (openPMD + ADIOS2 + MPI runtime)
   k     ~ 1.1-1.5 (transpose tiles, BP5 serialization, allocator slack)
   S_max bytes of the biggest SDF file (du -s; ~= sum of r8 fields +
         npart * nvars * 8), R = ranks
```

Consequences: doubling R halves the payload term (memory is *not* why you
avoid many ranks); `mem-per-cpu` must cover `C + k*S_max/R` (Orion: 32
ranks x 1960 MiB ~= 61 GiB usable per node).

The converter now tells you both sides of this directly:
* before each file: `~X MiB/rank at R ranks` (payload estimate `S/R`) - if
  this exceeds your per-rank budget, raise the rank count;
* at the end: `peak RSS/rank ... (max over R rank(s), mean ...)` - the true
  high-water mark including engine and runtime.

Sizing recipe: convert ONE file at your intended rank count (seconds of
work), read the `memory:` line, set `--mem-per-cpu` to ~1.3x that. Then the
rank choice is purely about I/O: as many ranks as saturate your Lustre
read+write (see contention section), not more.

## Lustre read+write contention

Conversion reads SDF and writes BP5 roughly simultaneously, ~1 byte out per
byte in (datatypes are preserved). The shape of that traffic is already
OST-friendly - big contiguous MPI-IO read extents in, and per iteration only
a few large aggregated BP5 streams (one `.dat` + one `.md` by default), not
dozens of small writers - but read and write colliding *on the same OSTs*
is where throughput collapses. (Terms: the **bbstage engine** of
`BBSTAGE=1` below is yours - a fork of ADIOS2; LRZ's **bbview/bbstage**
archival tools are a different thing and stage Lustre->EOS.) Order of
defenses:

1. **Separate filesystems**: read from `/project`, write `OUT_DIR` under
   `/work` (distinct Lustre instances on Orion); LRZ `bbstage`/bbview
   archives to EOS later.
2. **Disjoint OSTs on one FS**: `STRIPE=<count> ./convert_job.sh ...`
   (runs `lfs setstripe` on the empty output dir; iterations created under
   it inherit the layout).
3. **Single-node NVMe fast lane** for series that fit: `NODES=1`,
   `OUT_DIR=/ssd/$USER/$SLURM_JOB_ID/out`, then one `xrdcp`/rsync to EOS.
   Multi-node NVMe output is invalid (one `.md` per iteration needs a
   shared filesystem).
4. **Engine shaping** via `ENGINE_JSON` (forwarded verbatim into ADIOS2
   engine params): `SubFiles`/`NumSubFiles` to split or merge data files,
   `Collectives`, `ThreadsPerBatch`. Note: plain BP5 cannot write *directly*
   to EOS - ADIOS2 has no `DataStorage`/XRootD write path in 2.10/2.12; EOS
   archiving stays a post-hoc `bbview`/`xrdcp` step, off the critical path.
5. **The bbstage engine** (`BBSTAGE=1 ./convert_job.sh ...`): writes each
   iteration to node-local NVMe staging instantly and drains it to
   `OUT_DIR` from background threads *while the converter reads the next
   SDF file* - read and write phases pipeline instead of serializing, and
   Lustre stalls no longer halt the converter. Selected through the
   openPMD→ADIOS2 engine channel
   (`type: bbstage, pretend_engine: bp5`), so the output is standard BP5;
   this also rescues the multi-node NVMe idea: staging is local, the
   finished series always lands complete on `OUT_DIR`. `CloseMode=durable`
   (default here) means everything is on `OUT_DIR` when the converter
   exits; `CloseMode=staged` hands the tail to the engine's exit
   finalizer. Both modes verified byte-exact against the SDF-side
   analysis. Requires ADIOS2 from the [bbstage
   fork](https://github.com/denisbertini/ADIOS2) - the default of
   `build_sdf2opmd_cluster.sh` (upstream ADIOS2 simply lacks the engine).
   Drain capacity rule of thumb: the converter demands ~1 shared-FS read +
   1 write stream per node, well under the `ranks x store-rate` ceiling.

And the baseline mitigation: the converter is **offline** - it never runs
while EPOCH is dumping, and the rank count is a pressure dial (8 ranks ≈
one bandwidth stream; 128 buys per-file latency overlap, not more Lustre
politeness).
