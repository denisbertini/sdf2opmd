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
