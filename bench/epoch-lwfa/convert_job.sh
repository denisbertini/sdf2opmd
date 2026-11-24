#!/bin/bash
#SBATCH --job-name=sdf2opmd
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --no-requeue
#SBATCH --mem-per-cpu=1960
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH -o %j.convert.out.log
#SBATCH -e %j.convert.err.log
# Step 2: benchmark the SDF -> openPMD (BP5) conversion.
#
# Converts each EPOCH output-block series into its own file-based openPMD
# series (EPOCH gives every named output block its own files, which share
# step numbers between blocks):
#
#   normal_*.sdf  ->  $OUT_DIR/simData/simData_%04T.bp
#   test_*.sdf    ->  $OUT_DIR/simDataTest/simData_%04T.bp
#
# Set SDF2OPMD + matching MPI/ADIOS2 environment first (see
# build_sdf2opmd_cluster.sh; on dev/login nodes remember
# LD_LIBRARY_PATH=/usr/lib64 for libfabric >= FABRIC_1.10).
#
#   NODES=2 RANKS=64 ./convert_job.sh /path/to/dumps /path/to/output
#
# ---- where to write (Lustre read+write contention) ------------------------
# Conversion reads SDF and writes BP5 at ~the same time; keep those two
# streams off the same OSTs:
#   * best:      dumps under /project, OUT_DIR under /work - separate Lustre
#                instances, zero contention by construction; bbstage later
#   * same FS:   STRIPE=<n> pins OUT_DIR to a (fresh) OST stripe set so
#                reads and writes never share targets (Lustre inheritance:
#                set it while OUT_DIR is empty)
#   * small runs: single node + node-local NVMe (NODES=1, OUT_DIR under
#                /ssd/$USER, then xrdcp/rsync to EOS). NEVER multi-node
#                NVMe: BP5 file-based output needs one shared filesystem
#                (a single .md per iteration), the series would fragment.
# Engine-level write shaping goes through the openPMD -> ADIOS2 engine
# parameter channel, e.g. fewer/parallel data files per iteration:
#   ENGINE_JSON='{"adios2":{"engine":{"parameters":{\
#     "Collectives":"MPI_INIT","SubFiles":"All","NumSubFiles":"16"}}}}'
set -euo pipefail

SDF_DIR=${1:?sdf dump directory}
OUT_DIR=${2:?openPMD output directory}
: "${SDF2OPMD:=$HOME/sdf2opmd-cluster/bin/sdf2opmd}"
NODES=${NODES:-2}
RANKS=${RANKS:-$((NODES * 32))}

DEPS=${DEPS:-$HOME/sdf2opmd-cluster/deps}
export LD_LIBRARY_PATH="$DEPS/openpmd/lib:$DEPS/adios2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x $SDF2OPMD ]]; then
    echo "sdf2opmd not found/executable: $SDF2OPMD" >&2
    echo "run build_sdf2opmd_cluster.sh first" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
if [[ -n ${STRIPE:-} ]] && command -v lfs >/dev/null; then
    if [[ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]]; then
        echo "STRIPE set but $OUT_DIR not empty; keeping existing layout" >&2
    else
        lfs setstripe -c "$STRIPE" "$OUT_DIR"
        echo "stripe of $OUT_DIR: $(lfs getstripe "$OUT_DIR" | head -2 | tail -1)"
    fi
elif [[ -n ${STRIPE:-} ]]; then
    echo "WARNING: STRIPE=$STRIPE but no 'lfs' here; ignoring" >&2
fi

ENGINE_ARGS=()
# BBSTAGE=1 switches the writer to the bbstage engine (denisbertini/ADIOS2
# fork): iterations land in node-local staging instantly and background
# threads drain them to OUT_DIR while the converter reads the next SDF
# file - it pipelines read and write phases instead of serializing them,
# and absorbs Lustre stalls. CloseMode=durable keeps the guarantee that
# everything is on OUT_DIR when the converter exits; "staged" returns
# earlier and the engine's exit finalizer finishes the drain (verified:
# both modes byte-exact). Requires ADIOS2 built from the fork (default of
# build_sdf2opmd_cluster.sh).
if [[ ${BBSTAGE:-} == 1 || ${BBSTAGE:-} == on ]]; then
    ENGINE_JSON=${ENGINE_JSON:-'{"adios2":{"engine":{"type":"bbstage","pretend_engine":"bp5","parameters":{"CloseMode":"durable","Aggregate":"on"}}}}'}
fi
[[ -n ${ENGINE_JSON:-} ]] && ENGINE_ARGS+=(--engine "$ENGINE_JSON")

time srun -n "$RANKS" "$SDF2OPMD" -p "$SDF_DIR" -g 'normal_*.sdf' \
    -o "$OUT_DIR/simData" --name 'simData_%04T.bp' "${ENGINE_ARGS[@]}"

time srun -n "$RANKS" "$SDF2OPMD" -p "$SDF_DIR" -g 'test_*.sdf' \
    -o "$OUT_DIR/simDataTest" --name 'simData_%04T.bp' "${ENGINE_ARGS[@]}"

du -sh "$OUT_DIR"/simData* 2>/dev/null || true
echo "conversion done: $OUT_DIR"
