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

time srun -n "$RANKS" "$SDF2OPMD" -p "$SDF_DIR" -g 'normal_*.sdf' \
    -o "$OUT_DIR/simData" --name 'simData_%04T.bp'

time srun -n "$RANKS" "$SDF2OPMD" -p "$SDF_DIR" -g 'test_*.sdf' \
    -o "$OUT_DIR/simDataTest" --name 'simData_%04T.bp'

du -sh "$OUT_DIR"/simData* 2>/dev/null || true
echo "conversion done: $OUT_DIR"
