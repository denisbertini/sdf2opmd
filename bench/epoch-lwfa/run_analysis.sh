#!/bin/bash
#SBATCH --job-name=sdf-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --mem-per-cpu=4000
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH -o %j.analyze.out.log
#SBATCH -e %j.analyze.err.log
# Step 3: validate the conversion - same physics analysis on the original
# SDF dumps (sdfr) and on the converted openPMD series (openpmd_api), then
# a strict comparison.
#
#   sbatch ./run_analysis.sh /path/to/dumps /path/to/output
set -euo pipefail

SDF_DIR=${1:?sdf dump directory}
OUT_DIR=${2:?openPMD output directory}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python3}

# one-time (or already present):
#   pip install --user numpy sdfr openpmd-api

cd "$HERE"
STATUS=0

for set in normal test; do
    echo "=== $set ==="
    "$PY" analyze_sdf.py "$SDF_DIR" --pattern "${set}_*.sdf" \
        -o "$OUT_DIR/stats_sdf_${set}.json"
    "$PY" analyze_openpmd.py "$OUT_DIR/simData$([ "$set" = test ] && echo Test || echo '')" \
        -o "$OUT_DIR/stats_openpmd_${set}.json"
    "$PY" compare_analysis.py "$OUT_DIR/stats_sdf_${set}.json" \
        "$OUT_DIR/stats_openpmd_${set}.json" || STATUS=1
done

echo "validation status: $STATUS"
exit $STATUS
