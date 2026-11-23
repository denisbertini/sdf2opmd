#!/usr/bin/env bash
# End-to-end offline smoke test of the benchmark workflow:
#   synthetic EPOCH-like SDF -> sdf2opmd (2 series) -> both analyses -> compare
# Requires: built repo binaries (build/sdf2opmd), a python env with
# numpy + sdfr + openpmd-api (venv path in $VENV or on PATH).
#
# usage: ./run_local_smoke.sh [sdf2opmd_binary]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SDF2OPMD=${1:-$ROOT/build/sdf2opmd}
VENV=${VENV:-/tmp/opencode/venv-bench}
PY="$VENV/bin/python"
[[ -x "$PY" ]] || PY=$(command -v python3)

WORK=$(mktemp -d /tmp/sdf2opmd-smoke.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}

"$PY" "$HERE/make_bench_sdf.py" "$WORK/sdf" 3

NRANKS=${NRANKS:-3}
mpirun -n "$NRANKS" "$SDF2OPMD" -p "$WORK/sdf" -g 'normal_*.sdf' \
    -o "$WORK/normal" --quiet
mpirun -n "$NRANKS" "$SDF2OPMD" -p "$WORK/sdf" -g 'test_*.sdf' \
    -o "$WORK/test" --quiet

cd "$HERE"
for set in normal test; do
  "$PY" analyze_sdf.py "$WORK/sdf" --pattern "${set}_*.sdf" \
      -o "$WORK/stats_sdf_${set}.json"
  "$PY" analyze_openpmd.py "$WORK/$set" -o "$WORK/stats_opmd_${set}.json"
  "$PY" compare_analysis.py "$WORK/stats_sdf_${set}.json" \
      "$WORK/stats_opmd_${set}.json"
done

echo "SMOKE PASS"
