#!/usr/bin/env bash
# End-to-end test: SDF fixtures -> sdf2opmd -> openPMD readback validation.
set -euo pipefail

SDF2OPMD=${1:?path to sdf2opmd binary}
READBACK=${2:?path to readback binary}
ROOT=${3:?project root}

# The local ADIOS2 build needs the system libfabric (FABRIC_1.10), but
# mpirun injects oneAPI's older bundled libfabric via LD_LIBRARY_PATH.
# Put the system library directory first.
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}

WORK=$(mktemp -d /tmp/sdf2opmd-e2e.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

python3 "$ROOT/scripts/make_fixture.py" "$WORK/sdf"

"$SDF2OPMD" --info -p "$WORK/sdf" -f "0001.sdf"

mpirun -n 3 "$SDF2OPMD" -p "$WORK/sdf" -f "0001.sdf:0002.sdf" \
    -o "$WORK/out3d" --machine e2e
mpirun -n 2 "$SDF2OPMD" -p "$WORK/sdf" -f "1001.sdf" -o "$WORK/out2d"

mpirun -n 1 "$READBACK" "$WORK/out3d/simData_%04T.bp" 3d
mpirun -n 1 "$READBACK" "$WORK/out2d/simData_%04T.bp" 2d

echo "E2E PASS ($WORK cleaned)"
