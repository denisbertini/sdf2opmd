#!/usr/bin/env bash
# Build openPMD-api (ADIOS2-only) for sdf2opmd.
#
# Uses a pre-built MPI-enabled ADIOS2 (default: the local bbstage build tree).
# Set ADIOS2_DIR to point elsewhere, OPENPMD_SRC to an existing openPMD-api
# checkout, or nothing to clone the pinned tag into deps/.
#
#   ./scripts/build-openpmd.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$HERE/deps"
PREFIX="$DEPS/openpmd"

: "${ADIOS2_DIR:=/home/denis/work/io/build-bbstage-mpi}"
: "${CC:=mpicc}"
: "${CXX:=mpicxx}"
: "${OPENPMD_TAG:=v0.17.0}"

if [[ -n "${OPENPMD_SRC:-}" ]]; then
  SRC="$OPENPMD_SRC"
else
  SRC="$DEPS/openpmd-src"
  if [[ ! -d "$SRC" ]]; then
    git clone --depth 1 --branch "$OPENPMD_TAG" \
      https://github.com/openPMD/openPMD-api "$SRC"
  fi
fi

echo "openPMD source : $SRC"
echo "ADIOS2 dir     : $ADIOS2_DIR"
echo "install prefix : $PREFIX"

cmake -S "$SRC" -B "$DEPS/openpmd-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DopenPMD_USE_MPI=ON \
  -DopenPMD_USE_HDF5=OFF \
  -DopenPMD_USE_ADIOS2=ON \
  -DopenPMD_USE_PYTHON=OFF \
  -DopenPMD_USE_MPI4PY=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DopenPMD_BUILD_CLI_TOOLS=OFF \
  -DCMAKE_PREFIX_PATH="$ADIOS2_DIR" \
  -Dadios2_DIR="$ADIOS2_DIR"

cmake --build "$DEPS/openpmd-build" -j"$(nproc)"
cmake --install "$DEPS/openpmd-build"
echo "done -> $PREFIX"
