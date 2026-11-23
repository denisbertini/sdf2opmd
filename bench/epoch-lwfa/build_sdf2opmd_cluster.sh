#!/usr/bin/env bash
# Build the sdf2opmd converter (+ its deps) for cluster use, e.g. inside the
# same apptainer container that runs EPOCH, so binaries match its MPI.
#
#   srun --exclusive apptainer exec $CONT ./build_sdf2opmd_cluster.sh
#
# Everything installs under $PREFIX (default ~/sdf2opmd-cluster).
# Tip: if the cluster has adios2/openpmd modules, point openPMD_DIR at them
# and skip the dependency build instead.
set -euo pipefail

PREFIX=${PREFIX:-$HOME/sdf2opmd-cluster}
SRC=$PREFIX/src
JOBS=${JOBS:-8}
REPO=${REPO:-https://github.com/denisbertini/sdf2opmd.git}
ADIOS2_TAG=${ADIOS2_TAG:-v2.10.2}
OPENPMD_TAG=${OPENPMD_TAG:-v0.17.6}

mkdir -p "$SRC" "$PREFIX/bin"

# --- sources ----------------------------------------------------------------
if [[ ! -d $SRC/sdf2opmd ]]; then
    git clone --recurse-submodules "$REPO" "$SRC/sdf2opmd"
fi
git -C "$SRC/sdf2opmd" pull --recurse-submodules
[[ -d $SRC/adios2 ]] || git clone --depth 1 -b "$ADIOS2_TAG" \
    https://github.com/ornladios/ADIOS2 "$SRC/adios2"
[[ -d $SRC/openPMD-api ]] || git clone --depth 1 -b "$OPENPMD_TAG" \
    https://github.com/openPMD/openPMD-api "$SRC/openPMD-api"

# --- ADIOS2 (BP5, MPI) -------------------------------------------------------
cmake -S "$SRC/adios2" -B "$PREFIX/adios2-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX/deps/adios2" \
  -DADIOS2_USE_Blosc=OFF -DADIOS2_USE_BZip2=OFF -DADIOS2_USE_DataMan=OFF \
  -DADIOS2_USE_DataSpaces=OFF -DADIOS2_USE_HDF5=OFF -DADIOS2_USE_Python=OFF \
  -DADIOS2_USE_SSC=OFF -DADIOS2_USE_ZFP=OFF -DADIOS2_USE_ZeroMQ=OFF \
  -DADIOS2_USE_FORTIFY=OFF
cmake --build "$PREFIX/adios2-build" -j"$JOBS"
cmake --install "$PREFIX/adios2-build"

# --- openPMD-api (BP5 only) ---------------------------------------------------
cmake -S "$SRC/openPMD-api" -B "$PREFIX/openpmd-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX/deps/openpmd" \
  -DopenPMD_USE_MPI=ON -DopenPMD_USE_HDF5=OFF -DopenPMD_USE_ADIOS2=ON \
  -DopenPMD_USE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$PREFIX/deps/adios2"
cmake --build "$PREFIX/openpmd-build" -j"$JOBS"
cmake --install "$PREFIX/openpmd-build"

# --- sdf2opmd -----------------------------------------------------------------
cmake -S "$SRC/sdf2opmd" -B "$PREFIX/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DopenPMD_DIR="$PREFIX/deps/openpmd/lib/cmake/openPMD" \
  -DCMAKE_PREFIX_PATH="$PREFIX/deps/adios2;$PREFIX/deps/openpmd"
cmake --build "$PREFIX/build" -j"$JOBS"
cp "$PREFIX/build/sdf2opmd" "$PREFIX/bin/"

# quick self-test
"$PREFIX/build/sdf2opmd" --help >/dev/null && \
    echo "installed: $PREFIX/bin/sdf2opmd"
