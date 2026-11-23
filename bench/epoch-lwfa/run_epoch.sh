#!/bin/bash
# Step 1: run EPOCH3D (containerized, LRZ Orion pattern).
# Invoked by epoch_job.sh via sbatch -D ./data ./run-file.sh
export OMP_NUM_THREADS=1

export CONT=${CONT:-/lustre/rz/dbertini/orion/containers/pp/rlx9_ompi_ucx.sif}
export EPOCH_BIN=${EPOCH_BIN:-epoch3d}

export OMPI_MCA_io=romio341
export APPTAINER_BINDPATH=/lustre/rz/dbertini,/cvmfs
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER

srun --export=ALL apptainer exec "$CONT" "$EPOCH_BIN"
