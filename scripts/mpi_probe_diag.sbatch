#!/bin/bash
# =============================================
# mpi_probe_diag.sbatch -- why does a world MPI job bootstrapped by
# `srun --mpi=pmix` with `apptainer exec` per rank not come up?
#
# Steps escalate: bare srun -> container in srun -> image MPI/PMIx audit ->
# a real 4-rank world WITHOUT and WITH the host /dev/shm bound into the
# container (the prime suspect: per-rank apptainer instances get a private
# /dev/shm, so OMPI's vader sm transport and the PMIx segments cannot see
# across ranks). OMPI wireup tracing is on, every step has a hard timeout,
# so a hang ends as `rc=124`, never as a job someone has to scancel.
#
# Usage: sbatch --wait scripts/mpi_probe_diag.sbatch
# =============================================
#SBATCH --job-name=sdfmpidiag
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --partition=long
#SBATCH --output=sdfmpidiag_%j.out
#SBATCH --error=sdfmpidiag_%j.err
set -uo pipefail

export CONT=${CONT-/lustre/rz/dbertini2/mpiio_evolve/container/plasma_pp.sif}
export APPTAINER_BINDPATH=${APPTAINER_BINDPATH-/lustre/rz}
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER
export OMPI_MCA_io=romio341

S=${S:-/lustre/rz/dbertini/io-tests/sdf/sdf2opmd}
DATA=${DATA:-/lustre/rz/dbertini/io-tests/sdf/lwfa/data}
F=${F:-$DATA/0010.sdf}
BIN=$S/build/sdf2opmd
T=${T:-90}                                   # hard seconds per step
NT=${NT:-4}                                  # world size for steps 3/4
PPN=${PPN:-2}                                # ranks per node for steps 3/4

say() { echo; echo "===== $*"; }
cd "$S" || exit 1

say "0. bare srun across nodes (no container)"
srun --ntasks=2 --ntasks-per-node=1 --label \
    bash -c 'echo step0 rank $SLURM_PROCID on $(hostname) OK'

say "1. apptainer inside an srun task (cold start: image on Lustre)"
t0=$(date +%s)
srun --ntasks=2 --ntasks-per-node=1 --label timeout $T \
    apptainer exec "$CONT" bash -c \
    'echo step1 rank $SLURM_PROCID on $(hostname); df -h /dev/shm | tail -1;
     echo "pmix sockets: $(ls /run/pmix 2>/dev/null | tr "\n" " ") ${TMPDIR:-}/pmix*" '
echo "step1 wall: $(( $(date +%s) - t0 ))s"

say "2. image MPI / PMI support (what can it speak?)"
srun --ntasks=1 --label timeout $T apptainer exec "$CONT" \
    sh -c 'ompi_info --parsable 2>/dev/null | grep -iE "pmix|pmi_|ident:" | head'
srun --ntasks=1 --label timeout $T sh -c \
    'echo "host slurm: $(scontrol --version 2>/dev/null)";
     ls -l /usr/lib64/libpmix.so* /usr/lib/x86_64-linux-gnu/libpmix.so* 2>/dev/null | head -3;
     rpm -q pmix 2>/dev/null || dpkg -l "*pmix*" 2>/dev/null | tail -2'

say "3. world MPI, private /dev/shm per rank (the config job 16935 used)"
t0=$(date +%s)
OMPI_MCA_btl_base_verbose=10 OMPI_MCA_pml_base_verbose=10 \
srun --mpi=pmix --ntasks="$NT" --ntasks-per-node="$PPN" --label \
    bash -c "timeout $T apptainer exec '$CONT' \
        '$BIN' --info -p '$DATA' -f '$F' >/dev/null 2>/tmp/step3_\$SLURM_PROCID.log;
        rc=\$?; [ \$rc -gt 1 ] && tail -5 /tmp/step3_\$SLURM_PROCID.log >&2;
        echo step3 rank \$SLURM_PROCID on \$(hostname) rc=\$rc"
rc3=$?
echo "step3 wall: $(( $(date +%s) - t0 ))s rc=$rc3"

say "4. world MPI, HOST /dev/shm bound into the container"
t0=$(date +%s)
OMPI_MCA_btl_base_verbose=10 \
srun --mpi=pmix --ntasks="$NT" --ntasks-per-node="$PPN" --label \
    bash -c "timeout $T apptainer exec --bind /dev/shm '$CONT' \
        '$BIN' --info -p '$DATA' -f '$F' >/dev/null 2>/tmp/step4_\$SLURM_PROCID.log;
        rc=\$?; [ \$rc -gt 1 ] && tail -5 /tmp/step4_\$SLURM_PROCID.log >&2;
        echo step4 rank \$SLURM_PROCID on \$(hostname) rc=\$rc"
echo "step4 wall: $(( $(date +%s) - t0 ))s rc=$?"

say "done"
