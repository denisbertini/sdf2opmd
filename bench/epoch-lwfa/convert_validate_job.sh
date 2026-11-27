#!/usr/bin/env bash
#SBATCH --job-name=sdf2opmd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # one local launcher per node, see below
#SBATCH --output=sdf2opmd-%j.log
#
# Convert EPOCH SDF dumps -> openPMD on >= 1 nodes, then validate the result
# in the SAME job (so the job fails if the conversion is not bit-exact).
#
# Why sharding instead of one MPI world:
#   launching one MPI job across nodes needs the container's MPI to speak the
#   batch scheduler's launch protocol (PMIx/slurm plm), which is site- and
#   image-specific. Instead each node runs a fully local `mpirun` over its
#   share of the input files. openPMD file-based output makes one iteration
#   per .bp, self-contained, so independent writers add iterations to the same
#   series - the union is the series, and the validator opens it as such.
#   Costs: parallelism granularity is one file (a single fat file still only
#   scales to one node's ranks), and every node needs the shared OUT_DIR.
#
# usage (cluster, from the repo root):
#   sbatch --export=ALL,CONT=$CONT,SDF_DIR=$DATA,OUT_DIR=/lustre/.../opmd \
#          bench/epoch-lwfa/convert_validate_job.sh
#   sbatch --nodes=4 --export=ALL,...,PATTERN='normal_*.sdf',STRIPE=8,... \
#          bench/epoch-lwfa/convert_validate_job.sh
#
# usage (rehearse locally, no Slurm/container):
#   CONT= SDF2OPMD=../build/sdf2opmd NODES=2 PPT=2 PATTERN='00*.sdf' \
#   SDF_DIR=/tmp/fx OUT_DIR=/tmp/shard \
#       bench/epoch-lwfa/convert_validate_job.sh --local
#
# env: CONT         apptainer/singularity image (empty = run without container)
#      SDF2OPMD     converter binary            [../build/sdf2opmd]
#      SDFDUMP      raw block dumper for blocks > 2 GiB (validator)
#      SDF_DIR      directory with .sdf files   [required]
#      OUT_DIR      openPMD output, SHARED for > 1 node  [required]
#      PATTERN      shell glob within SDF_DIR   ['*.sdf']
#      NAME         iteration file name         ['simData_%04T.bp']
#      PPT          MPI ranks per node          [all cores of the node]
#      STRIPE       lfs setstripe count for OUT_DIR (set while empty)
#      EXTRA        extra converter flags, e.g. '--quiet'
#      ENGINE_JSON  openPMD engine config JSON for ADIOS2 tuning/bbstage
#      VALIDATE     1 (default) | 0
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONT=${CONT:-}
SDF2OPMD=${SDF2OPMD:-$HERE/../../build/sdf2opmd}
SDF_DIR=${SDF_DIR:?set SDF_DIR to the directory of .sdf dumps}
OUT_DIR=${OUT_DIR:?set OUT_DIR (a filesystem all ranks can see)}
PATTERN=${PATTERN:-*.sdf}
NAME=${NAME:-simData_%04T.bp}
PPT=${PPT:-$(getconf _NPROCESSORS_ONLN)}
VALIDATE=${VALIDATE:-1}
RUN_LOCAL=0
[[ ${1:-} == --local ]] && RUN_LOCAL=1

# ---- run `cmd` inside the container when one is configured ----------------
in_cont() {
    if [[ -n ${CONT:-} ]]; then
        apptainer exec "$CONT" "$@"
    else
        "$@"
    fi
}

# ---- the per-node work unit ------------------------------------------------
do_shard() {
    local id=$1 n=$2 list="" i count=0
    mapfile -t all < <(find "$SDF_DIR" -maxdepth 1 -name "$PATTERN" -printf '%p\n' | sort)
    for i in "${!all[@]}"; do
        (( i % n == id )) || continue
        list+="${list:+:}${all[i]}"
        (( ++count ))
    done
    if (( count == 0 )); then
        echo "shard $id: no files for pattern '$PATTERN'"
        return 0
    fi
    echo "shard $id: $count file(s), $PPT ranks"
    in_cont mpirun -np "$PPT" "$SDF2OPMD" -p "$SDF_DIR" --files "$list" \
        -o "$OUT_DIR" --name "$NAME" ${EXTRA:-} \
        ${ENGINE_JSON:+--engine "$ENGINE_JSON"}
}

do_validate() {
    # one serial reader compares every input file against the series
    in_cont env TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}" \
              ${SDFDUMP:+SDFDUMP=$SDFDUMP} \
        "${PYTHON:-python3}" "$HERE/validate_conversion.py" \
            --sdf-dir "$SDF_DIR" --pattern "$PATTERN" --series "$OUT_DIR"
}

# ---- dispatch: shard / validate / orchestrator -----------------------------
case ${1:-} in
    --shard)    do_shard "${SLURM_NODEID:-0}" "${SLURM_NTASKS:-1}"; exit $? ;;
    --validate) do_validate; exit $? ;;
esac
[[ ${1:-} == --local ]] && shift || true

mapfile -t files < <(find "$SDF_DIR" -maxdepth 1 -name "$PATTERN" -printf '%p\n' | sort)
(( ${#files[@]} > 0 )) || { echo "no '$PATTERN' files in $SDF_DIR" >&2; exit 1; }
NODES=${SLURM_NNODES:-${NODES:-1}}
mkdir -p "$OUT_DIR"
src=$(df -PT "$OUT_DIR" 2>/dev/null | awk 'NR==2 {print $2" "$1}')
echo "converting ${#files[@]} file(s) from $SDF_DIR"
echo "  pattern=$PATTERN  nodes=$NODES  ranks/node=$PPT"
echo "  out=$OUT_DIR  (fs: ${src:-unknown})"
if [[ ${src:-} == tmpfs* || ${src:-} == "devtmpfs "* ]]; then
    echo "OUT_DIR is node-local tmpfs: invisible to other nodes and deleted" >&2
    echo "when the job ends. Point OUT_DIR at a shared filesystem." >&2
    exit 1
fi

if [[ -n ${STRIPE:-} ]] && command -v lfs >/dev/null 2>&1; then
    # set while the directory is empty: iterations created below inherit the
    # OST layout, keeping these writes off the targets the SDF reads hit
    lfs setstripe -c "$STRIPE" "$OUT_DIR" || echo "note: lfs setstripe failed" >&2
fi

t0=$(date +%s)
if (( RUN_LOCAL )); then
    pids=()
    for ((k = 0; k < NODES; ++k)); do
        SLURM_NODEID=$k SLURM_NTASKS=$NODES "$0" --shard & pids+=($!)
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    (( rc == 0 )) || { echo "a shard failed" >&2; exit 1; }
else
    srun --label --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
        "$0" --shard
fi

n_iter=$(find "$OUT_DIR" -maxdepth 1 -name '*.bp' | wc -l)
echo "converted ${#files[@]} file(s) into $n_iter iteration(s) in $(( $(date +%s) - t0 )) s"
if (( n_iter != ${#files[@]} )); then
    echo "MISMATCH: ${#files[@]} input files but $n_iter iterations in $OUT_DIR" >&2
    exit 1
fi
du -sh "$OUT_DIR" | sed 's/^/output: /'

if (( VALIDATE )); then
    if (( RUN_LOCAL )); then do_validate; else
        srun --label --ntasks=1 --ntasks-per-node=1 "$0" --validate
    fi
else
    echo "VALIDATE=0: skipping the SDF <-> openPMD comparison"
fi
