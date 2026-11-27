#!/bin/bash
#SBATCH --job-name=sdf2opmd
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=sdf2opmd_%j.out
#SBATCH --error=sdf2opmd_%j.err
# =============================================
# SDF -> openPMD conversion + in-job validation
# =============================================
# Conversion is the measured window; validation runs AFTER it, so it costs
# job wall time but never pollutes the numbers. The job exits non-zero if a
# single comparison is not bit-exact, so an unvalidated series can never
# silently become "the converted data".
#
# LAUNCH=
#   world : srun launches the whole MPI world - one task per rank, the task
#           being `apptainer exec $CONT sdf2opmd`, MPI bootstrapping through
#           the PMI/PMIx that Slurm hands down (fd inheritance across the
#           container boundary). Only this mode spreads ONE fat dump over
#           several nodes. MPI_PROBE=1 checks it in seconds before the run.
#   shard : srun starts ONE task per node; that task runs a fully local
#           mpirun over 1/NODES of the input files. Needs no PMI/PMIx, so it
#           works with any container MPI. Granularity is one file: many-file
#           dump sets scale perfectly, one big file cannot pass one node.
#           Both modes yield the same series - openPMD file-based output
#           writes one self-contained .bp per iteration, so independent
#           writers legitimately assemble one series.
#   local : no Slurm, shards as background processes - rehearse on a
#           workstation: CONT= LAUNCH=local NODES=2 PPT=2 ...
#
# Layout is verified, not assumed (MDT defaults can change cluster-wide).
#
# Usage: sbatch bench/epoch-lwfa/convert_validate.sbatch
#        sbatch --nodes=2 --export=ALL,LAUNCH=shard,PAT='0000.sdf' \
#               bench/epoch-lwfa/convert_validate.sbatch
#        sbatch --export=ALL,MPI_PROBE=1 bench/epoch-lwfa/convert_validate.sbatch
#        sbatch --export=ALL,DRY=1 bench/epoch-lwfa/convert_validate.sbatch
set -uo pipefail

export CONT=${CONT-/lustre/rz/dbertini2/mpiio_evolve/container/plasma_pp.sif}
export APPTAINER_BINDPATH=${APPTAINER_BINDPATH-/lustre/rz}   # repo + data + image
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER
export OMPI_MCA_io=romio341              # same ROMIO choice as the PIO bench
export TMPDIR=${SLURM_TMPDIR:-/tmp}      # >2 GiB block dumps go to node NVMe

# ---- edit these (or --export=ALL,S=...,DATA=...) ----------------------------
R=${R:-/lustre/rz/dbertini/io-tests/sdf}    # benchmark root
S=${S:-$R/sdf2opmd}                         # repo clone (with build/)
DATA=${DATA:-$R/lwfa/data}                  # SDF dumps - read side
OUT=${OUT:-$R/opmd}                         # openPMD series - write side
PAT=${PAT:-'00[1-9][0-9].sdf'}              # glob within DATA
NAME=${NAME:-'simData_%04T.bp'}
STRIPE=${STRIPE:-16}                        # OSTs for OUT; 0 = MDT default
LAUNCH=${LAUNCH:-world}                     # world | shard | local
MPI_TAG=${MPI_TAG:-pmix}                    # srun --mpi=... ('' = scheduler default)
MPI_PROBE=${MPI_PROBE:-1}                   # 1 = cross-node MPI smoke test first
DRY=${DRY:-0}                               # 1 = print launch lines, run nothing
VALIDATE=${VALIDATE:-1}
EXTRA=${EXTRA:-}                            # e.g. '--quiet', '--species electron'
ENGINE_JSON=${ENGINE_JSON:-}                # ADIOS2 engine config (bbstage...)
# ------------------------------------------------------------------------------

NODES=${SLURM_NNODES:-${NODES:-1}}
PPT=${SLURM_NTASKS_PER_NODE:-${PPT:-$(getconf _NPROCESSORS_ONLN)}}
NT=${SLURM_NTASKS:-$((NODES * PPT))}

SRUN=(srun)
[[ -n ${MPI_TAG:-} ]] && SRUN+=(--mpi="$MPI_TAG")
WRAP=()                                     # run the task inside the image
[[ -n ${CONT:-} && $LAUNCH != local ]] && WRAP=(apptainer exec "$CONT")
BIN=$S/build/sdf2opmd
VAL=$S/bench/epoch-lwfa/validate_conversion.py

sayrun() { echo "+ $*"; [[ $DRY == 1 ]] && return 0; "$@"; }
sec() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.2f", (b-a)/1e9}'; }

cd "$S" || exit 1
[[ -x $BIN ]] || { echo "missing $BIN: cmake --build build -j4" >&2; exit 1; }
dump=()
[[ -x $S/build/sdfdump ]] && dump=(--sdfdump "$S/build/sdfdump") \
    || echo "note: no build/sdfdump -> blocks > 2 GiB cannot be validated" >&2
mapfile -t files < <(find "$DATA" -maxdepth 1 -name "$PAT" | sort)
(( ${#files[@]} > 0 )) || { echo "no '$PAT' files in $DATA" >&2; exit 1; }

echo "job=${SLURM_JOB_ID:-local} nodes=$NODES ranks/node=$PPT total_ranks=$NT"
echo "inputs=${#files[@]}  pattern=$PAT  launch=$LAUNCH${MPI_TAG:+ (srun --mpi=$MPI_TAG)}"
echo "read = $DATA"
echo "write= $OUT  ($(df -PT "$OUT" 2>/dev/null | awk 'NR==2{print $2, $1}'))"
if [[ $DRY != 1 && -n ${CONT:-} && $LAUNCH != local ]]; then
    apptainer exec "$CONT" true || { echo "container unusable: $CONT" >&2; exit 1; }
fi

# ---- fresh output directory, layout probed not assumed ----------------------
rm -rf "$OUT"; mkdir -p "$OUT"
if [[ $STRIPE != 0 && $DRY != 1 && $LAUNCH != local ]] && command -v lfs >/dev/null 2>&1; then
    lfs setstripe -c "$STRIPE" "$OUT"
    probe=$OUT/.layout_probe
    dd if=/dev/zero of=$probe bs=1M count=1 status=none
    got=$(lfs setstripe --get "$probe" 2>/dev/null | awk '/Stripe Count/{print $3}'); rm -f $probe
    [[ -z $got ]] && got=$(lfs getstripe -c "$probe" 2>/dev/null)
    echo "layout: stripes=${got:-?} (requested $STRIPE)"
    [[ -z ${got:-} || $got == "$STRIPE" ]] \
        || { echo "ABORT: layout is ${got:-?}, not $STRIPE - numbers would be meaningless" >&2; exit 1; }
fi

conv=(-p "$DATA" -o "$OUT" --name "$NAME")
[[ -n $ENGINE_JSON ]] && conv+=(--engine "$ENGINE_JSON")
conv+=($EXTRA)                             # word-split on purpose

# ---- can the container's MPI bootstrap under srun at all? ------------------
if [[ $MPI_PROBE == 1 && $LAUNCH == world && $DRY != 1 && $NODES -ge 1 ]]; then
    echo "probing MPI world: ${files[0]##*/} --info on $NT rank(s) over $NODES node(s)"
    if ! "${SRUN[@]}" --ntasks="$NT" --ntasks-per-node="$PPT" --label \
            "${WRAP[@]}" bash -c "'$BIN' --info -p '$DATA' -f '${files[0]}' >/dev/null"; then
        cat >&2 <<'EOF'
ABORT: the MPI world did not come up under srun.
  - other bootstrap plugins:  --export=ALL,MPI_TAG=pmi2   (or pmix_v3, or '')
  - what the image offers:    apptainer exec $CONT ompi_info | grep -iE 'pmix|pmi|slurm'
  - proven fallback:          --export=ALL,LAUNCH=shard
EOF
        exit 1
    fi
    echo "probe ok"
fi

# ---- measured window --------------------------------------------------------
t0=$(date +%s%N)
case $LAUNCH in
world)
    sayrun "${SRUN[@]}" --ntasks="$NT" --ntasks-per-node="$PPT" --label \
        "${WRAP[@]}" "$BIN" "${conv[@]}" -g "$PAT" \
        || { echo "ABORT: conversion failed" >&2; exit 1; }
    ;;
shard)
    # one task per node; each derives its own subset deterministically (same
    # find|sort everywhere) and runs a purely local mpirun
    sayrun "${SRUN[@]}" --ntasks="$NODES" --ntasks-per-node=1 --label \
        env S="$S" DATA="$DATA" OUT="$OUT" PAT="$PAT" NAME="$NAME" NODES="$NODES" \
        PPT="$PPT" CONT="${CONT:-}" EXTRA="$EXTRA" ENGINE_JSON="$ENGINE_JSON" \
        bash -c '
            cd "$S" || exit 1
            mapfile -t f < <(find "$DATA" -maxdepth 1 -name "$PAT" | sort)
            list=""
            for i in "${!f[@]}"; do
                (( i % NODES == SLURM_NODEID )) && list+="${list:+:}${f[i]}"
            done
            [[ -n $list ]] || { echo "shard $SLURM_NODEID: nothing to do"; exit 0; }
            echo "shard $SLURM_NODEID ($(hostname)): $(tr ":" "\n" <<<"$list" | wc -l) file(s), $PPT local ranks"
            set -- -p "$DATA" -o "$OUT" --name "$NAME" $EXTRA
            [[ -n $ENGINE_JSON ]] && set -- "$@" --engine "$ENGINE_JSON"
            if [[ -n $CONT ]]; then
                apptainer exec "$CONT" mpirun -np "$PPT" "$S/build/sdf2opmd" "$@" --files "$list"
            else
                mpirun -np "$PPT" "$S/build/sdf2opmd" "$@" --files "$list"
            fi' \
        || { echo "ABORT: a shard failed" >&2; exit 1; }
    ;;
local)
    pids=()
    for ((k = 0; k < NODES; ++k)); do
        (
            list=""
            for i in "${!files[@]}"; do (( i % NODES == k )) && list+="${list:+:}${files[i]}"; done
            [[ -n $list ]] || exit 0
            echo "shard $k: $(tr ':' '\n' <<<"$list" | wc -l) file(s)"
            sayrun "$BIN" "${conv[@]}" --files "$list"
        ) & pids+=($!)
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    (( rc == 0 )) || { echo "ABORT: a shard failed" >&2; exit 1; }
    ;;
*)  echo "unknown LAUNCH=$LAUNCH (world|shard|local)" >&2; exit 2 ;;
esac
win=$(sec "$t0" "$(date +%s%N)")

# ---- accounting -------------------------------------------------------------
if [[ $DRY == 1 ]]; then echo "DRY=1: nothing written, checks skipped"; exit 0; fi
n_iter=$(find "$OUT" -maxdepth 1 -name '*.bp' | wc -l)
bytes=$(du -sb "$OUT" | cut -f1)
echo "---------------------------------------------------------------"
awk -v f="${#files[@]}" -v i="$n_iter" -v b="$bytes" -v w="$win" -v n="$NT" \
    'BEGIN{printf "CONVERT  files=%d  iterations=%d  bytes=%d\n         wall=%.2fs  %.1f MB/s  %.2fs/file  %d ranks  %.1f MB/s/rank\n", f, i, b, w, b/1e6/w, w/f, n, b/1e6/w/n}'
echo "---------------------------------------------------------------"
(( n_iter == ${#files[@]} )) \
    || { echo "ABORT: ${#files[@]} inputs but $n_iter iterations written" >&2; exit 1; }

# ---- after the measured window: prove it is the same data ------------------
if (( VALIDATE )); then
    t2=$(date +%s%N)
    if [[ $LAUNCH == local ]]; then
        "${PYTHON:-python3}" "$VAL" --sdf-dir "$DATA" --pattern "$PAT" \
            --series "$OUT" "${dump[@]}"
    else
        "${SRUN[@]}" --ntasks=1 --ntasks-per-node=1 --label "${WRAP[@]}" \
            "${PYTHON:-python3}" "$VAL" --sdf-dir "$DATA" --pattern "$PAT" \
            --series "$OUT" "${dump[@]}"
    fi
    rc=$?
    echo "validate: $(sec "$t2" "$(date +%s%N)")s (outside the measured window), rc=$rc"
    exit $rc
fi
echo "VALIDATE=0: comparison skipped"
