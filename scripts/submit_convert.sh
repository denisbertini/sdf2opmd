#!/bin/bash
#SBATCH --job-name=sdfconv
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=sdfconv_%j.out
#SBATCH --error=sdfconv_%j.err
# submit_convert.sh -- the SDF -> openPMD conversion matrix in ONE sbatch job,
# so every number is a within-session comparison. Legs are "launch:placement"
# pairs, default matrix:
#
#   leg a  world:default  one MPI world over the allocation, MDT placement
#   leg b  shard:default  one local mpirun per node, MDT placement
#                          -> a/b isolates the LAUNCH MECHANISM
#   leg c  world:stripe   one MPI world, 16-stripe output directory
#                          -> a/c isolates the OUTPUT LAYOUT
#
# Each leg reads its OWN equal-sized slice of the input files (COLD=rotate,
# the default) so no leg runs on page cache left by a previous leg; with
# COLD=same every leg reads the same inputs and legs b,c are cache-warm --
# then the comparison is about mechanism only, never about bandwidth.
#
# A layout self-check probes one file per leg directory up front and aborts if
# the striped leg is not $STRIPE or if the striped and default legs are
# identical (the MDT default layout can change cluster-wide -- verify, don't
# assume).
#
# Validation runs AFTER each leg's measured window and gates the job: a leg
# whose output is not bit-exact against its SDF inputs fails the job.
#
# Launch modes:
#   world  srun starts one task per rank, the task being
#          `apptainer exec $CONT sdf2opmd`; the MPI world bootstraps through
#          the PMI/PMIx Slurm hands down. Only mode that spreads ONE fat dump
#          over several nodes.
#   shard  srun starts one task per node, which runs a purely local `mpirun`
#          over that node's file subset. No PMI/PMIx dependency at all.
#          Granularity is one file.
#   local  no Slurm (workstation rehearsal; legs run sequentially in-process)
#   Both cluster modes yield the same series: file-based openPMD output is one
#   self-contained .bp per iteration.
#
# Usage: sbatch scripts/submit_convert.sh
#        sbatch --nodes=2 --export=ALL,LEGS="world:default shard:default" \
#               scripts/submit_convert.sh
#        sbatch --export=ALL,PROBE=1,LEGS="world:default" scripts/submit_convert.sh
#        sbatch --export=ALL,DRY=1 scripts/submit_convert.sh
#        LOCAL rehearsal (no Slurm, no container):
#        CONT= LEGS="local:default local:stripe" DATA=/tmp/fx A=$PWD \
#        PYTHON=/path/to/venv/bin/python scripts/submit_convert.sh
set -uo pipefail

export CONT=${CONT-/lustre/rz/dbertini2/mpiio_evolve/container/plasma_pp.sif}
export APPTAINER_BINDPATH=${APPTAINER_BINDPATH-/lustre/rz}
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER
export OMPI_MCA_io=romio341              # cf. the PIO matrix: same ROMIO
export TMPDIR=${SLURM_TMPDIR:-/tmp}      # sdfdump block staging on node NVMe

# ---- configuration (override with --export=ALL,VAR=...) ---------------------
A=${A:-/lustre/rz/dbertini/io-tests/sdf/sdf2opmd}    # repo clone with build/
DATA=${DATA:-$A/../lwfa/data}                        # SDF dumps - read side
ROOT=${ROOT:-$A/../conv-bench}                       # leg output root
PAT=${PAT:-'00[1-9][0-9].sdf'}                       # glob within DATA
NAME=${NAME:-'simData_%04T.bp'}
LEGS=${LEGS:-"world:default shard:default world:stripe"}
STRIPE=${STRIPE:-16}                                 # OSTs for ':stripe' legs
MPI_TAG=${MPI_TAG:-pmix}                             # srun --mpi= (or '')
PROBE=${PROBE:-1}                                    # cross-node MPI smoke test
COLD=${COLD:-rotate}                                 # rotate | same
DRY=${DRY:-0}
VALIDATE=${VALIDATE:-1}
EXTRA=${EXTRA:-}
ENGINE_JSON=${ENGINE_JSON:-}
# ----------------------------------------------------------------------------

NODES=${SLURM_NNODES:-${NODES:-1}}
PPT=${SLURM_NTASKS_PER_NODE:-${PPT:-$(getconf _NPROCESSORS_ONLN)}}
NT=${SLURM_NTASKS:-$((NODES * PPT))}
SRUN=(srun); [[ -n ${MPI_TAG:-} ]] && SRUN+=(--mpi="$MPI_TAG")
sayrun() { echo "+ $*"; [[ $DRY == 1 ]] && return 0; "$@"; }
sec() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.2f", (b-a)/1e9}'; }

cd "$A" || exit 1
BIN=$A/build/sdf2opmd; VAL=$A/bench/epoch-lwfa/validate_conversion.py
[[ -x $BIN ]] || { echo "missing $BIN: cmake --build build -j4" >&2; exit 1; }
DUMP=(); [[ -x $A/build/sdfdump ]] && DUMP=(--sdfdump "$A/build/sdfdump") \
    || echo "note: no build/sdfdump -> blocks > 2 GiB cannot be validated" >&2

mapfile -t ALL < <(find "$DATA" -maxdepth 1 -name "$PAT" | sort)
(( ${#ALL[@]} > 0 )) || { echo "no '$PAT' files under $DATA" >&2; exit 1; }
read -r -a LEGS <<<"$LEGS"
nl=${#LEGS[@]}
if [[ $COLD == rotate && ${#ALL[@]} -lt $nl ]]; then
    echo "COLD=rotate needs >= $nl input files (have ${#ALL[@]}); using COLD=same" >&2
    COLD=same
fi

echo "job=${SLURM_JOB_ID:-local} nodes=$NODES ranks/node=$PPT ranks=$NT"
echo "inputs=${#ALL[@]}  pattern=$PAT  legs=${LEGS[*]}  cold=$COLD"
echo "read=$DATA  write root=$ROOT"

# ---- leg directories, layout set and PROBED ---------------------------------
leg_dir() { echo "$ROOT/${1%%:*}_${1##*:}"; }
stripes_of() {   # stripe count a file created inside $1 would get
    local d=$1 p
    command -v lfs >/dev/null 2>&1 || { echo "?"; return; }
    p=$d/.layout_probe
    dd if=/dev/zero of=$p bs=1M count=1 status=none 2>/dev/null
    lfs getstripe -c "$p" 2>/dev/null || echo "?"
    rm -f "$p"
}

declare -a OUTS PLACES SIZES
for ((i = 0; i < nl; ++i)); do
    spec=${LEGS[i]} place=${LEGS[i]##*:} out=$(leg_dir "$spec")
    OUTS[i]=$out; PLACES[i]=$place
    # equal slices, remainder to the first legs
    lo=$(( i * ${#ALL[@]} / nl )); hi=$(( (i + 1) * ${#ALL[@]} / nl ))
    SIZES[i]=$(( hi - lo ))
    if [[ $COLD == same ]]; then lo=0; hi=${#ALL[@]}; SIZES[i]=$hi; fi
    [[ $DRY == 1 ]] && continue
    rm -rf "$out"; mkdir -p "$out"
    [[ $place == stripe ]] && command -v lfs >/dev/null 2>&1 \
        && lfs setstripe -c "$STRIPE" "$out"
done

if [[ $DRY != 1 ]] && command -v lfs >/dev/null 2>&1; then
    n_stripe=0; def_dir=""
    for ((i = 0; i < nl; ++i)); do
        got=$(stripes_of "${OUTS[i]}")
        echo "layout: leg ${LEGS[i]} dir=${OUTS[i]##*/} stripes=$got"
        if [[ ${PLACES[i]} == stripe ]]; then
            n_stripe=$(( n_stripe + 1 ))
            [[ $got == "$STRIPE" ]] \
                || { echo "ABORT: striped leg ${LEGS[i]} has $got, want $STRIPE" >&2; exit 1; }
        elif [[ -z $def_dir ]]; then
            def_dir=${OUTS[i]}; def_stripes=$got
        fi
    done
    if (( n_stripe > 0 )) && [[ -n ${def_dir:-} && ${def_stripes:-?} == "$STRIPE" \
         && ${LAYOUT_STRICT:-1} == 1 ]]; then
        echo "ABORT: MDT default is already stripes=$STRIPE, so the :default and" >&2
        echo "       :stripe legs would measure the same layout (LAYOUT_STRICT=0" >&2
        echo "       to run anyway, and read the numbers accordingly)" >&2
        exit 1
    fi
fi

# ---- containers: verify + warm ONCE PER NODE before the world starts -------
# Orion's rootless apptainer mounts the image per instance (squashfuse_ll).
# Launching 32 cold instances per node simultaneously - which is exactly what
# a per-rank `apptainer exec` step does - is a measured stall point: instances
# fail with dbus/squashfuse timeouts and the MPI world dies behind them
# (scripts/mpi_probe_diag.sbatch reproduces both failure and fix; after one
# warm-up launch per node, the identical 128-rank world comes up in ~7 s).
# One instance per node first: names the broken node in seconds, warms the
# node's image cache for every rank that follows. NEVER add `--bind /dev/shm`
# to make the container "see" sm memory: it pushes rootless apptainer into a
# cgroups/dbus path that fails outright on Orion. The plain config works.
cluster_leg=0
for spec in "${LEGS[@]}"; do [[ $spec != local:* ]] && cluster_leg=1; done
wrap=(); [[ -n ${CONT:-} ]] && wrap=(apptainer exec "$CONT")
if [[ $DRY != 1 && $cluster_leg == 1 && -n ${CONT:-} ]]; then
    echo "warming $CONT once per node"
    t0=$(date +%s)
    srun --ntasks="$NODES" --ntasks-per-node=1 --label \
        timeout "${CONT_T:-300}" "${wrap[@]}" true \
        || { echo "ABORT: container did not start on every node" >&2; exit 1; }
    echo "container ok on $NODES node(s) in $(( $(date +%s) - t0 ))s"
fi

# ---- can the MPI world bootstrap? (hard timeout: a stall dies as rc=124 in
# ---- PROBE_T seconds, never as a job someone has to scancel) ---------------
need_probe=0
for spec in "${LEGS[@]}"; do [[ $spec == world:* ]] && need_probe=1; done
if [[ $PROBE == 1 && $DRY != 1 && $need_probe == 1 ]]; then
    echo "probing MPI world: ${ALL[0]##*/} --info on $NT rank(s) / $NODES node(s), timeout ${PROBE_T:-180}s"
    t0=$(date +%s)
    "${SRUN[@]}" --ntasks="$NT" --ntasks-per-node="$PPT" --label \
        bash -c "log=/tmp/sdfprobe.\$SLURM_JOB_ID.\$SLURM_PROCID
                 timeout ${PROBE_T:-180} ${wrap[*]:-} '$BIN' --info -p '$DATA' -f '${ALL[0]}' >/dev/null 2>\$log
                 rc=\$?; (( rc > 1 )) && { echo \"rank \$SLURM_PROCID on \$(hostname) rc=\$rc:\" >&2; tail -5 \$log >&2; }
                 echo \"probe rank \$SLURM_PROCID ok\"; rm -f \$log" \
        || { cat >&2 <<'EOF'
ABORT: the MPI world did not come up under srun.
  rc=124 = the probe hit its timeout (bootstrap stall); the rank logs above
           say who got stuck. The usual cause is container-launch contention:
           raise CONT_T/PROBE_T or check that the warm-up above passed.
  - other plugins:      --export=ALL,MPI_TAG=pmi2   (or pmix_v3, or '')
  - what the image has: apptainer exec $CONT ompi_info | grep -iE 'pmix|pmi|slurm'
  - or drop the world legs:  --export=ALL,LEGS="shard:default shard:stripe"
EOF
              exit 1; }
    echo "probe ok in $(( $(date +%s) - t0 ))s: MPI world spans $NODES node(s)"
fi

# ---- one leg ---------------------------------------------------------------
run_leg() {
    local spec=$1 idx=$2 launch place out conv wrap lo hi list
    launch=${spec%%:*}; place=${spec##*:}; out=${OUTS[idx]}
    if [[ $COLD == rotate ]]; then
        lo=$(( idx * ${#ALL[@]} / nl )); hi=$(( (idx + 1) * ${#ALL[@]} / nl ))
    else
        lo=0; hi=${#ALL[@]}
    fi
    local -a slice=( "${ALL[@]:lo:hi-lo}" )
    list=$(printf '%s:' "${slice[@]}"); list=${list%:}
    conv=(-p "$DATA" -o "$out" --name "$NAME" $EXTRA)
    [[ -n $ENGINE_JSON ]] && conv+=(--engine "$ENGINE_JSON")
    wrap=(); [[ -n ${CONT:-} && $launch != local ]] && wrap=(apptainer exec "$CONT")

    echo "==============================================================="
    echo "LEG $spec  slice=$(( hi - lo )) file(s)  placement=$place  out=${out##*/}"
    echo "==============================================================="
    local t0=$(date +%s%N) rc=0
    case $launch in
    world)
        # CONVT=<seconds> puts a hard ceiling on the measured window: an MPI
        # stall then ends the leg with rc=124 instead of eating the wall limit
        local -a to=(); [[ -n ${CONVT:-} ]] && to=(timeout "$CONVT")
        sayrun "${SRUN[@]}" --ntasks="$NT" --ntasks-per-node="$PPT" --label \
            "${to[@]}" "${wrap[@]}" "$BIN" "${conv[@]}" --files "$list" || rc=1 ;;
    shard)
        sayrun "${SRUN[@]}" --ntasks="$NODES" --ntasks-per-node=1 --label \
            env A="$A" DATA="$DATA" OUT="$out" LIST="$list" PPT="$PPT" \
            CONT="${CONT:-}" NAME="$NAME" EXTRA="$EXTRA" ENGINE_JSON="$ENGINE_JSON" \
            bash -c '
                cd "$A" || exit 1
                echo "shard $SLURM_NODEID ($(hostname))"
                set -- -p "$DATA" -o "$OUT" --name "$NAME" $EXTRA
                [[ -n $ENGINE_JSON ]] && set -- "$@" --engine "$ENGINE_JSON"
                if [[ -n $CONT ]]; then
                    apptainer exec "$CONT" mpirun -np "$PPT" "$A/build/sdf2opmd" "$@" --files "$LIST"
                else
                    mpirun -np "$PPT" "$A/build/sdf2opmd" "$@" --files "$LIST"
                fi' || rc=1 ;;
    local)
        sayrun "$BIN" "${conv[@]}" --files "$list" || rc=1 ;;
    *) echo "unknown launch '$launch' in leg '$spec'" >&2; return 2 ;;
    esac
    local win; win=$(sec "$t0" "$(date +%s%N)")
    if [[ $DRY == 1 ]]; then echo "LEG $spec DRY"; return 0; fi

    local n=$(find "$out" -maxdepth 1 -name '*.bp' | wc -l)
    local b=$(du -sb "$out" | cut -f1)
    awk -v s="$spec" -v f=$(( hi - lo )) -v i="$n" -v bb="$b" -v w="$win" -v n="$NT" \
        'BEGIN{printf "RESULT   leg=%-16s files=%d iterations=%d bytes=%d wall=%.2fs %.1f MB/s %.2fs/file %.1f MB/s/rank\n", s, f, i, bb, w, bb/1e6/w, w/f, bb/1e6/w/n}'
    if (( rc != 0 )); then echo "LEG $spec FAILED (rc=$rc)" >&2; return 1; fi
    if (( n != hi - lo )); then
        echo "LEG $spec FAILED: $f inputs, $n iterations" >&2; return 1; fi
    if (( VALIDATE )); then
        local t2=$(date +%s%N)
        # validate exactly the files THIS leg converted: with COLD=rotate the
        # other slices were never written into this leg's series
        if [[ $launch == local ]]; then
            "${PYTHON:-python3}" "$VAL" "${slice[@]}" \
                --series "$out" "${DUMP[@]}" || return 1
        else
            "${SRUN[@]}" --ntasks=1 --ntasks-per-node=1 --label \
                "${wrap[@]}" "${PYTHON:-python3}" "$VAL" "${slice[@]}" \
                --series "$out" "${DUMP[@]}" || return 1
        fi
        echo "         validated in $(sec "$t2" "$(date +%s%N)")s (outside the window)"
    fi
    return 0
}

# ---- run the matrix ---------------------------------------------------------
fail=0
for ((i = 0; i < nl; ++i)); do
    run_leg "${LEGS[i]}" $i || { echo "leg ${LEGS[i]} failed"; fail=1; }
done

echo "==============================================================="
echo "matrix legs=${#LEGS[@]}  cold=$COLD  ranks=$NT  nodes=$NODES"
echo "  (each line above: RESULT leg=... ; grep RESULT this file to plot)"
[[ $fail == 0 ]] && echo "ALL LEGS PASSED" || echo "AT LEAST ONE LEG FAILED"
exit $fail
