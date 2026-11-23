#!/bin/bash
# SLURM submission for the EPOCH 3D LWFA benchmark run (LRZ Orion pattern).
#
#   ./epoch_job.sh                 # uses decks/lwfa_bench.deck
#   DECK=decks/lwfa.deck ./epoch_job.sh
#
# Dumps land in ./data/. Adjust --nodes/--time/--partition to your intent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECK=${DECK:-$HERE/decks/lwfa_bench.deck}
DATADIR=${DATADIR:-$HERE/data}

mkdir -p "$DATADIR"
cp "$DECK" "$DATADIR/epoch.deck"
cp "$HERE/run_epoch.sh" "$DATADIR/run-file.sh"
chmod +x "$DATADIR/run-file.sh"

sbatch --nodes "${NODES:-4}" \
       --tasks-per-node 32 \
       --no-requeue \
       --job-name "${JOB_NAME:-lwfa-bench}" \
       --mem-per-cpu 1960 \
       --mail-type ALL \
       --mail-user d.bertini@gsi.de \
       --partition "${PARTITION:-long}" \
       --time "${TIME:-04:00:00}" \
       -D "$DATADIR" \
       -o "%j.out.log" -e "%j.err.log" \
       ./run-file.sh
echo "submitted $DECK -> $DATADIR"
