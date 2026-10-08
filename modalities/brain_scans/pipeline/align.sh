#!/usr/bin/env bash
# Usage: align.sh <output root> <task> <count>
# Runs <count> of the authors' 560 alignment runs, starting at run <task> * <count>. There are 10 runs for every
# ordered pair of the eight subjects, with the pairs in the order of the authors' script.
set -euo pipefail
OUT="$1"; TASK="$2"; COUNT="$3"
PIPELINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIRS=()
for x in 1 2 3 4 5 6 7 8; do for y in 1 2 3 4 5 6 7 8; do [ "$x" != "$y" ] && PAIRS+=("$x $y"); done; done
for index in $(seq $((TASK * COUNT)) $((TASK * COUNT + COUNT - 1))); do
  [ "$index" -ge $((10 * ${#PAIRS[@]})) ] && break
  RUN=$((index / ${#PAIRS[@]} + 1))
  read -r X Y <<< "${PAIRS[$((index % ${#PAIRS[@]}))]}"
  echo "run $RUN: sub-$X -> sub-$Y"
  pixi run -e platonic-brain python "$PIPELINE/official/scripts/4_mini_vec2vec_alignment.py" \
    --subject_x "$X" --subject_y "$Y" --input_folder "$OUT/mlp_embeddings_standarized" \
    --input_template "avg_ws_mlp_v1_128_768_sub-{subject:02d}.pt" --output_folder "$OUT/alignments/final_alignment_$RUN" \
    --n_runs_refinement 3000 --device cuda
done
