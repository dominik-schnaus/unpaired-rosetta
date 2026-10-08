#!/usr/bin/env bash
# Computes the per-subject encoders of Marcos-Manchón et al. (2026) and their cross-subject alignments with the
# authors' code (run ./clone.sh first) in the `platonic-brain` environment. Each stage is one SLURM array of
# single-node jobs or runs locally, and skips outputs that exist. Run the stages in order, each after the last ends.
#
#   ./run_pipeline.sh metadata      # NSD design, behavioural responses, nsdgeneral masks, index tables   (local)
#   ./run_pipeline.sh extract       # download the 1 mm betas and keep the nsdgeneral voxels             (8 CPU jobs)
#   ./run_pipeline.sh linear        # linear encoder: reliability, PCA 768, MCCA 128, ridge              (8 GPU jobs)
#   ./run_pipeline.sh mlp           # residual MLP on top of the linear encoder                          (8 GPU jobs)
#   ./run_pipeline.sh standardize   # standardize and average the repetitions of an image                (local)
#   ./run_pipeline.sh materialize   # write embeddings/NSD-private and embeddings/NSD-shared907          (local)
#   ./run_pipeline.sh align         # the authors' aligner: 10 runs x 56 ordered subject pairs           (GPU jobs)
#
# The last stage is the "platonic brain" baseline. `python -m experiments.domain_specific run` scores its maps.
# The authors' code does not seed these runs, so a rerun gives new maps of the same quality, not the same maps.
set -euo pipefail

PIPELINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$PIPELINE/../../.." && pwd)"
SCRIPTS="$PIPELINE/official/scripts"
cd "$ROOT"
DATA="${NSD_DATASET:-${UNPAIRED_ROSETTA_ROOT:-$ROOT/storage}/data/nsd}"
OUT="$DATA/runs/1mm"
LOGS="$DATA/logs"
mkdir -p "$OUT" "$LOGS"
SLURM="${UNPAIRED_ROSETTA_SLURM_PARTITION:+--partition=$UNPAIRED_ROSETTA_SLURM_PARTITION} --nodes=1 --ntasks=1 --output=$LOGS/%x_%A_%a.out"
GPU="--gres=gpu:1 ${UNPAIRED_ROSETTA_SLURM_EXCLUDE:+--exclude=$UNPAIRED_ROSETTA_SLURM_EXCLUDE}"
RUN="pixi run -e platonic-brain python"

case "${1:?stage}" in
  metadata)
    $RUN "$PIPELINE/metadata.py"
    ;;
  extract)
    sbatch $SLURM --job-name=nsd-extract --array=1-8 --cpus-per-task=3 --mem=64G --time=10:00:00 \
      --wrap="$RUN $PIPELINE/extract_betas.py --subject \$SLURM_ARRAY_TASK_ID"
    ;;
  linear)
    mkdir -p "$OUT/linear_embeddings"
    sbatch $SLURM $GPU --job-name=nsd-linear --array=1-8 --cpus-per-task=8 --mem=160G --time=08:00:00 \
      --wrap="$RUN $SCRIPTS/1_within_subject_linear.py --subjects \$SLURM_ARRAY_TASK_ID --rois 0 \
        --n_components_pca 768 --n_components_mcca 128 --betas_subfolder betas1mm \
        --output_folder $OUT/linear_embeddings --version v1 --print_eval"
    ;;
  mlp)
    mkdir -p "$OUT/mlp_embeddings"
    sbatch $SLURM $GPU --job-name=nsd-mlp --array=1-8 --cpus-per-task=4 --mem=64G --time=08:00:00 \
      --wrap="$RUN $SCRIPTS/2_within_subject_mlp.py --subjects \$SLURM_ARRAY_TASK_ID \
        --linear_embeddings_dir $OUT/linear_embeddings --embeddings_template 'ws_linear_v1_768_128_sub-{subject:02d}.pt' \
        --output_dir $OUT/mlp_embeddings --version v1 --print_eval"
    ;;
  standardize)
    mkdir -p "$OUT/mlp_embeddings_standarized"
    $RUN "$SCRIPTS/3_standarize_embeddings.py" --input_folder "$OUT/mlp_embeddings" \
      --input_template "ws_mlp_v1_128_768_sub-{subject:02d}.pt" --output_folder "$OUT/mlp_embeddings_standarized" \
      --output_template "avg_ws_mlp_v1_128_768_sub-{subject:02d}.pt" --subjects 1 2 3 4 5 6 7 8 --average --overwrite
    ;;
  materialize)
    NSD_DATASET="$DATA" pixi run python -c "from modalities import brain_scans; brain_scans.materialize()"
    ;;
  align)
    # 560 runs in 18 GPU jobs of 32 runs each (their scripts/mini-vec2vec_alignment.bash runs them one by one).
    sbatch $SLURM $GPU --job-name=nsd-align --array=0-17 --cpus-per-task=3 --mem=16G --time=12:00:00 \
      --wrap="$PIPELINE/align.sh $OUT \$SLURM_ARRAY_TASK_ID 32"
    ;;
  *)
    echo "unknown stage $1"; exit 1 ;;
esac
