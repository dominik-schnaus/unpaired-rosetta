#!/usr/bin/env bash
# Prepares the external code and weights of the text-to-image experiments. Safe to run again.
#   1. The submodules external/RAE and external/Scale-RAE at their pinned commits.
#   2. Our RAE patch (conditioning on image embeddings instead of class labels, preemptible training).
#   3. The pretrained RAE decoder and latent statistics (1.6 GB, nyu-visionx/RAE-collections), linked as
#      external/RAE/models, where the RAE configs expect them.
# The trained diffusion model is downloaded or trained separately (see README.md).
set -euo pipefail
REPOSITORY=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPOSITORY"

git submodule update --init external/RAE external/Scale-RAE

PATCH="$REPOSITORY/external/patches/rae_embedguidance.diff"
if git -C external/RAE apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "the RAE patch is already applied"
else
    git -C external/RAE apply "$PATCH"
    echo "applied the RAE patch"
fi

WEIGHTS="${UNPAIRED_ROSETTA_ROOT:-$REPOSITORY/storage}/data/rae"
for file in decoders/dinov2/wReg_base/ViTXL_n08/model.pt stats/dinov2/wReg_base/imagenet1k/stat.pt; do
    [ -e "$WEIGHTS/$file" ] || pixi run -e text2image hf download nyu-visionx/RAE-collections "$file" --local-dir "$WEIGHTS"
done
[ -e external/RAE/models ] || ln -s "$WEIGHTS" external/RAE/models
