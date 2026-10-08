#!/usr/bin/env bash
# Clone the official code that the low-rank Gromov-Wasserstein ablation (Table 2) is compared with into
# tests/lowrank_gw/official/ (gitignored).
#   LinearGromov: Scetbon, Peyré and Cuturi (2022), github.com/meyerscetbon/LinearGromov
#   SCOT: Demetci et al. (2020), github.com/rsinghlab/SCOT (only src/, for the geodesic costs of the single-cell data)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p official
LINEAR_GROMOV=4610075ad40c87c8a60867a3f586fa4a36a3f294
SCOT=14649be6e14017dcfe7ba619091b33d1df55f6a9
if [ ! -d official/LinearGromov/.git ]; then
    git clone --quiet https://github.com/meyerscetbon/LinearGromov official/LinearGromov
fi
git -C official/LinearGromov checkout --quiet "$LINEAR_GROMOV"
if [ ! -d official/SCOT/.git ]; then
    git clone --quiet --filter=blob:none --sparse https://github.com/rsinghlab/SCOT official/SCOT
    git -C official/SCOT sparse-checkout set src
fi
git -C official/SCOT checkout --quiet "$SCOT"
echo "LinearGromov @ $(git -C official/LinearGromov rev-parse --short HEAD), SCOT @ $(git -C official/SCOT rev-parse --short HEAD)"
