#!/usr/bin/env bash
# Clone the official t2v_metrics code (VQAScore, Lin et al., 2024) at the commit of its pip release 1.1 into
# tests/vqascore/official/.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=2df936838bfb8ec8e4e071c58e93c23e81fed2c8
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/linzhiqiu/t2v_metrics official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "linzhiqiu/t2v_metrics @ $(git -C official rev-parse --short HEAD)"
