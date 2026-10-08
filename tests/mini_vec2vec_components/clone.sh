#!/usr/bin/env bash
# Clone the official mini-vec2vec code (the notebook linear_vec2vec.ipynb) at the pinned commit into
# tests/mini_vec2vec_components/official/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=948de011eae17e7190991c99f2e5a4615a7a2a87
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/guy-dar/mini-vec2vec official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "mini-vec2vec @ $(git -C official rev-parse --short HEAD)"
