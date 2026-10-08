#!/usr/bin/env bash
# Clone the official vec2vec code (Jha et al.) at the pinned commit into tests/vec2vec/official/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=5efb3c02ee908847581c6488cf9c8ed6dace3c29
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/rjha18/vec2vec official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "vec2vec @ $(git -C official rev-parse --short HEAD)"
