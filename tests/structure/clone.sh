#!/usr/bin/env bash
# Clone the official STRUCTURE code (Gröger et al.) at the commit of external/STRUCTURE into tests/structure/official/
# (gitignored, unpatched).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=f7472070c0a1553faffd9e06fa7218f6710c518a
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/mlbio-epfl/STRUCTURE official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "STRUCTURE @ $(git -C official rev-parse --short HEAD)"
