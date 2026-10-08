#!/usr/bin/env bash
# Clone the official SUE code at the commit of our submodule into tests/sue/official/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=2be0c4ae05c2dec031ac663162eb71f543a6292d
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/shaham-lab/SUE official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "SUE @ $(git -C official rev-parse --short HEAD)"
