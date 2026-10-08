#!/usr/bin/env bash
# Clone the official RAE code (Zheng et al., 2026) at the commit of our submodule into tests/rae/official/ (unpatched).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/bytetriper/RAE official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "RAE @ $(git -C official rev-parse --short HEAD)"
