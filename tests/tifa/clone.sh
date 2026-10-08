#!/usr/bin/env bash
# Clone the official TIFA code (Hu et al., 2023) at the pinned commit into tests/tifa/official/.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=8e37050f44ab4ee681e4251f3c8a9d4cc86e901e
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/Yushi-Hu/tifa official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "Yushi-Hu/tifa @ $(git -C official rev-parse --short HEAD)"
