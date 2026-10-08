#!/usr/bin/env bash
# Clone the official CycleReward code (Bahng et al., 2025) at the pinned commit into tests/cyclereward/official/.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=36e10773d1385ed57d3cecf15a352d48705fd0fe
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/hjbahng/cyclereward official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "hjbahng/cyclereward @ $(git -C official rev-parse --short HEAD)"
