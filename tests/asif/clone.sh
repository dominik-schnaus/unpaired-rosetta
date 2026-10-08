#!/usr/bin/env bash
# Clone the official ASIF code (Norelli et al., 2023) at the pinned commit into tests/asif/official/.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=418ab249d16df7d7b8f49dc6024302800e714d71
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/noranta4/ASIF official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "ASIF @ $(git -C official rev-parse --short HEAD)"
