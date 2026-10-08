#!/usr/bin/env bash
# Clone the official SCOT code at the pinned commit into tests/scot_plus/official/ (gitignored).
# The repository also holds the SCOT+ sources.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=14649be6e14017dcfe7ba619091b33d1df55f6a9
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/rsinghlab/SCOT official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "SCOT @ $(git -C official rev-parse --short HEAD)"
