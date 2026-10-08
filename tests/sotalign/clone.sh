#!/usr/bin/env bash
# Clone the official SOTAlign code (Roschmann et al.) at the commit of external/SOTAlign into tests/sotalign/official/
# (gitignored, unpatched). At this commit the repository only releases the KLOT divergence (src/klot.py).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=4a810a1756599e5ff89d5de3321a3a7fb829ef2d
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/ExplainableML/SOTAlign official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "SOTAlign @ $(git -C official rev-parse --short HEAD)"
