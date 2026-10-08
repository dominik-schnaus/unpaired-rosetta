#!/usr/bin/env bash
# Clone the official CLIPScore code (Hessel et al., 2021) at the pinned commit into tests/clipscore/official/.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=1036465276513621f77f1c2208d742e4a430781f
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/jmhessel/clipscore official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "jmhessel/clipscore @ $(git -C official rev-parse --short HEAD)"
