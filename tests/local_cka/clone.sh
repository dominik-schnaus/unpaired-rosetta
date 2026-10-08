#!/usr/bin/env bash
# Clone the official Local CKA code (Maniparambil et al., 2024) at the pinned commit into tests/local_cka/official/.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=1b4ee3e3492575c8a8faa40afd1b19fbd963821e
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/mayug/0-shot-llm-vision official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "0-shot-llm-vision @ $(git -C official rev-parse --short HEAD)"
