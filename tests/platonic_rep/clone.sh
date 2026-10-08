#!/usr/bin/env bash
# Clone the platonic-rep code (Huh et al., 2024) at the pinned commit into tests/platonic_rep/official/ (gitignored).
# Wang et al. (2026) reuse its CKA with clipping.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/minyoungg/platonic-rep official
fi
git -C official fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C official checkout --quiet "$COMMIT"
echo "platonic-rep @ $(git -C official rev-parse --short HEAD)"
