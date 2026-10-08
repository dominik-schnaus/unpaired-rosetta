#!/usr/bin/env bash
# Clone the authors' code (Marcos-Manchón et al., 2026) at the commit the paper used into pipeline/official/
# (gitignored). device.patch only adds a --device flag to their alignment script (stage 4).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=1d50080480cca6262b9bb7d1a91a31e6aa5da562
if [ ! -d official/.git ]; then
    git clone --quiet https://github.com/memory-formation/platonic-representations-fmri official
fi
git -C official checkout --quiet "$COMMIT"
if git -C official apply --check ../device.patch 2>/dev/null; then
    git -C official apply ../device.patch
fi
echo "platonic-representations-fmri @ $(git -C official rev-parse --short HEAD) (patched)"
