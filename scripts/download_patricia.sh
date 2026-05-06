#!/usr/bin/env bash
# Fetch the Patricia v3 (AVX2) Linux binary into bin/. Run once after cloning.
# Requires: gh (GitHub CLI). For non-AVX2 CPUs, swap 'patricia_v3' for 'patricia_v2'.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin
gh release download 5 --repo Adam-Kulju/Patricia --pattern 'patricia_v3' --dir bin/ --clobber
mv bin/patricia_v3 bin/patricia
chmod +x bin/patricia
echo "Patricia v3 (AVX2) ready at bin/patricia"
