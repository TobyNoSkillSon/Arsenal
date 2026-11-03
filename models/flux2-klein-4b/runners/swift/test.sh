#!/bin/bash
# SPDX-License-Identifier: MIT
# Contract tests and tiny GPU layout checks (no checkpoint weights); reuse the runner metallib.
set -euo pipefail
root=$(cd -- "$(dirname "$0")" && pwd)
repo=$(cd -- "$root/../../../.." && pwd)
release=${ARSENAL_FLUX_BUILD:-$repo/.build/flux2-klein-swift}
scratch=$release-debug
clt=/Library/Developer/CommandLineTools
fw="$clt/Library/Developer/Frameworks"
# Swift Testing from the Command Line Tools, as scripts/test.sh does.
swiftc() {
  DEVELOPER_DIR="$clt" "$clt/usr/bin/swift" "$@" --package-path "$root" --scratch-path "$scratch" --build-system native \
    -Xswiftc -F -Xswiftc "$fw" -Xlinker -F -Xlinker "$fw" -Xlinker -rpath -Xlinker "$fw" -Xlinker -rpath -Xlinker "$clt/Library/Developer/usr/lib"
}
swiftc build --build-tests
metallib=$(find "$release" -path '*mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib' | head -1)
[[ -s "$metallib" ]] || { echo "Run build.sh first (it compiles the MLX metallib)." >&2; exit 1; }
bin=$(swiftc build --show-bin-path)
mkdir -p "$bin/mlx-swift_Cmlx.bundle/Contents/Resources"
cp "$metallib" "$bin/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
cp "$(dirname "$(dirname "$metallib")")/Info.plist" "$bin/mlx-swift_Cmlx.bundle/Contents/Info.plist"
# xctest loads the tests as a bundle: MLX also looks for mlx.metallib next to the binary that holds it.
for dir in "$bin" "$bin"/*.xctest/Contents/MacOS; do cp "$metallib" "$dir/mlx.metallib"; done
swiftc test "$@"
