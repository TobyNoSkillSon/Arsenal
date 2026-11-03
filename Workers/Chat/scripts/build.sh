#!/bin/bash
# SPDX-License-Identifier: MIT
# Family split build adapted from Vella/Worker/build-split.sh (Vireo source commit recorded in PORTED-FROM.md).
set -euo pipefail
root=$(cd -- "$(dirname "$0")/.." && pwd)
clt=/Library/Developer/CommandLineTools
xcode=/Applications/Xcode.app/Contents/Developer
export DEVELOPER_DIR="$clt"
"$clt/usr/bin/swift" build --package-path "$root" -c release --build-system native -j 4
bin=$("$clt/usr/bin/swift" build --package-path "$root" -c release --show-bin-path)
if nm -u "$bin/arsenal-worker-chat" | grep -E '_swift_(initBorrow|endBorrow)'; then echo 'Incompatible Swift runtime references.' >&2; exit 1; fi
source="$root/.build/checkouts/mlx-swift"
[[ $(git -C "$source" rev-parse HEAD) == 901941965d82e4a216d4d117231d847d194c563d ]]
bundle="$bin/mlx-swift_Cmlx.bundle/Contents/Resources"
mkdir -p "$bundle"
# A clean source build is reproducible; an existing metallib for the identical family pin may be clone-copied during development.
metal="$root/.build/split-metal"; mkdir -p "$metal"
sdk=$(DEVELOPER_DIR="$xcode" xcrun --show-sdk-path)
objects=()
while IFS= read -r file; do
  output="$metal/$(basename "$file" .metal).air"
  DEVELOPER_DIR="$xcode" xcrun metal -c -target air64-apple-macos14.0 -isysroot "$sdk" -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions "$file" -o "$output"
  objects+=("$output")
done < <(find "$source/Source/Cmlx/mlx-generated/metal" -name '*.metal' | sort)
[[ ${#objects[@]} == 10 ]]
DEVELOPER_DIR="$xcode" xcrun metallib "${objects[@]}" -o "$bundle/default.metallib"
printf '%s\n' "$bin/arsenal-worker-chat"
