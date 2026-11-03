#!/bin/bash
# SPDX-License-Identifier: MIT
# Lab build of this runner (arsenal-worker-flux, flux-lab) with the Command Line Tools Swift and MLX's pinned shaders
# compiled by Xcode's Metal toolchain, the recipe of Workers/Decisions/build.sh. The shipped binary is Workers/Native.
# Builds outside the model folder (the folder's bytes are catalogued): $ARSENAL_FLUX_BUILD or <repo>/.build/flux2-klein-swift.
set -euo pipefail
root=$(cd -- "$(dirname "$0")" && pwd)
repo=$(cd -- "$root/../../../.." && pwd)
scratch=${ARSENAL_FLUX_BUILD:-$repo/.build/flux2-klein-swift}
clt=/Library/Developer/CommandLineTools
xcode=${ARSENAL_XCODE_DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
swiftc() { DEVELOPER_DIR="$clt" "$clt/usr/bin/swift" "$@" --package-path "$root" --scratch-path "$scratch" -c release --build-system native; }
swiftc build
source="$scratch/checkouts/mlx-swift"
[[ $(git -C "$source" rev-parse HEAD) == 901941965d82e4a216d4d117231d847d194c563d ]]
bin=$(swiftc build --show-bin-path)
bundle="$bin/mlx-swift_Cmlx.bundle"
if [[ ! -s "$bundle/Contents/Resources/default.metallib" ]]; then
  metal="$scratch/split-metal"; mkdir -p "$metal" && : > "$metal/compile.log"
  sdk=$(DEVELOPER_DIR="$xcode" xcrun --show-sdk-path)
  flags=(-Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions)
  objects=()
  while IFS= read -r file; do
    out="$metal/$(basename "$file" .metal).air"
    DEVELOPER_DIR="$xcode" xcrun metal -c -target air64-apple-macos14.0 -isysroot "$sdk" "${flags[@]}" "$file" -o "$out" 2>>"$metal/compile.log" \
      || { tail -20 "$metal/compile.log" >&2; exit 1; }
    objects+=("$out")
  done < <(find "$source/Source/Cmlx/mlx-generated/metal" -name '*.metal' | sort)
  [[ ${#objects[@]} == 10 ]]
  mkdir -p "$bundle/Contents/Resources"
  DEVELOPER_DIR="$xcode" xcrun metallib "${objects[@]}" -o "$bundle/Contents/Resources/default.metallib"
  printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>' \
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
    '<plist version="1.0"><dict><key>CFBundleIdentifier</key><string>dev.arsenal.mlx-swift-Cmlx</string>' \
    '<key>CFBundleName</key><string>mlx-swift_Cmlx</string><key>CFBundlePackageType</key><string>BNDL</string>' \
    '<key>CFBundleVersion</key><string>1</string></dict></plist>' > "$bundle/Contents/Info.plist"
fi
printf '%s\n' "$bin"
