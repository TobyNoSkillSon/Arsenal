#!/bin/bash
# SPDX-License-Identifier: MIT
# Builds the standalone arsenal-worker-d1 (gate, benchmark, wire checker) as Workers/Decisions/build.sh builds its
# worker: the Command Line Tools' Swift compiles Swift/C++, Xcode's Metal toolchain compiles MLX's pinned shaders (no
# fast math) into the mlx-swift_Cmlx.bundle MLX finds next to the binary. The app ships this runner inside
# Workers/Native (family d1) instead. D1_BUILD_DIR moves the build out of the folder (default: .build here).
# Run through the lock:  scripts/mlock work d1-build -- models/d1-3b/runners/swift/build.sh
set -euo pipefail
root=$(cd -- "$(dirname "$0")" && pwd)
scratch=${D1_BUILD_DIR:-$root/.build}
clt=/Library/Developer/CommandLineTools
xcode=${ARSENAL_XCODE_DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
swift_build() { DEVELOPER_DIR="$clt" "$clt/usr/bin/swift" build --package-path "$root" --scratch-path "$scratch" -c release --build-system native "$@"; }
swift_build --product arsenal-worker-d1
swift_build --product d1-check
source="$scratch/checkouts/mlx-swift"
[[ $(git -C "$source" rev-parse HEAD) == 901941965d82e4a216d4d117231d847d194c563d ]]
bin=$(swift_build --show-bin-path)
bundle="$bin/mlx-swift_Cmlx.bundle"
if [[ ! -s "$bundle/Contents/Resources/default.metallib" ]]; then
    metal="$scratch/split-metal"
    mkdir -p "$metal" && : > "$metal/compile.log"
    sdk=$(DEVELOPER_DIR="$xcode" xcrun --show-sdk-path)
    metal_flags=(-Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions)
    objects=()
    while IFS= read -r file; do
        output="$metal/$(basename "$file" .metal).air"
        DEVELOPER_DIR="$xcode" xcrun metal -c -target air64-apple-macos14.0 -isysroot "$sdk" \
            "${metal_flags[@]}" "$file" -o "$output" 2>>"$metal/compile.log" || { tail -20 "$metal/compile.log" >&2; exit 1; }
        objects+=("$output")
    done < <(find "$source/Source/Cmlx/mlx-generated/metal" -name '*.metal' | sort)
    [[ ${#objects[@]} == 10 ]]
    mkdir -p "$bundle/Contents/Resources"
    DEVELOPER_DIR="$xcode" xcrun metallib "${objects[@]}" -o "$bundle/Contents/Resources/default.metallib"
    cat > "$bundle/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>dev.arsenal.mlx-swift-Cmlx</string>
<key>CFBundleName</key><string>mlx-swift_Cmlx</string>
<key>CFBundlePackageType</key><string>BNDL</string>
<key>CFBundleVersion</key><string>1</string>
</dict></plist>
PLIST
fi
if nm -u "$bin/arsenal-worker-d1" | grep -Eq '_swift_(init|end)Borrow'; then
    echo 'arsenal-worker-d1 links Swift 6.4 borrow symbols; build with the CLT Swift 6.3.3.' >&2; exit 1
fi
printf '%s\n' "$bin/arsenal-worker-d1"
