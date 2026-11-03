#!/bin/bash
# SPDX-License-Identifier: MIT
# Adapted from Projects/Vella/Worker/build-split.sh (Vella; relicensed MIT by its author, 2 Oct 2026).
# Builds the kernels smoke the way Verdict and Vella do (Vella Worker/build-split.sh): the Command Line Tools'
# Swift 6.3.3 compiles Swift/C++, Xcode's Metal toolchain compiles MLX's pinned shaders (no fast math). Xcode 27's
# Swift 6.4 output does not launch on macOS 26.6 (_swift_initBorrow), so such a binary is rejected.
# Run through the lock:  scripts/mlock work kernels-build -- Workers/Kernels/build.sh
set -euo pipefail
root=$(cd -- "$(dirname "$0")" && pwd)
clt=/Library/Developer/CommandLineTools
xcode=${ARSENAL_XCODE_DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
DEVELOPER_DIR="$clt" "$clt/usr/bin/swift" build --package-path "$root" -c release --build-system native
source="$root/.build/checkouts/mlx-swift"
[[ $(git -C "$source" rev-parse HEAD) == 901941965d82e4a216d4d117231d847d194c563d ]]
bin=$(DEVELOPER_DIR="$clt" "$clt/usr/bin/swift" build --package-path "$root" -c release --build-system native --show-bin-path)
metal="$root/.build/split-metal"
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
bundle="$bin/mlx-swift_Cmlx.bundle"
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
if nm -u "$bin/arsenal-kernels-smoke" | grep -Eq '_swift_(init|end)Borrow'; then
    echo 'arsenal-kernels-smoke links Swift 6.4 borrow symbols; build with the CLT Swift 6.3.3.' >&2; exit 1
fi
{
    DEVELOPER_DIR="$clt" "$clt/usr/bin/swift" --version
    DEVELOPER_DIR="$xcode" xcrun metal --version
    echo "Metal flags: ${metal_flags[*]}"
    shasum -a 256 "$bin/arsenal-kernels-smoke" "$bundle/Contents/Resources/default.metallib" "$root/Package.resolved"
} > "$root/.build/build-provenance.txt" 2>&1
printf '%s\n' "$bin/arsenal-kernels-smoke"
