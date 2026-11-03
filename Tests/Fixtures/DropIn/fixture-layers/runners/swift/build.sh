#!/bin/bash
# SPDX-License-Identifier: MIT
# Agent-only fixture build, never invoked by Arsenal. Keep generated SwiftPM trees outside the approved folder.
set -euo pipefail
package="$(cd "$(dirname "$0")" && pwd)"
scratch="${ARSENAL_FIXTURE_BUILD:?set an isolated build directory outside the model folder}"
clt=/Library/Developer/CommandLineTools
mkdir -p "$scratch" "$package/bin"
flags=()
# Same Swift 6.4 compatibility fix as scripts/build.sh; do not make swift_initBorrow a strong import.
if "$clt/usr/bin/swift" --version | /usr/bin/grep -Eq 'Swift version 6\.([4-9]|[1-9][0-9])'; then
  cat > "$scratch/weak-borrow.c" <<'C'
extern void swift_initBorrow(void) __attribute__((weak_import));
__attribute__((used)) void *const arsenal_weak_swift_initBorrow = (void *)&swift_initBorrow;
C
  "$clt/usr/bin/clang" -target arm64-apple-macos26.0 -c "$scratch/weak-borrow.c" -o "$scratch/weak-borrow.o"
  flags=(-Xlinker "$scratch/weak-borrow.o" -Xlinker -weak_reference_mismatches -Xlinker weak)
fi
DEVELOPER_DIR="$clt" "$clt/usr/bin/swift" build --package-path "$package" --scratch-path "$scratch/build" --build-system native -c release "${flags[@]}"
cp "$scratch/build/release/fixture-layers" "$package/bin/fixture-layers"
codesign --force --sign - --timestamp=none "$package/bin/fixture-layers"
codesign --verify --strict "$package/bin/fixture-layers"
