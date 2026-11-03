#!/bin/bash
# Shared debug flags keep build and test in the same SwiftPM graph.
source "$(dirname "${BASH_SOURCE[0]}")/pkg-lib.sh"
CLT=/Library/Developer/CommandLineTools
FRAMEWORKS="$CLT/Library/Developer/Frameworks"
swift_debug() {
  DEVELOPER_DIR="$CLT" "$CLT/usr/bin/swift" "$@" --build-system native \
    -Xswiftc -F -Xswiftc "$FRAMEWORKS" -Xlinker -F -Xlinker "$FRAMEWORKS" \
    -Xlinker -rpath -Xlinker "$FRAMEWORKS" -Xlinker -rpath -Xlinker "$CLT/Library/Developer/usr/lib"
}
# One phase's wall time on stderr, and into ARSENAL_PHASES_FILE when set. Keeps the command's exit status. Wrap only
# single commands: a shell function run inside `||` loses errexit for its own steps.
timed() {
  local name="$1" start=$SECONDS status=0; shift
  "$@" || status=$?
  echo "time: $name $((SECONDS - start))s$( (( status == 0 )) || echo " (exit $status)")" >&2
  if [[ -n "${ARSENAL_PHASES_FILE:-}" ]]; then printf '%s\t%s\t%s\n' "$name" "$((SECONDS - start))" "$status" >> "$ARSENAL_PHASES_FILE"; fi
  return "$status"
}
# Entry scripts (check.sh, ready.sh): a new log directory per run, never reused or truncated, everything also on the
# terminal, and each phase's wall time at the end. Per-suite logs land in the same directory (ARSENAL_TEST_LOG_DIR).
entry_log() {
  ENTRY_KIND="$1"
  mkdir -p "$PWD/.build/logs"
  ENTRY_DIR="$(mktemp -d "$PWD/.build/logs/$ENTRY_KIND-$(date +%Y%m%d-%H%M%S)-XXXX")"
  export ARSENAL_TEST_LOG_DIR="$ENTRY_DIR" ARSENAL_PHASES_FILE="$ENTRY_DIR/phases.tsv"
  : > "$ARSENAL_PHASES_FILE"
  exec > >(tee "$ENTRY_DIR/run.log") 2>&1
  trap 'entry_summary $?' EXIT
  echo "$ENTRY_KIND: $(git rev-parse --short HEAD)$([[ -z "$(git status --porcelain --untracked-files=no)" ]] || echo ' + uncommitted changes'), log $ENTRY_DIR/run.log"
}
# Compile width for the entry scripts: on AC power with no keyboard or pointer input
# for 10 minutes, every core; otherwise 2 jobs. Decided once at the start; ARSENAL_BUILD_JOBS set by the caller wins.
entry_jobs() {
  local idle
  if [[ -z "${ARSENAL_BUILD_JOBS:-}" ]]; then
    idle="$(ioreg -c IOHIDSystem | awk '/HIDIdleTime/ { print int($NF / 1000000000); exit }')"
    if pmset -g batt | grep -q "AC Power" && (( ${idle:-0} >= 600 )); then ARSENAL_BUILD_JOBS="$(sysctl -n hw.ncpu)"; else ARSENAL_BUILD_JOBS=2; fi
  fi
  export ARSENAL_BUILD_JOBS
  echo "$ENTRY_KIND: $ARSENAL_BUILD_JOBS compile jobs at nice 19"
}
entry_summary() {
  local status="$1"
  echo "--- $ENTRY_KIND phases, wall seconds (the native graph and release compile run beside the app suites)"
  awk -F'\t' '{ printf "  %-16s %6ss%s\n", $1, $2, ($3 == 0 ? "" : "  exit " $3) }' "$ARSENAL_PHASES_FILE"
  if (( status == 0 )); then echo "$ENTRY_KIND: passed; log $ENTRY_DIR/run.log"; else echo "$ENTRY_KIND: FAILED (exit $status); log $ENTRY_DIR/run.log"; fi
}
prepare_debug() { /usr/bin/python3 scripts/incremental.py prepare "$PWD/.build" debug; }
record_debug() { /usr/bin/python3 scripts/incremental.py record "$PWD/.build" debug; }

# The one native graph (Workers/Package.swift): arsenal-worker-native, the lab tools and every family's tests, with MLX
# compiled once per configuration in Workers/.build. Its dev helpers mirror the bundle's Helpers/: a copy of the worker,
# one arsenal-worker-<family> link per family (the name it dispatches on) and the shared no-fast-math metallib, so
# `scripts/model-check.py smoke --worker Workers/.build/helpers/arsenal-worker-<family>` runs the shipped dispatch path.
NATIVE_PACKAGE="$PWD/Workers"
NATIVE_HELPERS="$NATIVE_PACKAGE/.build/helpers"
native_test_binary() { echo "$NATIVE_PACKAGE/.build/debug/ArsenalWorkersPackageTests.xctest/Contents/MacOS/ArsenalWorkersPackageTests"; }
native_debug_build() {
  local bundle family
  mkdir -p "$NATIVE_PACKAGE/.build"
  timed native-build swift_debug test list --package-path "$NATIVE_PACKAGE" --jobs "${ARSENAL_BUILD_JOBS:-2}" > "$NATIVE_PACKAGE/.build/test-list.txt"
  bundle="$(mlx_metallib_bundle "$NATIVE_PACKAGE/.build/checkouts/mlx-swift")"
  rm -rf "$NATIVE_HELPERS"; mkdir -p "$NATIVE_HELPERS"
  # A clone, not a link: the worker resolves the metallib beside the path it was started from.
  cp -c "$NATIVE_PACKAGE/.build/debug/$NATIVE_EXE" "$NATIVE_HELPERS/$NATIVE_EXE"
  for family in "${NATIVE_FAMILIES[@]}"; do ln -s "$NATIVE_EXE" "$NATIVE_HELPERS/arsenal-worker-$family"; done
  cp -cR "$bundle" "$NATIVE_HELPERS/mlx-swift_Cmlx.bundle"
}
