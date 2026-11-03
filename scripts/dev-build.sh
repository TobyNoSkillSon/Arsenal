#!/bin/bash
# Build the app, its tests and the one native graph (worker, lab tools, every family's tests, dev helpers) into the
# per-worktree debug trees: .build/ and Workers/.build/. MLX compiles once per configuration, in Workers/.build.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${ARSENAL_DEV_LOCKED:-}" && "${ARSENAL_WORK_LOCKED_ROOT:-}" != "$PWD" ]]; then
  exec env ARSENAL_DEV_LOCKED=1 nice -n 10 scripts/mlock work dev-build -- "$PWD/scripts/dev-build.sh" "$@"
fi
source scripts/dev-lib.sh
scripts/lint.sh
prepare_debug
swift_debug test list --jobs "${ARSENAL_BUILD_JOBS:-2}" "$@" > "$PWD/.build/test-list.txt"
record_debug
native_debug_build
