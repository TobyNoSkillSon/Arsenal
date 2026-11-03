#!/bin/bash
set -euo pipefail
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
mkdir -p "$PWD/.build/conformance"
run_support="$(mktemp -d "$PWD/.build/conformance/support.XXXXXX")"
export ARSENAL_SUPPORT_DIR="${ARSENAL_SUPPORT_DIR:-$run_support}"
export ARSENAL_CACHE_DIR="${ARSENAL_CACHE_DIR:-$PWD/.build/conformance/cache}"
mkdir -p "$ARSENAL_SUPPORT_DIR" "$ARSENAL_CACHE_DIR"
[ -x tests/conformance/.venv/bin/python ] || uv venv tests/conformance/.venv --python 3.12 --quiet
uv pip sync --python tests/conformance/.venv/bin/python tests/conformance/requirements.lock --quiet
if [ "${1:-}" = "--app-chat" ]; then
  shift
  exec scripts/mlock gpulock api-chat-conformance -- tests/conformance/.venv/bin/python Tests/conformance/chat-app-conformance.py "$@"
fi
source scripts/dev-lib.sh
swift_debug build --scratch-path "${ARSENAL_SWIFT_SCRATCH:-.build}" --product arsenal-testserver --jobs "${ARSENAL_BUILD_JOBS:-2}" > "$ARSENAL_CACHE_DIR/build.log" 2>&1 || { tail -30 "$ARSENAL_CACHE_DIR/build.log"; exit 1; }
exec tests/conformance/.venv/bin/python tests/conformance/conformance.py "$@"
