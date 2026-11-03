#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if scripts/window-open >/dev/null; then echo 'API conformance paused: measurement window open.'; exit 75; fi
exec scripts/mlock work api-conformance -- bash tests/conformance/run-inner.sh "$@"
