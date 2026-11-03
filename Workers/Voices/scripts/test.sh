#!/bin/bash
# SPDX-License-Identifier: MIT
# Compatibility entry; the repo owns the shared CLT Swift Testing recipe.
set -euo pipefail
root=$(cd -- "$(dirname "$0")/../../.." && pwd)
# The voices suite lives in the one native graph (Workers/Package.swift); release, as before, for its speed checks.
exec "$root/scripts/test.sh" --package-path "$root/Workers" -c release -Xswiftc -enable-testing --filter VoicesTests "$@"
