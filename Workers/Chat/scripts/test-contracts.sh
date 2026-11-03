#!/bin/bash
# SPDX-License-Identifier: MIT
set -euo pipefail
root=$(cd -- "$(dirname "$0")/../../.." && pwd)
# The chat suite runs in the one native graph (Workers/Package.swift).
exec "$root/scripts/test.sh" --native --filter ChatContractTests "$@"
