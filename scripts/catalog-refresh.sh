#!/bin/sh
# SPDX-License-Identifier: MIT
set -eu
cd "$(dirname "$0")/.."
if [ "$(scripts/window-open)" = open ]; then echo 'Catalog refresh paused: measurement window open' >&2; exit 75; fi
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
scripts/mlock work catalog-build -- swift build --product Arsenal
ARSENAL=.build/debug/Arsenal
# Metadata refresh is opt-in; release builds do not need network when the folders are already pinned.
if [ "${1:-}" = --pin ]; then scripts/mlock work catalog-pin -- "$ARSENAL" catalog pin --all; fi
scripts/mlock work catalog-validate -- "$ARSENAL" catalog validate --all
# Do not commit contributor edits automatically. Build only hashes committed model folders.
if ! git diff --quiet -- models Resources/kinds; then echo 'Commit the model pins, then rerun catalog-refresh.sh' >&2; exit 1; fi
scripts/mlock work catalog-generate -- "$ARSENAL" catalog build
scripts/mlock work catalog-sign -- "$ARSENAL" catalog sign
scripts/mlock work catalog-verify -- "$ARSENAL" catalog verify
