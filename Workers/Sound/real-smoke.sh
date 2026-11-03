#!/bin/bash
# SPDX-License-Identifier: MIT
# Usage: real-smoke.sh <stable-audio-3-medium|stable-audio-3-small-sfx|re-use> <worker-path> <metallib-path>
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:?model required}; BINARY=${2:?absolute worker path required}; METALLIB=${3:?absolute metallib path required}
case "$MODEL" in stable-audio-3-medium|stable-audio-3-small-sfx|re-use) ;; *) echo 'Unknown sound model' >&2; exit 2 ;; esac
SUPPORT=/tmp/arsenal-mb-v2-sound/run-support
STORE="$HOME/Library/Application Support/Arsenal-dev/weights/mb-v2-sound/get-support"
mkdir -p "$SUPPORT" "$STORE" docs/evidence/mb-v2-sound
exec scripts/mlock gpulock "v2-sound-$MODEL" -- perl -e 'alarm 180; exec @ARGV' env \
 ARSENAL_SUPPORT_DIR="$SUPPORT" ARSENAL_CACHE_DIR=/tmp/arsenal-mb-v2-sound/cycle-cache \
 ARSENAL_SOUND_GET_SUPPORT="$STORE" \
 ARSENAL_SOUND_REAL_MODEL="$MODEL" ARSENAL_SOUND_BINARY="$BINARY" \
 ARSENAL_SOUND_EVIDENCE="$PWD/docs/evidence/mb-v2-sound" ARSENAL_MLX_METALLIB="$METALLIB" \
 scripts/test.sh --package-path Workers -c release --skip-build --filter realGetLoadRunUnload
