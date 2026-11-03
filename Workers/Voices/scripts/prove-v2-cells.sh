#!/bin/bash
# SPDX-License-Identifier: MIT
# Functional service cycles only, one bounded GPU job per row; no benchmark claims.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
helpers=${1:?usage: prove-v2-cells.sh HELPERS < CELLS (model dtype fast|exact)}
support="$HOME/Library/Application Support/Arsenal-dev/weights/voices/v2-service-support"
proof="$root/Workers/Voices/Proof/.build/debug/voices-get-proof"
evidence="$root/docs/evidence/mb-v2-voices"
mkdir -p "$evidence"
failed=0
while read -r model dtype mode; do
    [[ -n "$model" && "$model" != \#* ]] || continue
    tag="$model-$dtype-$mode"
    if [[ -n "${ARSENAL_PROOF_LABEL:-}" ]]; then tag="$tag-$ARSENAL_PROOF_LABEL"; fi
    if "$root/scripts/mlock" gpulock "voices-$tag" -- perl -e 'alarm 180; exec @ARGV' \
        env ARSENAL_SUPPORT_DIR="$support" ARSENAL_CACHE_DIR=/tmp/arsenal-v2-voices/cache \
        "$proof" "$root" "$helpers" "$model" "$dtype" "$mode" > "$evidence/$tag.log" 2>&1; then
        cp "$support/catalog-cycle-proof.json" "$evidence/$tag.json"
        printf 'voices: %s service cycle passed\n' "$tag"
    else
        code=$?
        printf 'voices: %s service cycle failed (%s); see %s.log\n' "$tag" "$code" "$tag"
        if [[ "$code" == 75 ]]; then exit 75; fi
        failed=1
    fi
done
exit "$failed"
