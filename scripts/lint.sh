#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# First-party Swift layout + rules. Run through scripts/mlock work.
# scripts/lint.sh [--fix] [path…] checks all first-party Swift, or only matching files/directories.
set -euo pipefail
cd "$(dirname "$0")/.."
SWIFT_FORMAT=/Library/Developer/CommandLineTools/usr/bin/swift-format
[[ -x "$SWIFT_FORMAT" ]] || SWIFT_FORMAT="$(xcrun --find swift-format)"
command -v swiftlint >/dev/null || { echo 'lint: install SwiftLint 0.65.1 (brew install swiftlint)' >&2; exit 1; }
fix=false
if [[ "${1:-}" == --fix ]]; then fix=true; shift; fi
files=()
while IFS= read -r file; do files+=("$file"); done < <(/usr/bin/python3 - "$@" <<'PY'
import pathlib, subprocess, sys
# Engine ports retain upstream layout; see .swiftlint.yml for provenance/exclusion reasons.
excluded = (
    'Packages/swift-sdk/', # Vendored official MCP SDK; provenance in PORTED-FROM.md.
    'Workers/Voices/Sources/MLXAudioCore/', 'Workers/Voices/Sources/MLXAudioTTS/', 'Workers/Voices/Sources/MLXAudioCodecs/',
    'Workers/Voices/Sources/FishS1Codec/',
    'Workers/Voices/Sources/VireoCore/', 'Workers/Voices/Sources/VireoEngine/', 'Workers/Voices/Sources/SmallMGEMM/',
    'Workers/Decisions/Sources/VerdictCore/', 'Workers/Decisions/Sources/VerdictEngine/',
    'Workers/Decisions/Sources/VerdictSmallMGEMM/', 'Workers/Decisions/Sources/DecisionsProtocol/',
    'models/birefnet/runners/swift/Sources/MLXBiRefNet/',
    'Workers/Sound/Sources/StableAudio3/', 'Workers/Sound/Sources/SoundEngine/',
    'Workers/Audio/Sources/VellaSTT/',
    'Workers/Kit/Sources/', # Kit model runners, byte-identical with their folder runners (Workers/Kit/PORTED-FROM.md).
)
paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '*.swift'], text=True).splitlines()
for path in sorted(set(paths)):
    if not (path in ('Package.swift', 'scripts/icon.swift') or path.startswith(('Sources/', 'Tests/', 'Packages/', 'Workers/', 'Tests/Fixtures/Templates/runners/swift/', 'models/d1-3b/runners/swift/', 'models/elephant-1/runners/swift/', 'models/flux2-klein-4b/runners/swift/', 'models/birefnet/runners/swift/'))): continue
    if path.startswith(excluded) or not pathlib.Path(path).is_file(): continue
    if sys.argv[1:] and not any(path == p.rstrip('/') or path.startswith(p.rstrip('/') + '/') for p in sys.argv[1:]): continue
    print(path)
PY
)
(( ${#files[@]} > 0 )) || { echo 'lint: no first-party Swift files matched' >&2; exit 1; }
export SCRIPT_INPUT_FILE_COUNT="${#files[@]}"
for ((i=0; i<${#files[@]}; i++)); do export "SCRIPT_INPUT_FILE_$i=${files[$i]}"; done
export XCODE_DEFAULT_TOOLCHAIN_OVERRIDE="${XCODE_DEFAULT_TOOLCHAIN_OVERRIDE:-/Library/Developer/CommandLineTools}"
if $fix; then
  swiftlint lint --fix --quiet --use-script-input-files
  "$SWIFT_FORMAT" format --configuration .swift-format --in-place --parallel "${files[@]}"
fi
# Cache only a complete successful pass, keyed by every selected input, rules and tool binaries.
lint_key=$(/usr/bin/python3 scripts/build-cache.py key --context "$(stat -f '%m:%z' "$SWIFT_FORMAT" "$(command -v swiftlint)")" \
  .swift-format .swiftlint.yml scripts/lint.sh "${files[@]}")
lint_receipt="$PWD/.build/lint/$lint_key.json"
if ! $fix && /usr/bin/python3 scripts/build-cache.py check --key "$lint_key" --receipt "$lint_receipt"; then
  echo "lint: ${#files[@]} unchanged files clean"; exit 0
fi
"$SWIFT_FORMAT" lint --strict --configuration .swift-format --parallel "${files[@]}"
swiftlint lint --strict --quiet --use-script-input-files
/usr/bin/python3 scripts/build-cache.py record --key "$lint_key" --receipt "$lint_receipt"
echo "lint: ${#files[@]} files clean"
