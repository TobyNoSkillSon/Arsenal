#!/bin/bash
# Pure CPU checks, no package/app build or model loaded. Caller owns scratch; remove it after evidence is saved.
set -euo pipefail
root=$(cd -- "$(dirname "$0")/../.." && pwd)
scratch=$1
mkdir -p "$scratch"
DEVELOPER_DIR=/Library/Developer/CommandLineTools /Library/Developer/CommandLineTools/usr/bin/swiftc -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk -target arm64-apple-macos26.0 -O -o "$scratch/regression" \
  "$root/Sources/D1Engine/"{PyJSON,Tokenizer,Prompt,Pictures}.swift "$root/Tests/Regression/main.swift"
"$scratch/regression" "$root/Tests/Regression/fixtures" "$scratch"
