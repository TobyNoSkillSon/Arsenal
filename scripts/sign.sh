#!/bin/bash
# Sign an assembled Arsenal.app inside-out with "Arsenal Release Signing", then verify.
# Usage: sign.sh <app-path>
# The identity is the pinned certificate (pkg-lib.sh RELEASE_SIGNING_SHA1), self-signed like Vella's "Vella Release
# Signing": no Apple account, not notarised. Every signature gets the explicit designated requirement
#   identifier "<id>" and certificate leaf = H"<sha1>"
# (codesign's default for this certificate would say `certificate root`; the explicit form is Vella's and is what
# release-identity.sh and release.yml compare). macOS ties Microphone and folder permissions to that requirement, and
# the updater accepts only a download that satisfies the installed app's requirement, so it must never change.
# The key is never in the login keychain: run through with-release-keychain.sh (local) or ci-keychain.sh (CI); both set
# ARSENAL_SIGN_KEYCHAIN, which is passed to codesign with --keychain.
# Hardened runtime is on. The app carries one entitlement, audio input, for live transcription from the microphone
# (Info.plist NSMicrophoneUsageDescription); helpers carry none (no JIT: MLX ships a prebuilt metallib; no
# Accessibility, no unsigned memory). spctl rejects the result by design (self-signed, not notarised).
set -euo pipefail
source "$(dirname "$0")/pkg-lib.sh"
APP="${1:?usage: sign.sh <app-path>}"
IDENTITY="$(resolve_identity)"
KEYCHAIN=()
[[ -z "${ARSENAL_SIGN_KEYCHAIN:-}" ]] || KEYCHAIN=(--keychain "$ARSENAL_SIGN_KEYCHAIN")
security find-identity -p codesigning ${ARSENAL_SIGN_KEYCHAIN:+"$ARSENAL_SIGN_KEYCHAIN"} | grep -Fq -- "$IDENTITY" \
  || die "signing identity $IDENTITY ($SIGN_NAME) is not available; run the build through with-release-keychain.sh"
PIN="$(tr '[:upper:]' '[:lower:]' <<<"$IDENTITY")"

# --timestamp=none: a self-signed identity has no timestamp authority.
sign() {
  local id="$1"; shift
  codesign ${KEYCHAIN[@]+"${KEYCHAIN[@]}"} --force --options runtime --timestamp=none --sign "$IDENTITY" --identifier "$id" \
    "-r=designated => identifier \"$id\" and certificate leaf = H\"$PIN\"" "$@"
}

# 1. Bundles that are signable code containers (have an Info.plist); plain resource folders are sealed by the app.
while IFS= read -r bundle; do
  plist="$bundle/Contents/Info.plist"; [[ -f "$plist" ]] || plist="$bundle/Info.plist"
  [[ -f "$plist" ]] || continue
  sign "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$plist")" "$bundle"
done < <(find "$APP/Contents/Helpers" -maxdepth 1 -name '*.bundle' -type d)
# 2. Every helper executable, each with a stable identifier (the designated requirement pins identifier + certificate).
#    Family links (arsenal-worker-<family> -> arsenal-worker-native) are sealed by the app as links, not signed again.
for file in "$APP"/Contents/Helpers/*; do
  [[ -f "$file" && -x "$file" && ! -L "$file" ]] || continue
  sign "dev.arsenal.helper.$(basename "$file")" "$file"
done
# 3. The app itself (main executable + seal over everything above), with its one hardened-runtime entitlement.
entitlements="$(mktemp "${TMPDIR:-/tmp}/arsenal-entitlements.XXXXXX")"
trap 'rm -f "$entitlements"' EXIT
cat > "$entitlements" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>com.apple.security.device.audio-input</key><true/>
</dict></plist>
PLIST
plutil -lint "$entitlements" >/dev/null
sign "$BUNDLE_ID" --entitlements "$entitlements" "$APP"

codesign --verify --deep --strict "$APP"
signed_entitlements="$(codesign -d --entitlements - --xml "$APP" 2>/dev/null || true)"
[[ "$signed_entitlements" == *com.apple.security.device.audio-input* ]] || die "the app lacks its audio-input entitlement"
requirement="$(codesign -d -r- "$APP" 2>&1 | sed -n 's/^designated => //p')"
[[ "$requirement" == "identifier \"$BUNDLE_ID\" and certificate leaf = H\"$PIN\"" ]] || die "unexpected designated requirement: $requirement"
echo "signed: $APP ($(codesign -dvv "$APP" 2>&1 | sed -n 's/^Authority=//p' | head -1), pinned $IDENTITY)"
