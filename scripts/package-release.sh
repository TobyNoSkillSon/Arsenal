#!/usr/bin/env bash
# Local release staging only: builds, zips and checksums. Installs, uploads and publishes nothing.
# Usage: package-release.sh [VERSION]   (default: Resources/Info.plist)
# Output: $ARSENAL_RELEASE_OUTPUT_DIR or .build/releases/VERSION/
#   Arsenal-VERSION-arm64.zip            the app (the name the updater, install.sh and the cask download)
#   Arsenal-VERSION-arm64-symbols.zip    unstripped binaries + UUIDS.txt, to symbolicate crash reports
#   SHA256SUMS                           `shasum -a 256` lines for both zips
# Signing: scripts/build.sh signs with ARSENAL_SIGN_IDENTITY (CI: the release-signing environment; locally:
# with-release-keychain.sh). A zip not signed by "Arsenal Release Signing" with the pinned requirement gets
# LOCAL-ONLY-NOT-FOR-UPLOAD.txt beside it (release-identity.sh mark-local) and is never uploaded.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$PROJECT/Resources/Info.plist"
VERSION="${1:-$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$PLIST")}"
BUILD="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$PLIST")"
REPO="$(/usr/libexec/PlistBuddy -c 'Print :ArsenalUpdateRepo' "$PLIST" 2>/dev/null || true)"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Invalid version: $VERSION" >&2; exit 2; }
[[ "$VERSION" == "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$PLIST")" ]] || {
  echo "Version $VERSION differs from Resources/Info.plist; bump the plist, not the command line" >&2; exit 2; }
# Builds stay above the last build numbered by commit count (1433), so LaunchServices and the installers never see a step back.
[[ "$BUILD" =~ ^[0-9]+$ && "$BUILD" -gt 1433 ]] || { echo "CFBundleVersion must be a number above 1433 (is $BUILD)" >&2; exit 2; }
[[ "$REPO" == "${ARSENAL_EXPECTED_UPDATE_REPO:-TobyNoSkillSon/Arsenal}" ]] || {
  echo "Resources/Info.plist ArsenalUpdateRepo is '${REPO}', expected ${ARSENAL_EXPECTED_UPDATE_REPO:-TobyNoSkillSon/Arsenal}" >&2; exit 2; }
OUT="${ARSENAL_RELEASE_OUTPUT_DIR:-$PROJECT/.build/releases/$VERSION}"
[[ ! -e "$OUT" ]] || { echo "Output already exists; preserving it: $OUT" >&2; exit 1; }
[[ "$(uname -m)" == arm64 ]] || { echo 'Requires an Apple Silicon Mac' >&2; exit 1; }

# build.sh writes dist/Arsenal.app, dist/Arsenal.symbols and dist/BUILD-INFO.json, or fails and leaves dist/ unchanged.
ARSENAL_BUILD_VERSION="$VERSION" "$PROJECT/scripts/build.sh" >/dev/null
DIST="$PROJECT/dist"
APP="$DIST/Arsenal.app"
/usr/bin/python3 - "$DIST/BUILD-INFO.json" "$VERSION" "$BUILD" <<'PY'
import json, sys
info, version, build = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
problems = []
if info.get("version") != version or str(info.get("build")) != build: problems.append(f"built {info.get('version')} ({info.get('build')}), expected {version} ({build})")
if info.get("dirty"): problems.append("built from a dirty tree")
if info.get("gaps"): problems.append("gaps: " + "; ".join(info["gaps"]))
if problems: sys.exit("package-release: refusing: " + " | ".join(problems))
PY
codesign --verify --deep --strict "$APP"
for f in MacOS/Arsenal Helpers/ArsenalUpdateHelper Helpers/arsenal-mcp Helpers/arsenal-worker-native; do
  [[ -x "$APP/Contents/$f" ]] || { echo "Built app lacks Contents/$f" >&2; exit 1; }
done
[[ -s "$APP/Contents/Helpers/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ]] || { echo 'Built app lacks the Metal library' >&2; exit 1; }
[[ "$(/usr/libexec/PlistBuddy -c 'Print :ArsenalUpdateRepo' "$APP/Contents/Info.plist")" == "$REPO" ]] || { echo 'Built app lacks ArsenalUpdateRepo' >&2; exit 1; }

mkdir -p "$PROJECT/.build"
STAGE="$(mktemp -d "$PROJECT/.build/.package-stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
ZIP="Arsenal-$VERSION-arm64.zip"
SYMBOLS_ZIP="Arsenal-$VERSION-arm64-symbols.zip"
"$PROJECT/scripts/release-zip.sh" "$APP" "$STAGE/$ZIP"
LISTING="$(zipinfo -1 "$STAGE/$ZIP")"
for f in MacOS/Arsenal Helpers/ArsenalUpdateHelper Helpers/arsenal-mcp Helpers/arsenal-worker-native \
         Helpers/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib Info.plist; do
  grep -qx "Arsenal.app/Contents/$f" <<<"$LISTING" || { echo "Archive is missing Arsenal.app/Contents/$f" >&2; exit 1; }
done
# The add-a-model kit's two scripts ship on purpose (Contents/Resources/kit, docs/ADD-A-MODEL.md); any other .py does not.
if grep -v -E '/Contents/Resources/kit/(new-model|folder_check)\.py$' <<<"$LISTING" | grep -E '\.py$|/\.build/|/lab/|\.dSYM/|__pycache__' >&2; then
  echo 'Archive holds development files' >&2; exit 1
fi
mkdir -p "$STAGE/public"
ditto "$DIST/Arsenal.symbols" "$STAGE/public/Arsenal.symbols"
ditto -c -k --norsrc --noextattr --noqtn --noacl --keepParent "$STAGE/public/Arsenal.symbols" "$STAGE/$SYMBOLS_ZIP"
(cd "$STAGE" && shasum -a 256 "$ZIP" "$SYMBOLS_ZIP") > "$STAGE/SHA256SUMS"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
mv "$STAGE/$ZIP" "$STAGE/$SYMBOLS_ZIP" "$STAGE/SHA256SUMS" "$OUT/"
CLASS="$("$PROJECT/scripts/release-identity.sh" mark-local "$OUT")"
if [[ "$CLASS" == release ]]; then
  "$PROJECT/scripts/release-identity.sh" for-upload "$OUT" >/dev/null || {
    echo "LOCAL ONLY: signed by Arsenal Release Signing but not with the pinned requirement" > "$OUT/LOCAL-ONLY-NOT-FOR-UPLOAD.txt"; CLASS=unpinned; }
fi
echo "Packaged locally: $OUT/$ZIP, $OUT/$SYMBOLS_ZIP and $OUT/SHA256SUMS (version $VERSION build $BUILD, signing: $CLASS; not published or installed)"
