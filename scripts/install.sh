#!/bin/bash
# Arsenal installer: curl -fsSL https://raw.githubusercontent.com/TobyNoSkillSon/Arsenal/main/scripts/install.sh | bash
# Downloads the prebuilt release with curl, verifies its SHA-256, contents, version and code signature (the pinned
# "Arsenal Release Signing" certificate) before touching anything, installs it in ~/Applications, links the
# `arsenal-mcp` command for MCP clients and starts Arsenal. Needs no Xcode or developer account.
#   bash -s -- --dry-run            verify without installing
#   bash -s -- --migrate-signing    replace an Arsenal signed with another certificate (macOS asks for its
#                                   permissions once more); the previous app is kept beside the new one
#   bash -s -- --allow-downgrade    install an older version than the installed one
# Environment: ARSENAL_VERSION (default: the latest published release), ARSENAL_DESTINATION_APP, ARSENAL_BIN_DIR,
# ARSENAL_RELEASE_BASE_URL (file:// serves a release packaged by scripts/package-release.sh, for testing).
set -euo pipefail

REPO=TobyNoSkillSon/Arsenal
PIN=011c682945c225dbdec95aeaa187becbb0207282   # SHA-1 of "Arsenal Release Signing" (public; same in every release)
BUNDLE_ID=dev.arsenal.app

# Everything runs from main, so a download cut short executes nothing.
main() {
  local DRY_RUN=0 option
  MIGRATE=0 ALLOW_DOWNGRADE=0
  for option in "$@"; do
    case "$option" in
      --dry-run) DRY_RUN=1 ;;
      --migrate-signing) MIGRATE=1 ;;
      --allow-downgrade) ALLOW_DOWNGRADE=1 ;;
      *) fail 'Usage: install.sh [--dry-run] [--migrate-signing] [--allow-downgrade]' 2 ;;
    esac
  done
  [[ "$(uname -s)" == Darwin && "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == 1 ]] || fail 'Arsenal requires an Apple Silicon Mac'
  local OS; OS="$(sw_vers -productVersion)"
  [[ "${OS%%.*}" -ge 26 ]] || fail "Arsenal requires macOS 26 or newer ($OS)"
  TEMP="$(mktemp -d "${TMPDIR:-/tmp}/arsenal-release.XXXXXX")"
  trap 'rm -rf "$TEMP"' EXIT

  VERSION="${ARSENAL_VERSION:-}"
  if [[ -z "$VERSION" ]]; then
    fetch "https://api.github.com/repos/$REPO/releases" latest latest.json
    VERSION="$(sed -n 's/^ *"tag_name": *"v\([0-9][0-9.]*\)".*/\1/p' "$TEMP/latest.json" | sed -n 1p)"
  fi
  [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail 'ARSENAL_VERSION must be a release version, e.g. 1.0.0' 2

  local BASE="${ARSENAL_RELEASE_BASE_URL:-https://github.com/$REPO/releases/download/v$VERSION}"
  [[ "$BASE" == https://* || "$BASE" == file://* ]] || fail 'Release base URL must use HTTPS'
  local ZIP="Arsenal-$VERSION-arm64.zip"
  echo "Downloading Arsenal ${VERSION}…"
  fetch "$BASE" SHA256SUMS SHA256SUMS
  fetch "$BASE" "$ZIP" "$ZIP"
  local EXPECTED ACTUAL
  EXPECTED="$(awk -v name="$ZIP" '$2 == name || $2 == "*" name { print $1 }' "$TEMP/SHA256SUMS")"
  [[ "$EXPECTED" =~ ^[a-fA-F0-9]{64}$ ]] || fail "Missing or ambiguous SHA-256 for $ZIP; nothing installed."
  ACTUAL="$(shasum -a 256 "$TEMP/$ZIP" | awk '{print $1}')"
  [[ "$(tr '[:upper:]' '[:lower:]' <<<"$ACTUAL")" == "$(tr '[:upper:]' '[:lower:]' <<<"$EXPECTED")" ]] || fail "SHA-256 mismatch for $ZIP; nothing installed."
  echo "verified SHA-256 $ACTUAL  $ZIP"
  # The hash detects corruption; a checksum fetched from the same release is not a signature. The pinned certificate
  # check below is what ties the app to Arsenal's release key.
  zipinfo -1 "$TEMP/$ZIP" | awk '
    BEGIN { good = 1 } { n++; if (n > 50000 || $0 !~ /^Arsenal\.app\// || $0 ~ /(^|\/)\.\.(\/|$)/ || $0 ~ /\\/ || $0 ~ /(^|\/)\._/ || $0 ~ /__MACOSX/) good = 0 }
    END { exit !(good && n > 0) }' || fail 'Unsafe or unexpected archive entries; nothing installed.'
  mkdir "$TEMP/unpacked"
  ditto -x -k "$TEMP/$ZIP" "$TEMP/unpacked"
  local APP="$TEMP/unpacked/Arsenal.app" f
  for f in MacOS/Arsenal Helpers/ArsenalUpdateHelper Helpers/arsenal-mcp Helpers/arsenal-worker-native; do
    [[ -x "$APP/Contents/$f" ]] || fail "Release archive lacks Contents/$f; nothing installed."
  done
  [[ -s "$APP/Contents/Helpers/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ]] || fail 'Release archive lacks the Metal library; nothing installed.'
  [[ "$(plist "$APP" CFBundleIdentifier)" == "$BUNDLE_ID" ]] || fail 'Release archive does not hold Arsenal; nothing installed.'
  [[ "$(plist "$APP" CFBundleShortVersionString)" == "$VERSION" ]] || fail 'App version does not match the requested release; nothing installed.'
  codesign --verify --deep --strict "$APP" || fail 'App signature does not verify; nothing installed.'
  local NEW_REQ; NEW_REQ="$(requirement "$APP")"
  [[ "$NEW_REQ" == "identifier \"$BUNDLE_ID\" and certificate leaf = H\"$PIN\"" ]] || fail 'App is not signed with Arsenal Release Signing; nothing installed.'
  echo "verified Arsenal.app $VERSION (build $(plist "$APP" CFBundleVersion)): helpers, Metal library, Arsenal Release Signing"

  local DEST="${ARSENAL_DESTINATION_APP:-$HOME/Applications/Arsenal.app}"
  check_existing "$APP" "$DEST" "$NEW_REQ"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "dry run: would install to $DEST; nothing installed"
    return 0
  fi
  install_app "$APP" "$DEST"
}

fail() { echo "$1" >&2; exit "${2:-1}"; }
plist() { /usr/libexec/PlistBuddy -c "Print :$2" "$1/Contents/Info.plist"; }
requirement() { codesign -d -r- "$1" 2>&1 | sed -n 's/^designated => //p'; }

fetch() {
  local http result=0
  http="$(curl --fail --location --silent --show-error --proto '=https,file' --proto-redir '=https' --tlsv1.2 \
    --connect-timeout 20 --max-time 1800 --write-out '%{http_code}' "$1/$2" -o "$TEMP/$3" 2>"$TEMP/curl-error")" || result=$?
  [[ $result == 0 ]] && return 0
  [[ ! -s "$TEMP/curl-error" ]] || head -1 "$TEMP/curl-error" >&2
  if [[ "$http" == 404 ]]; then fail "Arsenal ${VERSION:-(latest)} is not available at the release URL; nothing was installed."
  else fail 'Arsenal could not download the release. Check your connection and try again; the installed app is unchanged.'; fi
}

# Refuses a running Arsenal, an older version (unless --allow-downgrade) and an app signed by another certificate
# (unless --migrate-signing: macOS ties Microphone and folder permissions to the certificate and asks again once).
check_existing() {
  local APP="$1" DEST="$2" NEW_REQ="$3" OLD NEW OLD_BUILD NEW_BUILD OLD_REQ
  if pgrep -f "$DEST/Contents/MacOS/Arsenal" >/dev/null; then
    fail "Arsenal is running from $DEST. Quit it (menu bar → Quit Arsenal) and run the installer again; nothing installed."
  fi
  [[ -e "$DEST" ]] || return 0
  [[ "$(plist "$DEST" CFBundleIdentifier 2>/dev/null)" == "$BUNDLE_ID" ]] || fail "$DEST is not Arsenal; nothing installed."
  OLD="$(plist "$DEST" CFBundleShortVersionString)"; NEW="$(plist "$APP" CFBundleShortVersionString)"
  OLD_BUILD="$(plist "$DEST" CFBundleVersion)"; NEW_BUILD="$(plist "$APP" CFBundleVersion)"
  [[ "$OLD" =~ ^[0-9]{1,9}\.[0-9]{1,9}\.[0-9]{1,9}$ && "$NEW" =~ ^[0-9]{1,9}\.[0-9]{1,9}\.[0-9]{1,9}$ &&
     "$OLD_BUILD" =~ ^[0-9]{1,9}$ && "$NEW_BUILD" =~ ^[0-9]{1,9}$ ]] || fail "Cannot read the installed Arsenal's version/build; nothing installed."
  if [[ "$ALLOW_DOWNGRADE" != 1 ]] && ! awk -v old="$OLD" -v new="$NEW" -v ob="$OLD_BUILD" -v nb="$NEW_BUILD" 'BEGIN {
      split(old,a,"."); split(new,b,"."); for (i=1;i<=3;i++) { if (b[i]+0 < a[i]+0) exit 1; if (b[i]+0 > a[i]+0) exit 0 }
      exit (nb+0 < ob+0) }'; then
    fail "Refusing to replace Arsenal $OLD (build $OLD_BUILD) with older Arsenal $NEW (build $NEW_BUILD); nothing installed. To downgrade on purpose, repeat with --allow-downgrade."
  fi
  OLD_REQ="$(requirement "$DEST")"
  if [[ "$OLD_REQ" != "$NEW_REQ" && "$MIGRATE" != 1 ]]; then
    fail "The installed Arsenal is signed with another certificate. Installing this release moves it to Arsenal Release Signing, and macOS will ask once more for Microphone and folder access. Repeat with --migrate-signing to go ahead; nothing installed."
  fi
}

install_app() {
  local APP="$1" DEST="$2" NEW PREVIOUS="" BIN="${ARSENAL_BIN_DIR:-$HOME/.local/bin}"
  mkdir -p "$(dirname "$DEST")"
  NEW="$DEST.new.$$"
  [[ ! -e "$NEW" ]] || fail "Staging path exists: $NEW; nothing installed."
  # The release zip carries no extended attributes and curl sets no quarantine; clear it anyway, as the cask does.
  xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
  ditto "$APP" "$NEW"
  codesign --verify --deep --strict "$NEW" || { rm -rf "$NEW"; fail 'Copied app does not verify; nothing installed.'; }
  if [[ -e "$DEST" ]]; then
    PREVIOUS="$DEST.previous"
    rm -rf "$PREVIOUS"            # one rollback copy: the app being replaced
    mv "$DEST" "$PREVIOUS"
  fi
  mv "$NEW" "$DEST"
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$DEST" || true
  # `arsenal-mcp` for stdio MCP clients: a link into the app, so updates carry it along.
  mkdir -p "$BIN"
  ln -sfn "$DEST/Contents/Helpers/arsenal-mcp" "$BIN/arsenal-mcp"
  case ":$PATH:" in *":$BIN:"*) echo "command: $BIN/arsenal-mcp" ;; *) echo "command: $BIN/arsenal-mcp (add $BIN to PATH)" ;; esac
  echo "installed $DEST; starting…"
  open "$DEST" || true
  for _ in $(seq 1 60); do pgrep -f "$DEST/Contents/MacOS/Arsenal" >/dev/null && break; sleep 1; done
  if ! pgrep -f "$DEST/Contents/MacOS/Arsenal" >/dev/null; then
    if [[ -n "$PREVIOUS" ]]; then
      rm -rf "$DEST" && mv "$PREVIOUS" "$DEST" && open "$DEST" || true
      fail "Arsenal $VERSION did not start; the previous app was restored."
    fi
    fail "Arsenal did not start; it is installed at $DEST. Open it from Applications to see why."
  fi
  if [[ -n "$PREVIOUS" ]]; then
    if [[ "$MIGRATE" == 1 ]]; then echo "Previous app kept at $PREVIOUS; to roll back, quit Arsenal and move it to $DEST."
    else rm -rf "$PREVIOUS"; fi
  fi
  echo "Arsenal $(plist "$DEST" CFBundleShortVersionString) is running (menu bar)."
}

main "$@"
