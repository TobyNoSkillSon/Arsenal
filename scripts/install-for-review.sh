#!/bin/bash
# Install the reviewed candidate: dist/Arsenal.app -> ~/Applications/Arsenal.app.
# REFUSES to run unless ARSENAL_REVIEWED=1 — nobody runs this before v-family's review has approved the candidate.
#   ARSENAL_REVIEWED=1 scripts/install-for-review.sh [path/to/Arsenal.app]
# Overrides (for rehearsals): ARSENAL_DESTINATION_APP, ARSENAL_SUPPORT_DIR.
# Verifies the candidate first (signature, per-file checksums), refuses while Arsenal is running, swaps with rollback,
# and snapshots only catalog/model instructions in the support dir (never weights or user settings).
set -euo pipefail
source "$(dirname "$0")/pkg-lib.sh"
[[ "${ARSENAL_REVIEWED:-}" == 1 ]] || die "refusing to install: this candidate has not been reviewed. After v-family approves it, run: ARSENAL_REVIEWED=1 $0"

SRC="${1:-$PKG_ROOT/dist/Arsenal.app}"
DEST="${ARSENAL_DESTINATION_APP:-$HOME/Applications/Arsenal.app}"
[[ -d "$SRC/Contents" ]] || die "no candidate at $SRC; run scripts/build.sh"

codesign --verify --deep --strict "$SRC" || die "candidate signature does not verify; nothing installed"
# Only the pinned release identity can take updates from the public repository (docs/UPDATES.md, Identity).
# ARSENAL_REVIEW_ALLOW_UNPINNED=1 is for the install/rollback tests, whose fixture apps are signed ad hoc.
if [[ "${ARSENAL_REVIEW_ALLOW_UNPINNED:-0}" != 1 ]]; then
  "$PKG_ROOT/scripts/release-identity.sh" pinned "$SRC" \
    || die "candidate is not signed with Arsenal Release Signing (pinned); such an install cannot take updates. Build through with-release-keychain.sh; nothing installed"
fi
if [[ -d "$DEST/Contents" ]] && [[ "$(codesign -d -r- "$DEST" 2>&1 | sed -n 's/^designated => //p')" != "$(codesign -d -r- "$SRC" 2>&1 | sed -n 's/^designated => //p')" ]]; then
  echo "note: the installed Arsenal is signed with another certificate; macOS will ask once more for Microphone and folder access"
fi
if [[ -f "$(dirname "$SRC")/Arsenal.app.sha256" && "$(basename "$SRC")" == Arsenal.app ]]; then
  (cd "$(dirname "$SRC")" && shasum -a 256 -c Arsenal.app.sha256 >/dev/null) || die "candidate files differ from Arsenal.app.sha256; nothing installed"
fi

# A running copy keeps its models loaded; replacing files under it would break it. The user quits it, we do not.
if pgrep -f "$DEST/Contents/MacOS/Arsenal" >/dev/null; then die "Arsenal is running from $DEST; quit it (menu bar → Quit) and run again; nothing installed"; fi
PROFILE="$(profile_dir)"

mkdir -p "$(dirname "$DEST")"
NEW="$DEST.new.$$"; OLD="$DEST.previous"
[[ -e "$OLD" || ! -e "$DEST.previous.rollback-state" ]] || die "snapshot pointer has no previous app; repair that pair before installing"
[[ ! -e "$NEW" ]] || die "staged app already exists at $NEW"
ditto "$SRC" "$NEW"
codesign --verify --deep --strict "$NEW" || { rm -rf "$NEW"; die "copy does not verify; nothing installed"; }
# One rollback: the app being replaced becomes the rollback, and the older rollback is deleted for good (rm -rf, never
# the Trash).
if [[ -e "$OLD" ]]; then
  /usr/bin/python3 -B "$PKG_ROOT/scripts/launch_services.py" unregister "$OLD"
  FREED=$(du -sh "$OLD" | cut -f1)
  rm -rf "$OLD"
  rm -f "$DEST.previous.rollback-state"
  echo "older rollback app deleted: $OLD ($FREED freed)"
fi
if [[ -e "$DEST" ]]; then
  VERSION=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$DEST/Contents/Info.plist")
  SNAPSHOT=$(ARSENAL_SUPPORT_DIR="$PROFILE" "$PKG_ROOT/scripts/rollback.sh" --snapshot "$VERSION") || { rm -rf "$NEW"; die "could not snapshot previous instructions; nothing installed"; }
  printf '%s\n' "$SNAPSHOT" > "$DEST.previous.rollback-state"
  mv "$DEST" "$OLD"
fi
mv "$NEW" "$DEST"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$DEST"
[[ ! -e "$OLD" ]] || /usr/bin/python3 -B "$PKG_ROOT/scripts/launch_services.py" unregister "$OLD"
[[ ! -e "$OLD" ]] || echo "previous app kept at $OLD with its instruction snapshot; rollback: scripts/rollback.sh"
echo "installed $DEST; configure agents with the bundled skill or docs/MCP.md"
