#!/usr/bin/env bash
# Zip a signed app for release and check the archive (package-release.sh uses it).
#   release-zip.sh APP ZIP      write ZIP from APP, then check it
#   release-zip.sh --verify ZIP check an existing ZIP
# The archive carries no extended attributes: ditto would store them (com.apple.provenance, quarantine) as
# AppleDouble ._* entries, which /usr/bin/unzip extracts into the bundle and breaks its signature.
# Check: no ._* or __MACOSX entries, one .app at the top, within the updater's limits (UpdateNetwork: 256 MiB zip,
# 512 MiB unpacked, 50 000 entries), and the app extracted by /usr/bin/unzip passes codesign --verify --deep --strict.
set -euo pipefail
usage() { echo "usage: $0 APP ZIP | $0 --verify ZIP" >&2; exit 2; }
if [[ "${1:-}" == --verify ]]; then
  [[ $# -eq 2 ]] || usage
  ZIP="$2"
else
  [[ $# -eq 2 && -d "$1" ]] || usage
  APP="$1"; ZIP="$2"
  [[ ! -e "$ZIP" ]] || { echo "Archive already exists: $ZIP" >&2; exit 1; }
  # Every user of the Mac must be able to run the app, whatever umask the build ran under (signing keeps no modes).
  chmod -R u+rwX,go+rX,go-w "$APP"
  ditto -c -k --norsrc --noextattr --noqtn --noacl --keepParent "$APP" "$ZIP"
fi
LISTING="$(zipinfo -1 "$ZIP")"
if grep -E '(^|/)\._|^__MACOSX/' <<<"$LISTING" >&2; then
  echo "Archive holds AppleDouble (._*) or __MACOSX entries: $ZIP" >&2; exit 1
fi
TOP="$(awk -F/ 'NR==1 {print $1}' <<<"$LISTING")"
[[ "$TOP" == *.app ]] && ! grep -qvE "^$TOP(/|$)" <<<"$LISTING" || { echo "Archive must hold one .app at its top level: $ZIP" >&2; exit 1; }
# Owner-only or read-only entries: other users could not run the app, and xattr -dr fails on read-only files.
if zipinfo "$ZIP" | awk '$1 ~ /^[-dl]/ && $1 !~ /^l/ && (substr($1,2,2) != "rw" || substr($1,5,1) != "r" || substr($1,8,1) != "r") {bad=1} END {exit !bad}'; then
  echo "Archive holds entries that are not owner-writable and world-readable: $ZIP" >&2; exit 1
fi
BYTES="$(stat -f %z "$ZIP")"
(( BYTES <= 256 * 1024 * 1024 )) || { echo "Archive is $BYTES bytes; the updater accepts at most 256 MiB: $ZIP" >&2; exit 1; }
EXPANDED="$(zipinfo -t "$ZIP" | awk '{print $3}')"
(( EXPANDED <= 512 * 1024 * 1024 )) || { echo "Archive unpacks to $EXPANDED bytes; the updater accepts at most 512 MiB: $ZIP" >&2; exit 1; }
(( $(wc -l <<<"$LISTING") <= 50000 )) || { echo "Archive has more than 50 000 entries: $ZIP" >&2; exit 1; }
CHECK="$(mktemp -d "${TMPDIR:-/tmp}/release-zip.XXXXXX")"
trap 'rm -rf "$CHECK"' EXIT
/usr/bin/unzip -q "$ZIP" -d "$CHECK"
codesign --verify --deep --strict "$CHECK/$TOP" || { echo "App extracted with /usr/bin/unzip fails verification: $ZIP" >&2; exit 1; }
echo "Checked $ZIP: no AppleDouble entries; within the updater's limits; $TOP from /usr/bin/unzip verifies"
