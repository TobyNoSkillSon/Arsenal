#!/usr/bin/env bash
# Which identity signed an Arsenal build, and whether it may be uploaded. An app installed from a build signed with any
# other identity (an Apple Development certificate, an older local certificate, ad-hoc) rejects every later update
# signed with "Arsenal Release Signing", and a development certificate exposes its owner's email in `codesign -dvv`.
# Such a build is for local checks only and must never be uploaded.
#
#   release-identity.sh classify APP|ZIP       print release | development | adhoc | unsigned | other | mixed
#   release-identity.sh mark-local DIR         write LOCAL-ONLY-NOT-FOR-UPLOAD.txt into a package directory whose zip is
#                                              not signed by Arsenal Release Signing (package-release.sh calls this)
#   release-identity.sh for-upload PATH        exit 0 only for a release-signed zip or app, or a directory whose
#                                              Arsenal-*-arm64.zip is release-signed with the pinned designated
#                                              requirement and which carries no LOCAL-ONLY marker; refuses everything else
#   release-identity.sh pinned APP             exit 0 when the app and all its code carry the pinned requirement
#   release-identity.sh text-class FILE        classify one saved `codesign -dvv` output (tests)
# Names of non-release identities are never printed: a development identity's name is the developer's email.
set -euo pipefail
PIN_SHA1=011C682945C225DBDEC95AEAA187BECBB0207282   # "Arsenal Release Signing", same pin as release.yml and pkg-lib.sh
AUTHORITY="Arsenal Release Signing"
APP_ID=dev.arsenal.app
MARKER=LOCAL-ONLY-NOT-FOR-UPLOAD.txt

die() { echo "release-identity: $*" >&2; exit 1; }

# One `codesign -dvv` output on stdin -> release | development | adhoc | unsigned | other
classify_text() {
  local text leaf
  text="$(cat)"
  if grep -q '^Signature=adhoc' <<<"$text"; then echo adhoc; return; fi
  leaf="$(sed -n 's/^Authority=//p' <<<"$text" | sed -n '1p')"
  case "$leaf" in
    "") echo unsigned ;;
    "$AUTHORITY") echo release ;;
    "Apple Development:"*|"Mac Developer:"*|"iPhone Developer:"*|"Apple Distribution:"*|"3rd Party Mac Developer"*|"Developer ID"*) echo development ;;
    *) echo other ;;
  esac
}

# Every signed item of the bundle as "relative-path<TAB>identifier" ("." is the app itself): the app, each helper executable (scripts/sign.sh
# signs them as dev.arsenal.helper.<name>) and each code bundle in Helpers (its CFBundleIdentifier). The relative
# arsenal-worker-<family> links share arsenal-worker-native's signature and are sealed by the app as links.
signed_items() {
  local app="$1" f plist
  printf '.\t%s\n' "$APP_ID"
  for f in "$app"/Contents/Helpers/*; do
    [[ -L "$f" ]] && continue
    if [[ -d "$f" && "$f" == *.bundle ]]; then
      plist="$f/Contents/Info.plist"; [[ -f "$plist" ]] || plist="$f/Info.plist"
      [[ -f "$plist" ]] || continue  # a plain resource folder, sealed by the app
      printf 'Helpers/%s\t%s\n' "$(basename "$f")" "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$plist")"
    elif [[ -f "$f" && -x "$f" ]]; then
      printf 'Helpers/%s\tdev.arsenal.helper.%s\n' "$(basename "$f")" "$(basename "$f")"
    fi
  done
}

# Overall class of an app bundle: the app itself and every helper must agree; any development signature wins.
classify_app() {
  local app="$1" rel c target classes=()
  [[ -d "$app" ]] || die "not an app bundle: $app"
  [[ -x "$app/Contents/Helpers/ArsenalUpdateHelper" ]] || classes+=(unsigned)
  while IFS=$'\t' read -r rel _; do
    [[ "$rel" == . ]] && target="$app" || target="$app/Contents/$rel"
    c="$(codesign -dvv "$target" 2>&1 | classify_text)"
    classes+=("$c")
  done < <(signed_items "$app")
  local first="${classes[0]}" all_same=1
  for c in "${classes[@]}"; do
    [[ "$c" == development ]] && { echo development; return; }
    [[ "$c" == "$first" ]] || all_same=0
  done
  if [[ $all_same == 1 ]]; then echo "$first"; else echo mixed; fi
}

# ZIP -> extracted app in a temporary directory, which the caller removes.
unpack() {
  local zip="$1" dir="$2" app
  /usr/bin/unzip -q "$zip" -d "$dir" || die "cannot unzip $zip"
  app="$(find "$dir" -maxdepth 1 -name '*.app' -print | sed -n '1p')"
  [[ -n "$app" ]] || die "no .app at the top of $zip"
  echo "$app"
}

classify_path() {
  local path="$1" tmp app
  if [[ -d "$path" && "$path" == *.app ]]; then classify_app "$path"
  elif [[ -f "$path" ]]; then
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/release-identity.XXXXXX")"
    app="$(unpack "$path" "$tmp")" || { rm -rf "$tmp"; return 1; }
    classify_app "$app"; rm -rf "$tmp"
  else die "neither an app nor a zip: $path"; fi
}

# The app AND each signed helper/bundle: valid signature, and exactly `identifier X and certificate leaf = H"<pin>"`.
designated_pinned() {
  local app="$1" rel identifier target expected requirement pin
  pin="$(tr '[:upper:]' '[:lower:]' <<<"$PIN_SHA1")"
  codesign --verify --deep --strict "$app" >/dev/null 2>&1 || return 1
  while IFS=$'\t' read -r rel identifier; do
    [[ "$rel" == . ]] && target="$app" || target="$app/Contents/$rel"
    expected="identifier \"$identifier\" and certificate leaf = H\"$pin\""
    codesign --verify --strict -R "=$expected" "$target" >/dev/null 2>&1 || return 1
    requirement="$(codesign -d -r- "$target" 2>&1 | sed -n 's/^designated => //p')"
    # codesign prints simple identifiers without quotes; dotted ones keep them.
    requirement="$(sed -E 's/identifier "([^"]+)"/identifier \1/' <<<"$requirement")"
    expected="$(sed -E 's/identifier "([^"]+)"/identifier \1/' <<<"$expected")"
    [[ "$requirement" == "$expected" ]] || return 1
  done < <(signed_items "$app")
}

for_upload() {
  local path="$1" zip tmp app class count
  if [[ -d "$path" && "$path" != *.app ]]; then
    [[ ! -e "$path/$MARKER" ]] || die "refusing $path: it is marked local-only ($MARKER); only the CI-built release may be uploaded"
    count="$(find "$path" -maxdepth 1 -name 'Arsenal-*-arm64.zip' ! -name '*-symbols.zip' -print | wc -l | tr -d ' ')"
    [[ "$count" == 1 ]] || die "expected exactly one Arsenal-*-arm64.zip in $path (found $count)"
    path="$(find "$path" -maxdepth 1 -name 'Arsenal-*-arm64.zip' ! -name '*-symbols.zip' -print)"
  fi
  if [[ -f "$path" ]]; then
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/release-identity.XXXXXX")"
    app="$(unpack "$path" "$tmp")" || { rm -rf "$tmp"; return 1; }
  else
    app="$path"; tmp=""
  fi
  class="$(classify_app "$app")"
  if [[ "$class" != release ]]; then
    [[ -z "$tmp" ]] || rm -rf "$tmp"
    die "refusing to upload $path: signed as '$class', not '$AUTHORITY'. Upload only the CI build (a v* tag runs .github/workflows/release.yml)"
  fi
  if ! designated_pinned "$app"; then
    [[ -z "$tmp" ]] || rm -rf "$tmp"
    die "refusing to upload $path: its designated requirement is not the pinned $AUTHORITY certificate"
  fi
  [[ -z "$tmp" ]] || rm -rf "$tmp"
  echo "release-identity: $path is signed by $AUTHORITY with the pinned requirement"
}

[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0

case "${1:-}" in
  classify) [[ $# -eq 2 ]] || die "usage: classify APP|ZIP"; classify_path "$2" ;;
  text-class) [[ $# -eq 2 ]] || die "usage: text-class FILE"; classify_text <"$2" ;;
  pinned) [[ $# -eq 2 && -d "$2" ]] || die "usage: pinned APP"; designated_pinned "$2" || die "$2 is not signed with the pinned $AUTHORITY requirement" ;;
  mark-local)
    [[ $# -eq 2 && -d "$2" ]] || die "usage: mark-local DIR"
    zip="$(find "$2" -maxdepth 1 -name 'Arsenal-*-arm64.zip' ! -name '*-symbols.zip' -print | sed -n '1p')"
    [[ -n "$zip" ]] || die "no Arsenal-*-arm64.zip in $2"
    class="$(classify_path "$zip")"
    if [[ "$class" != release ]]; then
      echo "Signed as '$class', not $AUTHORITY: for local checks only. Never upload this build; release only the CI build (.github/workflows/release.yml)." >"$2/$MARKER"
    fi
    echo "$class" ;;
  for-upload) [[ $# -eq 2 ]] || die "usage: for-upload PATH"; for_upload "$2" ;;
  *) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
