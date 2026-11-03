#!/bin/bash
# Lay out an unsigned Arsenal.app from already-built products (scripts/build.sh compiles first).
# Usage: assemble.sh <app-path>      Layout is CONTRACTS.md §2:
#   Contents/MacOS/Arsenal  Contents/Helpers/{arsenal-mcp, ArsenalUpdateHelper, arsenal-worker-native, arsenal-worker-<family> -> arsenal-worker-native,
#                                           *.bundle, mlx-swift_Cmlx.bundle}
#   Contents/Info.plist
#   Contents/Resources/{LICENSE, NOTICE, THIRD_PARTY_NOTICES.txt, Licenses/, Arsenal.icns, MenuBarTemplate*.png,
#                       catalog-public-key, catalog/{catalog.json(.sig), source.json}, SKILL.md, DECISIONS.md, ADDING.md, models/, kinds/, schemas/, runners/, figures/}
# Things other workers own and may not have merged yet are reported on stdout as "missing: …" (build.sh records
# them in BUILD-INFO.json; ARSENAL_STRICT=1 turns them into a failure). Nothing is guessed to fill them.
set -euo pipefail
source "$(dirname "$0")/pkg-lib.sh"
APP="${1:?usage: assemble.sh <app-path>}"
STAGE="$PKG_ROOT/.build/pkg"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Helpers" "$APP/Contents/Resources"
missing() { echo "missing: $*"; }

root_bin="$(clt_swift build --package-path "$PKG_ROOT" -c release --build-system native --show-bin-path)"
cp "$root_bin/Arsenal" "$APP/Contents/MacOS/Arsenal"
cp "$root_bin/ArsenalUpdateHelper" "$APP/Contents/Helpers/ArsenalUpdateHelper"
# Stdio MCP bridge; inference remains inside the app.
cp "$root_bin/arsenal-mcp" "$APP/Contents/Helpers/arsenal-mcp"

# --- the native worker (one binary, one symlink per family) --------------------------------------------------------
if [[ -f "$NATIVE_DIR/Package.swift" ]]; then
  bin="$(clt_swift build --package-path "$NATIVE_DIR" -c release --build-system native --show-bin-path)"
  [[ -x "$bin/$NATIVE_EXE" ]] || die "$NATIVE_EXE not built in $bin"
  cp "$bin/$NATIVE_EXE" "$APP/Contents/Helpers/$NATIVE_EXE"
  # Relative links, so the sealed bundle stays relocatable; codesign seals them as links.
  for family in "${NATIVE_FAMILIES[@]}"; do ln -s "$NATIVE_EXE" "$APP/Contents/Helpers/arsenal-worker-$family"; done
  # SwiftPM resource bundles (Bundle.module) are looked up next to the executable; MLX's is replaced by ours below.
  resource_list="$STAGE/native-resources.txt"
  /usr/bin/python3 scripts/native-layout.py resources "$bin" > "$resource_list"
  while IFS= read -r bundle; do
    target="$APP/Contents/Helpers/$(basename "$bundle")"
    cp -R "$bundle" "$target"
    chmod -R u+w "$target"
    /usr/bin/python3 scripts/native-layout.py compact "$target"
    # SwiftPM's flat resource bundles have no Info.plist, and codesign refuses a *.bundle directory without one when it
    # sits in Helpers/. A root Info.plist makes it a signable shallow bundle; Bundle.module lookups are unaffected
    # (checked: Bundle.main of a Helpers executable is Helpers/, and Bundle(path:) resolves resources in the flat layout).
    if [[ ! -f "$target/Info.plist" && ! -d "$target/Contents" ]]; then
      name="$(basename "$target" .bundle)"
      cat > "$target/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>dev.arsenal.resources.$(echo "$name" | tr -c 'A-Za-z0-9.\n' '-')</string>
<key>CFBundleName</key><string>$name</string>
<key>CFBundlePackageType</key><string>BNDL</string>
<key>CFBundleVersion</key><string>1</string>
</dict></plist>
PLIST
    fi
  done < "$resource_list"
  # Licences of every family linked in and of every SwiftPM dependency the binary was built from, kept with the app.
  licences="$APP/Contents/Resources/Licenses/native"
  mkdir -p "$licences"
  while IFS= read -r dir; do
    for file in "$dir"/LICENSE* "$dir"/Licenses/*; do [[ -f "$file" ]] && cp "$file" "$licences/$(basename "$dir")-$(basename "$file")"; done
  done < <(family_dirs)
  for checkout in "$NATIVE_DIR"/.build/checkouts/*; do
    [[ -d "$checkout" ]] || continue
    for file in "$checkout"/LICENSE* "$checkout"/COPYING* "$checkout"/NOTICE*; do
      [[ -f "$file" ]] && cp "$file" "$licences/$(basename "$checkout")-$(basename "$file")"
    done
  done
  # SwiftPM checkouts are read-only; copies keep that mode, and xattr -dr (the updater, the cask) fails on them.
  chmod -R u+w,go+r "$licences"
  [[ -s "$STAGE/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ]] || die "MLX metallib missing: run scripts/build.sh, not assemble.sh alone"
  cp -R "$STAGE/mlx-swift_Cmlx.bundle" "$APP/Contents/Helpers/"
  /usr/bin/python3 "$PKG_ROOT/scripts/quality-identities.py" "$APP/Contents/Resources/quality-identities.json"
  workers=1
else
  workers=0
  missing "Workers/Package.swift (the native worker graph)"
fi

# --- Resources --------------------------------------------------------------------------------------------------
res="$APP/Contents/Resources"
[[ -f "$PKG_ROOT/Resources/Info.plist" ]] || die "Resources/Info.plist is missing; it carries the version, build number and update repository"
cp "$PKG_ROOT/Resources/Info.plist" "$APP/Contents/Info.plist"
plist="$APP/Contents/Info.plist"
set_plist() { /usr/libexec/PlistBuddy -c "Set :$1 $2" "$plist" 2>/dev/null || /usr/libexec/PlistBuddy -c "Add :$1 string $2" "$plist"; }
# The packaging contract: whatever the app worker wrote, the bundle must launch as the menu-bar agent that Helpers expect.
set_plist CFBundleIconFile Arsenal
set_plist CFBundleExecutable Arsenal
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$plist")" == "$BUNDLE_ID" ]] || die "Info.plist bundle id is not $BUNDLE_ID"
[[ "$(/usr/libexec/PlistBuddy -c 'Print :LSUIElement' "$plist")" == true ]] || die "Info.plist lacks LSUIElement=true (menu-bar agent)"
# Version and build number live in one place, Resources/Info.plist, as in Vella: a count of commits would differ between
# this checkout, a fresh public export and CI's shallow clone. Builds stay above the last count-based build (1433).
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$plist")"
build_number="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$plist")"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Info.plist CFBundleShortVersionString '$version' is not MAJOR.MINOR.PATCH"
[[ -z "${ARSENAL_BUILD_VERSION:-}" || "$ARSENAL_BUILD_VERSION" == "$version" ]] || die "ARSENAL_BUILD_VERSION $ARSENAL_BUILD_VERSION differs from Resources/Info.plist $version; bump the plist"
[[ "$build_number" =~ ^[0-9]+$ ]] && (( build_number > 1433 )) || die "Info.plist CFBundleVersion '$build_number' must be a number above 1433"
# Every build names the public repository for updates; a fork build sets ARSENAL_UPDATE_REPO= (empty) to drop the key.
if [[ -n "${ARSENAL_UPDATE_REPO+set}" && -z "$ARSENAL_UPDATE_REPO" ]]; then
  /usr/libexec/PlistBuddy -c 'Delete :ArsenalUpdateRepo' "$plist" 2>/dev/null || true
else
  [[ "$(/usr/libexec/PlistBuddy -c 'Print :ArsenalUpdateRepo' "$plist" 2>/dev/null)" =~ ^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$ ]] \
    || die "Info.plist lacks ArsenalUpdateRepo (owner/name); set ARSENAL_UPDATE_REPO= only for a fork build without updates"
fi
plutil -lint "$plist" >/dev/null

# Icons are rendered at build time from the cannon mark's source SVGs (Resources/Brand/icon.svg for the app, glyph.svg
# for the menu-bar template); no binary assets in the repo.
icons="$STAGE/icons"
mkdir -p "$icons"
clt_swiftc="$CLT/usr/bin/swiftc"
icon_key=$(/usr/bin/python3 scripts/build-cache.py key --context "$(stat -f '%m:%z' "$clt_swiftc")|$(sw_vers -productVersion)" scripts/icon.swift scripts/assemble.sh Resources/Brand/icon.svg Resources/Brand/glyph.svg Resources/Brand/icon-16.svg Resources/Brand/icon-32.svg)
icon_outputs=("$icons/Arsenal.icns" "$icons/MenuBarTemplate.png" "$icons/MenuBarTemplate@2x.png")
if ! /usr/bin/python3 scripts/build-cache.py check --key "$icon_key" --receipt "$icons/cache.json" "${icon_outputs[@]}"; then
  DEVELOPER_DIR="$CLT" "$clt_swiftc" -O -sdk "$CLT/SDKs/MacOSX.sdk" "$PKG_ROOT/scripts/icon.swift" -o "$icons/icon" 2>/dev/null
  "$icons/icon" "$PKG_ROOT/Resources/Brand/icon.svg" "$PKG_ROOT/Resources/Brand/glyph.svg" "$icons" \
    "$PKG_ROOT/Resources/Brand/icon-16.svg" "$PKG_ROOT/Resources/Brand/icon-32.svg"
  iconset="$icons/Arsenal.iconset"
  rm -rf "$iconset" && mkdir -p "$iconset"
  for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$icons/app-icon.png" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    sips -z "$((size * 2))" "$((size * 2))" "$icons/app-icon.png" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
  done
  # The smallest sizes use the mark's hand-tuned small sources (heavier strokes), not a downscale of the 1024 drawing.
  cp "$icons/app-icon-16.png" "$iconset/icon_16x16.png"
  cp "$icons/app-icon-32.png" "$iconset/icon_16x16@2x.png"
  cp "$icons/app-icon-32.png" "$iconset/icon_32x32.png"
  iconutil -c icns "$iconset" -o "$icons/Arsenal.icns"
  /usr/bin/python3 scripts/build-cache.py record --key "$icon_key" --receipt "$icons/cache.json" "${icon_outputs[@]}"
else
  log "reusing generated icons"
fi
cp "$icons/Arsenal.icns" "$res/"
cp "$icons/MenuBarTemplate.png" "$icons/MenuBarTemplate@2x.png" "$res/"

# Review B1: a fresh install must open a verified catalog without a cached copy or developer overrides, so the signed
# seed pair (core reads Resources/catalog/catalog.json[.sig]) and its public key are required, never optional.
for file in Resources/catalog-public-key Resources/catalog/catalog.json Resources/catalog/catalog.json.sig; do
  [[ -s "$PKG_ROOT/$file" ]] || die "$file is missing (fetch worker: scripts/catalog-refresh.sh); a candidate without the signed catalog has no models"
done
cp "$PKG_ROOT/Resources/catalog-public-key" "$res/"
mkdir -p "$res/catalog"
cp "$PKG_ROOT/Resources/catalog/catalog.json" "$PKG_ROOT/Resources/catalog/catalog.json.sig" "$res/catalog/"
# v2 has no checkout dependency: every instruction file is bundled exactly as authenticated by the catalog.
printf '{"source":"bundle"}\n' > "$res/catalog/source.json"
/usr/bin/python3 "$PKG_ROOT/scripts/bundle-inventory.py" copy "$PKG_ROOT" "$res"
for doc in SKILL.md DECISIONS.md ADDING.md; do
  if [[ -f "$PKG_ROOT/skill/$doc" ]]; then cp "$PKG_ROOT/skill/$doc" "$res/$doc"; else missing "skill/$doc (MCP worker)"; fi
done
for folder in schemas runners figures; do
  [[ -d "$PKG_ROOT/Resources/$folder" ]] && cp -R "$PKG_ROOT/Resources/$folder" "$res/$folder"
done
# The add-a-model kit (docs/ADD-A-MODEL.md), so an app installed without the repository has it: read-only here, its
# working folders live in ~/Library/Caches/Arsenal-new-models. Its own tests stay in the repository.
mkdir -p "$res/kit"
cp "$PKG_ROOT/kit/new-model.py" "$PKG_ROOT/kit/folder_check.py" "$res/kit/"
cp -R "$PKG_ROOT/kit/runner" "$res/kit/runner"
# --- licence and notices ----------------------------------------------------------------------------------------
# Arsenal is MIT: LICENSE is the MIT text, NOTICE the repo header plus what was
# ported (Workers/*/PORTED-FROM.md) and what is bundled (docs/notices/*.md), THIRD_PARTY_NOTICES.txt the latter alone.
MIT_SHA256=67f9e8393e7f1c5afc0824047c8e7c3b3ac5666309916a75a41dd647a80fea56
[[ "$(shasum -a 256 "$PKG_ROOT/LICENSE" 2>/dev/null | cut -d' ' -f1)" == "$MIT_SHA256" ]] || die "LICENSE is missing or is not the Arsenal MIT text"
[[ -s "$PKG_ROOT/NOTICE" ]] || die "NOTICE is missing"
section() { echo "================================================================"; echo "[$1]"; echo; cat "$2"; echo; }
notices_text() {
  for file in "$PKG_ROOT"/docs/notices/*.md; do [[ -f "$file" ]] && section "$(basename "$file" .md)" "$file"; done
}
{
  echo "Arsenal third-party notices"
  echo "Generated by scripts/assemble.sh from docs/notices/*.md. Full licence texts are in Licenses/ next to this file."
  echo
  notices_text
} > "$res/THIRD_PARTY_NOTICES.txt"
{
  cat "$PKG_ROOT/NOTICE"
  echo
  echo "Ported and vendored code"
  echo "------------------------"
  ported=0
  while IFS= read -r dir; do
    [[ -f "$dir/PORTED-FROM.md" ]] && { section "$(basename "$dir") (Workers/$(basename "$dir")/PORTED-FROM.md)" "$dir/PORTED-FROM.md"; ported=1; }
  done < <(family_dirs)
  (( ported )) || echo "(none in this build)"
  echo
  echo "Third-party notices"
  echo "-------------------"
  notices_text
} > "$res/NOTICE"
cp "$PKG_ROOT/LICENSE" "$res/LICENSE"
mkdir -p "$res/Licenses"
cp "$PKG_ROOT"/docs/notices/LICENSE-* "$res/Licenses/" 2>/dev/null || true

# Preserve every attributed licence path while storing identical texts once.
/usr/bin/python3 - "$res/Licenses" <<'PYLICENSES'
import hashlib, os, pathlib, sys
root = pathlib.Path(sys.argv[1]); retained = {}
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.is_symlink(): continue
    key = hashlib.sha256(path.read_bytes()).digest()
    if key in retained:
        target = retained[key]
        path.unlink(); path.symlink_to(os.path.relpath(target, path.parent))
    else:
        retained[key] = path
PYLICENSES

echo "assembled: $APP ($workers native worker binary: ${NATIVE_FAMILIES[*]})"
