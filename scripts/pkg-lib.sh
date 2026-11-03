#!/bin/bash
# Shared by the packaging scripts (sourced, never run). Why a lib: build, assemble, sign and smoke must agree on
# the toolchain, the identity and the bundle layout, and none of them may drift from CONTRACTS.md §2.

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# CLT Swift compiles Swift/C++; Swift >= 6.4 gets the weak borrow shim in build.sh.
# Xcode is used only for its Metal compiler.
CLT=/Library/Developer/CommandLineTools
XCODE="${ARSENAL_XCODE_DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
# Xcode's own macOS SDK, named explicitly: after an Xcode update whose first-launch step has not run, xcrun falls back to
# the CLT SDK, which that Xcode then refuses to locate.
XCODE_SDK="$XCODE/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
xcode_xcrun() { DEVELOPER_DIR="$XCODE" SDKROOT="$XCODE_SDK" xcrun "$@"; }
MLX_SWIFT_PIN=901941965d82e4a216d4d117231d847d194c563d
# How many .metal files the pinned Cmlx Xcode target compiles; a different count means the pin moved.
MLX_METAL_FILES=10

# The user's profile: ARSENAL_SUPPORT_DIR, else ~/Library/Application Support/Arsenal.
profile_dir() {
  if [[ -n "${ARSENAL_SUPPORT_DIR:-}" ]]; then printf '%s\n' "$ARSENAL_SUPPORT_DIR"; return; fi
  printf '%s\n' "$HOME/Library/Application Support/Arsenal"
}

log() { printf 'pkg: %s\n' "$*" >&2; }
die() { printf 'pkg: %s\n' "$*" >&2; exit 1; }

clt_swift() { DEVELOPER_DIR="$CLT" "$CLT/usr/bin/swift" "$@"; }

# Release signing, Vella's scheme: one self-signed certificate "Arsenal Release Signing", pinned by SHA-1 in every
# designated requirement (scripts/sign.sh), in release.yml, release-identity.sh and install.sh. Its key lives only in
# the private backup and the GitHub environment release-signing; builds reach it through with-release-keychain.sh
# (local) or ci-keychain.sh (CI). Changing the pin is a rotation: every install asks for its permissions again and the
# updater refuses the new builds.
SIGN_NAME='Arsenal Release Signing'
RELEASE_SIGNING_SHA1=011C682945C225DBDEC95AEAA187BECBB0207282
BUNDLE_ID=dev.arsenal.app

# Print the signing identity (SHA-1). ARSENAL_SIGN_IDENTITY may only name the pin, unless ARSENAL_ALLOW_OTHER_IDENTITY=1
# (disposable test builds; such an app neither takes nor gives updates).
resolve_identity() {
  local identity="${ARSENAL_SIGN_IDENTITY:-$RELEASE_SIGNING_SHA1}"
  identity="$(tr '[:lower:]' '[:upper:]' <<<"$identity")"
  if [[ "$identity" != "$RELEASE_SIGNING_SHA1" && "${ARSENAL_ALLOW_OTHER_IDENTITY:-0}" != 1 ]]; then
    die "ARSENAL_SIGN_IDENTITY $identity is not the pinned $SIGN_NAME certificate $RELEASE_SIGNING_SHA1"
  fi
  printf '%s\n' "$identity"
}

# The shipped native worker: Workers/Package.swift is the one native build graph. It links every native family into
# one binary (one MLX copy, Vella's VellaWorker/VellaStreamingWorker recipe) and owns every family's tests, so pinned
# dependencies compile once per configuration. The bundle carries Helpers/arsenal-worker-native plus one relative
# symlink arsenal-worker-<family> per family, which is the name core launches and the binary dispatches on.
# Workers/<Family>/ packages hold the families' libraries; nothing builds them on their own.
NATIVE_DIR="$PKG_ROOT/Workers"
NATIVE_EXE=arsenal-worker-native
# Kit models (Workers/Kit) are one family each, kit-<model id>, read from each module's Entry.swift.
KIT_FAMILIES=($(sed -n 's/^    public static let modelID = "\(.*\)"$/kit-\1/p' "$PKG_ROOT"/Workers/Kit/Sources/*/Entry.swift | LC_ALL=C sort))
NATIVE_FAMILIES=(voices chat sound decisions d1 elephant flux birefnet language embed audio music restoration perception geometry qwenimage "${KIT_FAMILIES[@]}")
# Family packages linked into Native: their PORTED-FROM.md and licences ship with the app (Workers/Kit once for every kit model).
family_dirs() { local f; (( ${#KIT_FAMILIES[@]} )) && echo "$PKG_ROOT/Workers/Kit"; for f in "${NATIVE_FAMILIES[@]}"; do [[ "$f" == kit-* ]] && continue; if [[ "$f" == birefnet ]]; then echo "$PKG_ROOT/Workers/BiRefNetEngine"; continue; fi; if [[ "$f" == qwenimage ]]; then echo "$PKG_ROOT/Workers/QwenImage"; continue; fi; echo "$PKG_ROOT/Workers/$(tr '[:lower:]' '[:upper:]' <<< "${f:0:1}")${f:1}"; done; }
# MLX's shaders, compiled once per mlx-swift source, Metal compiler/SDK identity, macOS and flags into a cache shared by
# every worktree (Vella Worker/build-split.sh recipe: Xcode's Metal compiler, no fast math, one metallib for the one
# worker binary). Prints the cached mlx-swift_Cmlx.bundle; callers clone it into their own layout and never edit it.
ARSENAL_BUILD_CACHE="${ARSENAL_BUILD_CACHE:-$HOME/Library/Caches/dev.arsenal.build}"
mlx_metallib_bundle() {
  local source="$1" identity key cache bundle air sdk file out objects=() flags=(-Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions)
  [[ -d "$source/Source/Cmlx" ]] || die "mlx-swift checkout missing at $source; build the native graph first"
  [[ "$(git -C "$source" rev-parse HEAD)" == "$MLX_SWIFT_PIN" ]] || die "$source is not the family pin $MLX_SWIFT_PIN"
  xcode_xcrun metal --version >/dev/null 2>&1 || die "Metal Toolchain missing. Fix: xcodebuild -downloadComponent MetalToolchain"
  identity=$(/usr/bin/python3 "$PKG_ROOT/scripts/build-cache.py" metal "$XCODE")
  # Shader debug info and __FILE__ name the checkout as mlx-swift/…, never its absolute path. The key records the map's
  # form, not this worktree's path, so every worktree keeps sharing one cache entry.
  local prefix_map="-ffile-prefix-map=$(cd "$source" && pwd -P)=mlx-swift"
  key=$(/usr/bin/python3 "$PKG_ROOT/scripts/build-cache.py" key --context "$identity|$(sw_vers -productVersion)|${flags[*]} -ffile-prefix-map=<checkout>=mlx-swift|$MLX_SWIFT_PIN" "$source/Source/Cmlx")
  cache="$ARSENAL_BUILD_CACHE/mlx-metallib/$key"
  bundle="$cache/mlx-swift_Cmlx.bundle"
  # Receipts name bundle-relative paths, so an entry verifies wherever it was staged.
  local outputs=(mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib mlx-swift_Cmlx.bundle/Contents/Info.plist)
  if [[ -d "$cache" ]] && (cd "$cache" && /usr/bin/python3 "$PKG_ROOT/scripts/build-cache.py" check --key "$key" --receipt receipt.json "${outputs[@]}"); then
    log "reusing compiled MLX shaders ($cache)"; echo "$bundle"; return
  fi
  [[ ! -e "$cache" ]] || die "shared MLX shader cache entry $cache does not verify; delete it and rebuild"
  log "compiling MLX shaders into the shared cache"
  mkdir -p "$ARSENAL_BUILD_CACHE/mlx-metallib"
  local staging; staging="$(mktemp -d "$ARSENAL_BUILD_CACHE/mlx-metallib/.staging.XXXXXX")"
  air="$staging/air"; mkdir -p "$air" "$staging/mlx-swift_Cmlx.bundle/Contents/Resources"
  sdk="$(xcode_xcrun --show-sdk-path)"
  while IFS= read -r file; do
    out="$air/$(basename "$file" .metal).air"
    xcode_xcrun metal -c -target air64-apple-macos14.0 -isysroot "$sdk" "${flags[@]}" "$prefix_map" "$file" -o "$out" 2>>"$air/compile.log" \
      || { tail -20 "$air/compile.log" >&2; rm -rf "$staging"; die "Metal compile failed"; }
    objects+=("$out")
  done < <(find "$source/Source/Cmlx/mlx-generated/metal" -name '*.metal' | sort)
  (( ${#objects[@]} == MLX_METAL_FILES )) || { rm -rf "$staging"; die "expected $MLX_METAL_FILES .metal files at the mlx-swift pin, found ${#objects[@]}"; }
  xcode_xcrun metallib "${objects[@]}" -o "$staging/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
  cat > "$staging/mlx-swift_Cmlx.bundle/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>dev.arsenal.mlx-swift-Cmlx</string>
<key>CFBundleName</key><string>mlx-swift_Cmlx</string>
<key>CFBundlePackageType</key><string>BNDL</string>
<key>CFBundleVersion</key><string>1</string>
</dict></plist>
PLIST
  rm -rf "$air"
  (cd "$staging" && /usr/bin/python3 "$PKG_ROOT/scripts/build-cache.py" record --key "$key" --receipt receipt.json "${outputs[@]}")
  # rename(2) publishes the complete entry atomically; when another worktree published the same key first, keep theirs.
  /usr/bin/python3 -c 'import os, sys
try: os.rename(sys.argv[1], sys.argv[2])
except OSError: pass' "$staging" "$cache"
  rm -rf "$staging"
  (cd "$cache" && /usr/bin/python3 "$PKG_ROOT/scripts/build-cache.py" check --key "$key" --receipt receipt.json "${outputs[@]}") \
    || die "shared MLX shader cache entry $cache failed verification"
  echo "$bundle"
}

# Size budget for the whole bundle (small on disk; Vella, the bar, is 31 MB).

# Inspect all undefined symbols; weak borrow imports are safe on macOS 26. Fail closed if nm cannot inspect.
reject_borrow_symbols() {
  local binary symbols
  for binary in "$@"; do
    symbols="$(nm -m -u "$binary" 2>&1)" || die "could not inspect runtime symbols in $binary: $symbols"
    if printf '%s\n' "$symbols" | awk '/_swift_(init|end)Borrow/ && !/weak external/ { bad=1 } END { exit !bad }'; then
      die "$(basename "$binary") strongly imports Swift 6.4 borrow symbols; rebuild with the native worker compatibility shim"
    fi
  done
}
