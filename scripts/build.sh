#!/bin/bash
# Build, sign and smoke an Arsenal.app candidate into dist/. Installs nothing.
#   scripts/build.sh            candidate from this checkout (whatever targets exist; gaps are listed, not hidden)
#   ARSENAL_STRICT=1             a gap, a dirty tree or a missing app --smoke is a failure (use for the review candidate)
#   ARSENAL_PKG_COMPILE_ONLY=1   compile both release graphs and stop: nothing assembled, signed or published (scripts/ready.sh
#                                runs it beside the test suites, so the signed build afterwards finds its compile done)
# Signing needs "Arsenal Release Signing" (ARSENAL_SIGN_IDENTITY + ARSENAL_SIGN_KEYCHAIN): locally run this through the
# private with-release-keychain.sh wrapper, in CI release.yml sets them (scripts/check-identity.sh).
# Outputs: dist/Arsenal.app, dist/Arsenal.app.sha256 (per-file sha256, verify with `cd dist && shasum -c Arsenal.app.sha256`),
#          dist/BUILD-INFO.json, dist/Arsenal.symbols/ (unstripped binaries, to symbolicate crash reports).
# dist/Arsenal.app is replaced only after the new candidate signed and passed its smoke; a failed build leaves the old one.
set -euo pipefail
source "$(dirname "$0")/pkg-lib.sh"
cd "$PKG_ROOT"

# Every build and compile is heavy CPU: WORKER-RULES route it through the lock (direct by day, Vella's wrapper in a window).
if [[ -z "${ARSENAL_PKG_LOCKED:-}" && "${ARSENAL_WORK_LOCKED_ROOT:-}" != "$PKG_ROOT" ]]; then
  exec env ARSENAL_PKG_LOCKED=1 nice -n 10 scripts/mlock work pkg-build -- "$PKG_ROOT/scripts/build.sh" "$@"
fi

if [[ "${ARSENAL_STRICT:-0}" == 1 ]]; then scripts/lint.sh; fi

[[ "$(uname -m)" == arm64 ]] || die "Apple Silicon required"
[[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 26 ]] || die "macOS 26 or newer required"
[[ -x "$CLT/usr/bin/swift" ]] || die "Command Line Tools Swift missing. Fix: xcode-select --install"
STAGE="$PKG_ROOT/.build/pkg"
APP="$STAGE/stage/Arsenal.app"
DIST="$PKG_ROOT/dist"
smoke_tmp=""
cleanup_bundles() {
  local result=$? cleanup_result=0
  # AppKit registers even a directly executed bundle. Clean on failures as well as successful builds.
  for path in "$APP" "$DIST/Arsenal.app" "${smoke_tmp:+$smoke_tmp/Arsenal.app}"; do
    [[ -n "$path" ]] || continue
    /usr/bin/python3 -B scripts/launch_services.py unregister "$path" || cleanup_result=1
  done
  [[ -z "$smoke_tmp" ]] || rm -rf "$smoke_tmp"
  if (( result == 0 && cleanup_result != 0 )); then result=$cleanup_result; fi
  exit "$result"
}
trap cleanup_bundles EXIT
mkdir -p "$STAGE"
build_commit="$(git rev-parse HEAD)"

# An identical signed candidate is itself the reusable release artifact.
source_key=$(/usr/bin/python3 scripts/build-cache.py candidate)
metal_identity=""
if [[ -f "$NATIVE_DIR/Package.swift" ]]; then
  metal_identity=$(/usr/bin/python3 scripts/build-cache.py metal "$XCODE")
  checkout="$NATIVE_DIR/.build/checkouts/mlx-swift"
  if [[ -d "$checkout" ]]; then
    [[ "$(git -C "$checkout" rev-parse HEAD)" == "$MLX_SWIFT_PIN" ]] || die "the native worker resolves mlx-swift to a revision other than the family pin $MLX_SWIFT_PIN"
  fi
fi
candidate_key=$(/usr/bin/python3 scripts/build-cache.py key --context "$source_key|$metal_identity|$(clt_swift --version)|$(sw_vers -productVersion)|$XCODE|$(stat -f '%m:%z' "$CLT/usr/bin/swift" "$CLT/usr/bin/swiftc" "$XCODE/usr/bin/xcrun" "$XCODE/../Info.plist" 2>/dev/null || true)|$(resolve_identity)|${SDKROOT:-}|${MACOSX_DEPLOYMENT_TARGET:-}|${ARSENAL_STRICT:-0}|$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' -c 'Print :CFBundleVersion' Resources/Info.plist | tr '\n' ' ')" scripts/build.sh)
if [[ -z "$(git status --porcelain --untracked-files=no)" ]] \
  && /usr/bin/python3 scripts/build-cache.py check --key "$candidate_key" --receipt "$STAGE/candidate-cache.json" "$DIST/BUILD-INFO.json" "$DIST/Arsenal.app.sha256" \
  && [[ -d "$DIST/Arsenal.app" ]] \
  && codesign --verify --deep --strict "$DIST/Arsenal.app" \
  && (cd "$DIST" && shasum -a 256 -c Arsenal.app.sha256 >/dev/null) \
  && [[ -x "$DIST/Arsenal.app/Contents/MacOS/Arsenal" && -x "$DIST/Arsenal.app/Contents/Helpers/arsenal-mcp" && -x "$DIST/Arsenal.app/Contents/Helpers/ArsenalUpdateHelper" && -x "$DIST/Arsenal.app/Contents/Helpers/$NATIVE_EXE" ]] \
  && scripts/release-identity.sh pinned "$DIST/Arsenal.app" 2>/dev/null; then
  log "reusing verified signed candidate"
  echo "$DIST/Arsenal.app"; exit 0
fi

compile_only=false; [[ "${ARSENAL_PKG_COMPILE_ONLY:-0}" != 1 ]] || compile_only=true
if ! $compile_only; then scripts/check-identity.sh >/dev/null; fi
IDENTITY="$(resolve_identity)"

# --- compile --------------------------------------------------------------------------------------------------
# No build-machine path lands in a shipped binary: Swift debug info, C/C++/Objective-C __FILE__ and debug info, and the
# linker's debug map name paths relative to the checkout. What no flag reaches (Swift #filePath literals and SwiftPM's
# Bundle.module build-directory fallback) is neutralised by scrub-build-paths.py before signing, and checked after.
BUILD_ROOT="$(cd "$PKG_ROOT" && pwd -P)"
prefix_maps=(-Xswiftc -file-prefix-map -Xswiftc "$BUILD_ROOT=." -Xswiftc -debug-prefix-map -Xswiftc "$BUILD_ROOT=."
  -Xcc "-ffile-prefix-map=$BUILD_ROOT=." -Xcxx "-ffile-prefix-map=$BUILD_ROOT=." -Xlinker -oso_prefix -Xlinker "$BUILD_ROOT/")
# Native SwiftPM applies whole-module -Osize to the app and helpers only; inference keeps main's flags.
log "compiling the root package (CLT Swift)"
clt_swift build --package-path "$PKG_ROOT" -c release --jobs "${ARSENAL_BUILD_JOBS:-2}" --build-system native -Xswiftc -Osize -Xlinker -no_exported_symbols "${prefix_maps[@]}"
root_bin="$(clt_swift build --package-path "$PKG_ROOT" -c release --build-system native --show-bin-path)"
reject_borrow_symbols "$root_bin/Arsenal" "$root_bin/arsenal-mcp" "$root_bin/ArsenalUpdateHelper"
"$root_bin/Arsenal" catalog validate --all
"$root_bin/Arsenal" catalog verify

metal_source=""
if [[ -f "$NATIVE_DIR/Package.swift" ]]; then
  log "compiling the native worker ($(IFS=,; echo "${NATIVE_FAMILIES[*]}"))"
  link=()
  swift_version="$(clt_swift --version 2>&1 | sed -n 's/.*Swift version \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p')"
  read -r swift_major swift_minor <<<"$swift_version" || true
  [[ "${swift_major:-}" =~ ^[0-9]+$ && "${swift_minor:-}" =~ ^[0-9]+$ ]] || die "Could not read the Command Line Tools Swift version."
  # swift-collections' uncalled Optional._borrow() references swift_initBorrow with Swift 6.4.
  # macOS 26 lacks it. Port Vella's weak declaration and link-wide mismatch resolution.
  # The latter also weakens Foundation constants exported by macOS 26. Swift 6.3 flags stay unchanged.
  if (( swift_major > 6 || (swift_major == 6 && swift_minor >= 4) )); then
    shim="$NATIVE_DIR/.build/weak-swift-borrow.o"
    mkdir -p "$NATIVE_DIR/.build"
    shim_key=$(/usr/bin/python3 scripts/build-cache.py key --context "$swift_version|$(stat -f '%m:%z' "$CLT/usr/bin/clang")" scripts/build.sh)
    if ! /usr/bin/python3 scripts/build-cache.py check --key "$shim_key" --receipt "$shim.cache.json" "$shim"; then
      printf '%s\n' 'extern void swift_initBorrow(void) __attribute__((weak_import));' \
        '__attribute__((used)) void *const arsenal_weak_swift_initBorrow = (void *)&swift_initBorrow;' |
        DEVELOPER_DIR="$CLT" "$CLT/usr/bin/clang" -target arm64-apple-macos26.0 -x c -c - -o "$shim"
      /usr/bin/python3 scripts/build-cache.py record --key "$shim_key" --receipt "$shim.cache.json" "$shim"
    fi
    link=(-Xlinker "$shim" -Xlinker -weak_reference_mismatches -Xlinker weak)
  fi
  clt_swift build --package-path "$NATIVE_DIR" -c release --jobs "${ARSENAL_BUILD_JOBS:-2}" --build-system native --product "$NATIVE_EXE" -Xlinker -no_exported_symbols -Xlinker -dead_strip ${link[@]+"${link[@]}"} "${prefix_maps[@]}"
  bin="$(clt_swift build --package-path "$NATIVE_DIR" -c release --build-system native --show-bin-path)"
  reject_borrow_symbols "$bin/$NATIVE_EXE"
  checkout="$NATIVE_DIR/.build/checkouts/mlx-swift"
  [[ "$(git -C "$checkout" rev-parse HEAD)" == "$MLX_SWIFT_PIN" ]] || die "the native worker resolves mlx-swift to a revision other than the family pin $MLX_SWIFT_PIN"
  metal_source="$checkout"
fi

# MLX's shaders: one metallib for the one worker binary, from the cache shared by every worktree (pkg-lib.sh).
if [[ -n "$metal_source" ]]; then
  metal_bundle="$(mlx_metallib_bundle "$metal_source")"
  rm -rf "$STAGE/mlx-swift_Cmlx.bundle"
  cp -cR "$metal_bundle" "$STAGE/mlx-swift_Cmlx.bundle"
fi

if $compile_only; then
  trap - EXIT
  log "release compile done (compile only: nothing assembled, signed or published)"
  exit 0
fi

# --- assemble, strip, sign --------------------------------------------------------------------------------------
rm -rf "$STAGE/stage" && mkdir -p "$STAGE/stage"
scripts/assemble.sh "$APP" | tee "$STAGE/assemble.log"
gaps=()
while IFS= read -r line; do gaps+=("${line#missing: }"); done < <(grep '^missing: ' "$STAGE/assemble.log" || true)
[[ -x "$APP/Contents/Helpers/arsenal-mcp" ]] || die "assembled bundle is missing the stdio MCP bridge"
[[ ! -e "$APP/Contents/Helpers/arsenal" ]] || die "assembled bundle contains the retired CLI helper"

/usr/bin/python3 scripts/scrub-build-paths.py scrub "$BUILD_ROOT" "$APP"
# Strip debug info from our own code, keeping unstripped copies outside the app for symbolication (Vella's
# strip-release.sh: strip -S -x, UUID unchanged).
symbols="$STAGE/stage/Arsenal.symbols"; mkdir -p "$symbols"
for binary in "$APP/Contents/MacOS/Arsenal" "$APP/Contents/Helpers/arsenal-mcp" "$APP/Contents/Helpers/ArsenalUpdateHelper" "$APP"/Contents/Helpers/arsenal-worker-*; do
  [[ -f "$binary" && ! -L "$binary" ]] || continue  # family symlinks share the native worker's image
  uuid="$(dwarfdump --uuid "$binary" | awk '{print $2}')"
  # Keep symbol archives named by bundle-relative path, preserving each executable's location.
  name="$(echo "${binary#"$APP/Contents/"}" | tr / -)"
  cp "$binary" "$symbols/$name"
  echo "$uuid $name" >> "$symbols/UUIDS.txt"
  /usr/bin/strip -S -x "$binary"
  [[ "$uuid" == "$(dwarfdump --uuid "$binary" | awk '{print $2}')" ]] || die "strip changed the UUID of $name"
done

reject_borrow_symbols "$APP/Contents/MacOS/Arsenal" "$APP/Contents/Helpers/arsenal-mcp" "$APP/Contents/Helpers/ArsenalUpdateHelper"
scripts/sign.sh "$APP"

# --- verify the signed image, not just the parts ----------------------------------------------------------------
/usr/bin/python3 scripts/scrub-build-paths.py check "$BUILD_ROOT" "$HOME" "$APP" >"$STAGE/path-check.log" \
  || { cat "$STAGE/path-check.log" >&2; gaps+=("build-machine paths in shipped binaries (see .build/pkg/path-check.log)"); }
scripts/smoke-helpers.sh "$APP"
smoke_tmp="$(mktemp -d "${TMPDIR:-/tmp}/arsenal-pkg-smoke.XXXXXX")"
ditto "$APP" "$smoke_tmp/Arsenal.app"
set +e
# The verdict is the one line on stdout; macOS writes its own notices to stderr for a bundle under $TMPDIR, so those are
# logged, not matched.
app_smoke="$(ARSENAL_STUB=1 ARSENAL_SUPPORT_DIR="$smoke_tmp/support" ARSENAL_CACHE_DIR="$smoke_tmp/cache" \
  perl -e 'alarm 60; exec @ARGV' "$smoke_tmp/Arsenal.app/Contents/MacOS/Arsenal" --smoke 2>"$smoke_tmp/smoke-stderr.log")"
app_smoke_code=$?
set -e
log "app --smoke: ${app_smoke:-<no output>}"
[[ -s "$smoke_tmp/smoke-stderr.log" ]] && log "app --smoke stderr: $(cat "$smoke_tmp/smoke-stderr.log")"
[[ "$app_smoke" == "smoke ok"* && "$app_smoke_code" == 0 ]] || gaps+=("Arsenal --smoke failed (exit $app_smoke_code): $app_smoke")

# 3.1 validation uses deterministic fixtures only; real model weights are never loaded.
fresh_receipt="$STAGE/fresh-install-smoke/source.json"
fresh_smoke_ran=0
if [[ "${ARSENAL_STRICT:-0}" == 1 ]]; then
  rm -f "$fresh_receipt"  # A previous candidate's receipt cannot establish this one's source.
  if /usr/bin/python3 scripts/smoke-dropin.py "$APP" --evidence "$STAGE/dropin-smoke"; then
    if fresh_source="$(/usr/bin/python3 - "$fresh_receipt" <<'PYSOURCE'
import json, sys
try:
    receipt = json.load(open(sys.argv[1]))
    labels = {"network": "network Hugging Face", "verified-local": "verified local copy"}
    print(labels[receipt["source"]])
except (OSError, ValueError, KeyError, TypeError):
    sys.exit("Fresh-install smoke did not provide a valid source receipt.")
PYSOURCE
)"; then
      fresh_smoke_ran=1; log "fresh-install source: $fresh_source"
    else
      gaps+=("fresh-install smoke source receipt missing or invalid")
    fi
  else
    gaps+=("drop-in smoke failed (16 descriptors + native Swift fixtures + MCP/HTTP + idle swap + rollback)")
  fi
fi

# Size is reported, not gated.
/usr/bin/python3 scripts/native-layout.py verify "$APP" "${NATIVE_FAMILIES[@]}" > "$STAGE/native-layout.json"
bundle_bytes="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bytes"])' "$STAGE/native-layout.json")"
log "bundle: $bundle_bytes bytes ($(awk -v b="$bundle_bytes" 'BEGIN {printf "%.1f", b / 1048576}') MiB)"

[[ "$(git rev-parse HEAD)" == "$build_commit" ]] || gaps+=("checkout changed during compilation; rebuild this commit")

dirty=false
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { dirty=true; gaps+=("working tree has uncommitted changes"); }
if [[ "${ARSENAL_STRICT:-0}" == 1 && ${#gaps[@]} -gt 0 ]]; then
  printf 'pkg: STRICT: %s\n' "${gaps[@]}" >&2; die "candidate not published to dist/"
fi

# --- publish to dist/ -------------------------------------------------------------------------------------------
mkdir -p "$DIST"
rm -rf "$DIST/Arsenal.app" "$DIST/Arsenal.symbols"
ditto "$APP" "$DIST/Arsenal.app"
ditto "$symbols" "$DIST/Arsenal.symbols"
codesign --verify --deep --strict "$DIST/Arsenal.app"
scripts/smoke-helpers.sh "$DIST/Arsenal.app" >/dev/null
# LaunchServices registers a freshly written bundle asynchronously, so a registration can reappear after the first
# unregister; settle it (a few tries) before the strict check.
for attempt in 1 2 3 4 5; do
  /usr/bin/python3 -B scripts/launch_services.py unregister "$APP" "$DIST/Arsenal.app" "$smoke_tmp/Arsenal.app"
  [[ "${ARSENAL_STRICT:-0}" == 1 ]] || break
  sleep 2
  /usr/bin/python3 -B scripts/launch_services.py check >/dev/null 2>&1 && break
  (( attempt < 5 )) || /usr/bin/python3 -B scripts/launch_services.py check
done
(cd "$DIST" && find Arsenal.app -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256 > Arsenal.app.sha256 && shasum -a 256 -c Arsenal.app.sha256 >/dev/null)

GAPS="$(IFS=$'\n'; echo "${gaps[*]-}")" DIRTY="$dirty" APP_SMOKE="$app_smoke" IDENTITY="$IDENTITY" BUNDLE_BYTES="$bundle_bytes" \
FRESH_INSTALL_RAN="$fresh_smoke_ran" FRESH_INSTALL_RECEIPT="$fresh_receipt" \
/usr/bin/python3 - "$DIST" "$PKG_ROOT" <<'PY'
import json, os, subprocess, sys, datetime, plistlib, hashlib
dist, root = sys.argv[1:3]
run = lambda *a: subprocess.run(a, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
app = os.path.join(dist, "Arsenal.app")
plist = plistlib.load(open(os.path.join(app, "Contents/Info.plist"), "rb"))
catalog = json.load(open(os.path.join(app, "Contents/Resources/catalog/catalog.json")))
sign = run("codesign", "-dvv", app)
field = lambda key: next((l.split("=", 1)[1] for l in sign.splitlines() if l.startswith(key + "=")), None)
helpers = sorted(os.listdir(os.path.join(app, "Contents/Helpers")))
fresh_smoke = {"status": "not_run", "reason": "non-strict build"}
if os.environ["FRESH_INSTALL_RAN"] == "1":
    fresh_smoke = {
        "status": "passed",
        "steps": ["16 descriptor resolutions", "native Swift new kind", "Refresh", "MCP approval", "Get", "Load fixture", "HTTP JSON + MCP", "idle revision swap", "rollback", "Unload", "Quit"],
        "source_receipt": json.load(open(os.environ["FRESH_INSTALL_RECEIPT"])),
    }
info = {
    "app": "Arsenal", "bundle_id": plist["CFBundleIdentifier"],
    "version": plist.get("CFBundleShortVersionString"), "build": plist.get("CFBundleVersion"),
    "commit": subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "dirty": os.environ["DIRTY"] == "true",
    "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "host": {"macos": run("sw_vers", "-productVersion").strip(), "arch": run("uname", "-m").strip()},
    "toolchain": {
        "swift": run("/Library/Developer/CommandLineTools/usr/bin/swift", "--version").splitlines()[0],
        "metal": next(iter(run("xcrun", "metal", "--version").splitlines()), None),
        "mlx_swift_pin": "901941965d82e4a216d4d117231d847d194c563d",
    },
    "bundle_bytes": int(os.environ["BUNDLE_BYTES"]),
    "catalog": {"source": "bundle", "sequence": catalog["sequence"], "source_commit": catalog["source_commit"],
                "models": len(catalog["models"]), "kind_files": len(catalog["kind_files"])},
    "licence": "MIT",
    "signing": {
        "identity_sha1": os.environ["IDENTITY"], "authority": field("Authority"), "identifier": field("Identifier"),
        "designated_requirement": next((l.split("=> ", 1)[1] for l in run("codesign", "-d", "-r-", app).splitlines() if l.startswith("designated => ")), None),
        "hardened_runtime": "runtime" in (field("CodeDirectory") or "") or "flags=0x10000(runtime)" in sign,
        "self_signed_not_notarised": True,
    },
    "helpers": helpers,
    "app_smoke": os.environ["APP_SMOKE"],
    "fresh_install_smoke": fresh_smoke,
    "gaps": [g for g in os.environ["GAPS"].splitlines() if g],
    "checksums": "Arsenal.app.sha256 (sha256 of that file: %s)" % hashlib.sha256(open(os.path.join(dist, "Arsenal.app.sha256"), "rb").read()).hexdigest(),
}
json.dump(info, open(os.path.join(dist, "BUILD-INFO.json"), "w"), indent=2)
open(os.path.join(dist, "BUILD-INFO.json"), "a").write("\n")
PY
if (( ${#gaps[@]} > 0 )); then printf 'pkg: gap: %s\n' "${gaps[@]}" >&2; fi
if [[ "$dirty" == false && ${#gaps[@]} == 0 && "$source_key" == "$(/usr/bin/python3 scripts/build-cache.py candidate)" ]]; then
  /usr/bin/python3 scripts/build-cache.py record --key "$candidate_key" --receipt "$STAGE/candidate-cache.json" "$DIST/BUILD-INFO.json" "$DIST/Arsenal.app.sha256"
fi
echo "$DIST/Arsenal.app"
