#!/bin/bash
# Fresh-install smoke (review B1): the assembled, signed bundle on an empty support dir with NO developer overrides
# (no ARSENAL_CATALOG, ARSENAL_HELPERS, ARSENAL_STUB, ARSENAL_API_PORT; only ARSENAL_SUPPORT_DIR/ARSENAL_CACHE_DIR point at a
# new temp dir, because tests never touch the real ~/Library/Application Support/Arsenal) must open its bundled signed
# catalog through the default bundle source and Add/Get/Load/Speak/Unload a small model through the bundled stdio MCP bridge.
# Usage: smoke-fresh-install.sh <Arsenal.app> [model-id]      (default kokoro-82m: ~0.35 GB from Hugging Face)
# Get (including fp16 derivation) and Load/Speak/Unload each use gpulock with a 180-second deadline. No UI events.
# ARSENAL_SMOKE_LOCAL_SOURCE supplies verified local weights without changing the shipping recipe.
# Otherwise Get downloads real weights, so build.sh runs it only for the review candidate (ARSENAL_STRICT=1). One line per check.
set -euo pipefail
source "$(dirname "$0")/pkg-lib.sh"
# Private subprocess entry: only the parent invokes this through mlock gpulock.
if [[ "${1:-}" == --get || "${1:-}" == --cycle ]]; then
  mode="$1"
  APP="$2"; MODEL="$3"; support="$4"; cache="$5"; tmp="$6"; app_pid="$7"
  env_fresh=(env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/sbin:/sbin" "USER=${USER:-}" "TMPDIR=${TMPDIR:-/tmp}"
             "ARSENAL_SUPPORT_DIR=$support" "ARSENAL_CACHE_DIR=$cache" "ARSENAL_PLAYBACK_MUTED=1")
  call() { "${env_fresh[@]}" /usr/bin/python3 "$PKG_ROOT/scripts/smoke-fresh-install.py" "$1" "$support" "$MODEL" "$APP" "$tmp"; }
  if [[ "$mode" == --get ]]; then
    call get
    exit 0
  fi
  call cycle
  [[ -z "$(pgrep -P "$app_pid" || true)" ]] || die "worker left after unload"
  trap - EXIT
  echo "unload: no worker child remains"
  exit 0
fi
SOURCE_APP="$(cd "${1:?usage: smoke-fresh-install.sh <Arsenal.app> [model-id]}" && pwd)"; MODEL="${2:-kokoro-82m}"
tmp="$(mktemp -d /tmp/arsenal-fresh.XXXXXX)"
APP="$tmp/Arsenal.app"
support="$tmp/support"; cache="$tmp/cache"
env_fresh=(env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/sbin:/sbin" "USER=${USER:-}" "TMPDIR=${TMPDIR:-/tmp}"
           "ARSENAL_SUPPORT_DIR=$support" "ARSENAL_CACHE_DIR=$cache" "ARSENAL_PLAYBACK_MUTED=1")
app_pid=""
evidence="$PKG_ROOT/.build/pkg/fresh-install-smoke"
mkdir -p "$evidence"
cleanup() {
  local result=$? cleanup_result=0
  if [[ -n "$app_pid" ]] && kill -0 "$app_pid" 2>/dev/null; then
    kill -TERM "$app_pid" 2>/dev/null || true
    for _ in $(seq 1 50); do kill -0 "$app_pid" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$app_pid" 2>/dev/null || true
  fi
  /usr/bin/python3 -B "$PKG_ROOT/scripts/launch_services.py" cleanup-tree "$tmp" || cleanup_result=1
  # Preserve the useful transcript, then write a per-file deletion receipt before removing test data.
  for file in app.log get.log cycle.log source.json; do [[ ! -f "$tmp/$file" ]] || cp "$tmp/$file" "$evidence/$file" || cleanup_result=1; done
  /usr/bin/python3 - "$tmp" "$evidence/cleanup.json" <<'PY' || cleanup_result=1
import json, pathlib, shutil, sys
root, receipt = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
files = [{"path": str(p), "bytes": p.lstat().st_size} for p in sorted(root.rglob("*")) if p.is_file() or p.is_symlink()]
data = {"root": str(root), "files": files, "bytes": sum(f["bytes"] for f in files)}
receipt.write_text(json.dumps(data, indent=2) + "\n")
shutil.rmtree(root)
data["verified_removed"] = not root.exists()
receipt.write_text(json.dumps(data, indent=2) + "\n")
PY
  [[ ! -d "$tmp" ]] || rm -rf "$tmp"
  if (( result == 0 && cleanup_result != 0 )); then result=$cleanup_result; fi
  exit "$result"
}
trap cleanup EXIT
ditto "$SOURCE_APP" "$APP"
"${env_fresh[@]}" "$APP/Contents/MacOS/Arsenal" >"$tmp/app.log" 2>&1 &
app_pid=$!
for _ in $(seq 1 200); do [[ -s "$support/status.json" ]] && break; kill -0 "$app_pid" || die "app exited at launch: $(tail -3 "$tmp/app.log")"; sleep 0.1; done
[[ -s "$support/status.json" ]] || die "app wrote no status.json within 20 s"
echo "launch: app up on a fresh support dir, no developer overrides"

"${env_fresh[@]}" /usr/bin/python3 "$PKG_ROOT/scripts/smoke-fresh-install.py" bootstrap "$support" "$MODEL" "$APP" "$tmp" "${ARSENAL_SMOKE_LOCAL_SOURCE:-}"

start=$SECONDS
perl -e 'alarm 180; exec @ARGV' \
  "$PKG_ROOT/scripts/smoke-fresh-install.sh" --get "$APP" "$MODEL" "$support" "$cache" "$tmp" "$app_pid" \
  >"$tmp/get.log" 2>&1 || die "MCP Get $MODEL failed or GPU gate closed: $(tail -3 "$tmp/get.log")"
[[ -d "$support/models/$MODEL" ]] || die "model folder $MODEL was not installed"
[[ -n "$(ls "$support/weights/snapshots/$MODEL" 2>/dev/null)" ]] || die "no weights snapshot for $MODEL"
[[ ! -e "$support/tools" ]] || die "a native model Get fetched uv"
echo "get: $MODEL folder installed from the default source, weights verified, $((SECONDS - start)) s, no uv fetched ($(tail -1 "$tmp/get.log"))"
[[ -n "$(find "$support/weights/snapshots/$MODEL" -path '*/fp16/*.safetensors' -type f -print -quit)" ]] || die "fp16 derived weights missing"
perl -e 'alarm 180; exec @ARGV' \
  "$PKG_ROOT/scripts/smoke-fresh-install.sh" --cycle "$APP" "$MODEL" "$support" "$cache" "$tmp" "$app_pid" \
  >"$tmp/cycle.log" 2>&1 || die "Load/Speak/Unload failed or GPU gate closed: $(tail -3 "$tmp/cycle.log")"
cat "$tmp/cycle.log"
[[ ! -e "$support/tools" ]] || die "native model cycle fetched uv"

kill -TERM "$app_pid"
for _ in $(seq 1 100); do kill -0 "$app_pid" 2>/dev/null || break; sleep 0.1; done
! kill -0 "$app_pid" 2>/dev/null || die "app did not quit within 10 s of SIGTERM"
left="$(pgrep -f "$APP/Contents/" || true)"
[[ -z "$left" ]] || die "processes left after quit: $left"
app_pid=""
echo "quit: app exited, no worker/uv/python left"
