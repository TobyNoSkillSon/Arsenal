#!/bin/bash
# Smoke the shipped helpers before a candidate is declared good. No model is loaded, no GPU work is done.
# Usage: smoke-helpers.sh <Arsenal.app>
#   Helpers/arsenal-worker-native is the only worker binary (one MLX copy) and refuses to start under its own name;
#   each Helpers/arsenal-worker-<family> is a relative link to it that launches headless (not an app, no window owner),
#   sends the wire handshake (first frame = hello, proto 1, its family, its own pid), answers `ping`, exits 0 on EOF;
#   LICENSE (MIT), NOTICE and THIRD_PARTY_NOTICES.txt are in Resources;
#   the MLX metallib is present and is a Metal library; `arsenal-mcp --help` runs from the bundle; no uv is bundled.
# Exits non-zero on the first failure. One line per check on stdout.
set -euo pipefail
source "$(dirname "$0")/pkg-lib.sh"
APP="${1:?usage: smoke-helpers.sh <Arsenal.app>}"
[[ -d "$APP/Contents/Helpers" ]] || die "$APP has no Contents/Helpers"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/arsenal-smoke.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

# Helpers (workers need the metallib next to them; checked for any worker present).
workers=()
native="$APP/Contents/Helpers/$NATIVE_EXE"
if [[ -e "$native" ]]; then
  [[ -f "$native" && ! -L "$native" && -x "$native" ]] || die "$NATIVE_EXE is not an executable file"
  for family in "${NATIVE_FAMILIES[@]}"; do
    link="$APP/Contents/Helpers/arsenal-worker-$family"
    [[ -L "$link" && "$(readlink "$link")" == "$NATIVE_EXE" ]] || die "arsenal-worker-$family is not a relative link to $NATIVE_EXE"
    workers+=("$link")
  done
  # Exactly one worker image: any other Mach-O in Helpers besides the MCP bridge and update helper would be a second MLX copy.
  others=()
  for file in "$APP"/Contents/Helpers/*; do
    [[ -f "$file" && ! -L "$file" && "$file" != "$native" && "$(basename "$file")" != arsenal-mcp && "$(basename "$file")" != ArsenalUpdateHelper ]] || continue
    file -b "$file" | grep -q Mach-O && others+=("$(basename "$file")")
  done
  (( ${#others[@]} == 0 )) || die "unexpected executables in Helpers: ${others[*]}"
  set +e; "$native" </dev/null >/dev/null 2>&1; code=$?; set -e
  (( code == 64 )) || die "$NATIVE_EXE started under its own name exited $code, expected 64 (usage)"
  echo "$NATIVE_EXE: the only worker image ($(stat -f %z "$native") bytes); links: ${NATIVE_FAMILIES[*]}; refuses its own name (exit 64)"
fi
if (( ${#workers[@]} > 0 )); then
  metallib="$APP/Contents/Helpers/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
  [[ -s "$metallib" ]] || die "Metal library missing: $metallib"
  [[ "$(head -c 4 "$metallib")" == MTLB ]] || die "$metallib is not a Metal library"
  size=$(stat -f %z "$metallib"); (( size > 1000000 )) || die "$metallib is only $size bytes"
  echo "metallib: present, MTLB, $size bytes"
  /usr/bin/python3 - "$tmp" "${workers[@]}" <<'PY'
import json, os, struct, subprocess, sys, time, select
tmp, workers = sys.argv[1], sys.argv[2:]

def fail(msg):
    print(f"smoke: {msg}", file=sys.stderr); sys.exit(1)

def read_frame(proc, deadline):
    def read_exact(n):
        data = b""
        while len(data) < n:
            left = deadline - time.time()
            if left <= 0: fail(f"{os.path.basename(proc.args[0])}: timed out waiting for a frame")
            ready, _, _ = select.select([proc.stdout], [], [], left)
            if not ready: continue
            chunk = os.read(proc.stdout.fileno(), n - len(data))
            if not chunk: fail(f"{os.path.basename(proc.args[0])}: stdout closed (exit {proc.poll()})")
            data += chunk
        return data
    (length,) = struct.unpack(">I", read_exact(4))
    if not 0 < length <= 16 * 1024 * 1024: fail("malformed frame length %d" % length)
    body = read_exact(length)
    return body[0], body[1:]

def lsappinfo_has(pid):
    out = subprocess.run(["/usr/bin/lsappinfo", "list"], capture_output=True, text=True).stdout
    return f"pid = {pid} " in out or f"pid = {pid}\n" in out

for path in workers:
    name = os.path.basename(path)
    family = name.removeprefix("arsenal-worker-")
    support = os.path.join(tmp, name)
    os.makedirs(support)
    env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "ARSENAL_SUPPORT_DIR": support, "ARSENAL_CACHE_DIR": support + "-cache"}
    proc = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    try:
        kind, payload = read_frame(proc, time.time() + 30)
        hello = json.loads(payload) if kind == 1 else fail(f"{name}: first frame is kind {kind}, not JSON")
        if hello.get("event") != "hello" or hello.get("proto") != 1: fail(f"{name}: bad handshake {hello}")
        # Kit models serve the kit runner protocol, whose hello names the kit (Workers/Kit).
        expected = "arsenal-kit" if family.startswith("kit-") else family
        if hello.get("worker") != expected: fail(f"{name}: handshake says worker={hello.get('worker')}")
        if hello.get("pid") != proc.pid: fail(f"{name}: handshake pid {hello.get('pid')} != child {proc.pid}")
        if lsappinfo_has(proc.pid): fail(f"{name}: registered as an app (lsappinfo)")
        request = json.dumps({"id": 1, "op": "ping"}).encode()
        proc.stdin.write(struct.pack(">I", len(request) + 1) + b"\x01" + request); proc.stdin.flush()
        deadline = time.time() + 20
        while True:
            kind, payload = read_frame(proc, deadline)
            reply = json.loads(payload) if kind == 1 else {}
            if reply.get("id") == 1: break
        if reply.get("ok") is not True: fail(f"{name}: ping failed: {reply}")
        proc.stdin.close()
        try: code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired: fail(f"{name}: ignored stdin EOF for 10 s")
        if code != 0: fail(f"{name}: exit {code} on stdin EOF")
        print(f"{name}: handshake ok (proto 1, pid {proc.pid}), headless, ping ok, exit 0 on stdin EOF")
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()
PY
else
  echo "workers: none in this bundle"
fi

# Licence files travel with the app: the MIT text, NOTICE and the third-party notices.
for file in LICENSE NOTICE THIRD_PARTY_NOTICES.txt; do [[ -s "$APP/Contents/Resources/$file" ]] || die "Contents/Resources/$file is missing or empty"; done
[[ "$(shasum -a 256 "$APP/Contents/Resources/LICENSE" | cut -d' ' -f1)" == 67f9e8393e7f1c5afc0824047c8e7c3b3ac5666309916a75a41dd647a80fea56 ]] || die "bundled LICENSE is not the Arsenal MIT text"
grep -q 'MIT License' "$APP/Contents/Resources/NOTICE" || die "bundled NOTICE does not state the licence"
echo "licence: MIT LICENSE, NOTICE, THIRD_PARTY_NOTICES.txt present"
for file in catalog-public-key catalog/catalog.json catalog/catalog.json.sig catalog/source.json; do [[ -s "$APP/Contents/Resources/$file" ]] || die "Contents/Resources/$file is missing (review B1: a fresh install needs the signed catalog)"; done
/usr/bin/python3 - "$APP/Contents/Resources/catalog/source.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1])) == {"source": "bundle"}, "default catalog source is not bundle"
PY
echo "catalog: signed seed pair, public key and default bundle source present"
/usr/bin/python3 "$PKG_ROOT/scripts/bundle-inventory.py" verify "$APP/Contents/Resources"
[[ -s "$APP/Contents/Resources/SKILL.md" ]] || die "SKILL.md is missing"
[[ ! -e "$APP/Contents/Resources/python" ]] || die "Python model runtime must not be bundled"
echo "runner support: SKILL.md present; native engines only"
"$APP/Contents/Helpers/arsenal-mcp" --help >/dev/null || die "arsenal-mcp --help failed from the bundle"
echo "arsenal-mcp: --help runs from the bundle"
for command in catalog runner-check; do
  "$APP/Contents/MacOS/Arsenal" "$command" --help >/dev/null || die "Arsenal $command --help failed from the bundle"
done
echo "folder tools: catalog and runner checker run from the bundle"
[[ ! -e "$APP/Contents/Helpers/uv" ]] || die "uv must not be bundled"
echo "uv: not bundled"

# The installed drainer is held before it can discover an engine or load weights.
/usr/bin/python3 - "$APP" "$tmp/stats" <<'PYTHON'
import os, pathlib, subprocess, sys
app, support = map(pathlib.Path, sys.argv[1:])
support.mkdir()
env = os.environ | {"ARSENAL_SUPPORT_DIR": str(support), "ARSENAL_CACHE_DIR": str(support / "cache")}
result = subprocess.run(
    [str(app / "Contents/MacOS/Arsenal"), "stats", "run", "--support", str(support),
     "--models", str(app / "Contents/Resources/models")],
    env=env, capture_output=True, text=True, timeout=10)
assert result.returncode != 0 and "requires" in result.stderr and "--go" in result.stderr, result.stderr
assert not (support / "status.json").exists() and not (support / "config.json").exists(), "drainer started app state"
print("Arsenal stats: headless entry; real queue held without --go")
PYTHON
