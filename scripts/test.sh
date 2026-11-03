#!/bin/bash
# Complete suites: scripts/test.sh. Affected complete suites: scripts/test.sh --changed.
# Both run the app package's suites, then the native graph's (Workers/Package.swift: every family's tests, one MLX).
# Native suites alone: scripts/test.sh --native [--changed | --filter <suite>].
# Heavy tests (model inference on the GPU, tagged `.heavy`) never run by default: scripts/test.sh --heavy [--filter <suite>]
# runs only those, one test process at a time, at nice 19. Heavy and default runs share no state: the default unsets ARSENAL_HEAVY_TESTS.
set -euo pipefail
cd "$(dirname "$0")/.."
heavy=false; level=10
for arg in "$@"; do if [[ "$arg" == --heavy ]]; then heavy=true; level=19; fi; done
if [[ -z "${ARSENAL_DEV_LOCKED:-}" && "${ARSENAL_WORK_LOCKED_ROOT:-}" != "$PWD" ]]; then
  exec env ARSENAL_DEV_LOCKED=1 nice -n "$level" scripts/mlock work tests -- "$PWD/scripts/test.sh" "$@"
fi
source scripts/dev-lib.sh
# Lint runs beside the compile (it reads sources, the compiler writes .build); no test runs before it has passed.
mkdir -p "$PWD/.build"
lint_log="$(mktemp "$PWD/.build/lint-run.XXXXXX")"
# Background jobs get their own process group (set -m), so an early exit stops every compiler and test process they
# started, and waits for them, before the work lock is released.
set -m
timed lint scripts/lint.sh > "$lint_log" 2>&1 &
lint_pid=$!
set +m
background=("$lint_pid")
stop_background() {
  local pid
  for pid in ${background[@]+"${background[@]}"}; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in ${background[@]+"${background[@]}"}; do wait "$pid" 2>/dev/null || true; done
  rm -f "$lint_log"
}
trap stop_background EXIT
lint_wait() {
  [[ -n "$lint_pid" ]] || return 0
  local status=0
  wait "$lint_pid" || status=$?
  lint_pid=""
  cat "$lint_log"
  (( status == 0 )) || { echo 'test: lint failed' >&2; exit "$status"; }
}
changed=(); args=(); native=false
for arg in "$@"; do
  if [[ "$arg" == --changed ]]; then changed=(--changed); elif [[ "$arg" == --native ]]; then native=true; elif [[ "$arg" == --heavy ]]; then :; else args+=("$arg"); fi
done
if $heavy; then
  if (( ${#changed[@]} > 0 )); then echo 'test: --heavy does not combine with --changed' >&2; exit 1; fi
  # Heavy tests are native-only; one at a time (width 1, one target process) so one model owns the GPU.
  native=true
  export ARSENAL_HEAVY_TESTS=1 ARSENAL_TEST_WIDTH=1 ARSENAL_TEST_TARGET_JOBS=1
else
  unset ARSENAL_HEAVY_TESTS
fi
root_package=true; custom=false; inspect=false; skip_build=false; scratch="$PWD/.build"; configuration=debug; next=""
for arg in ${args[@]+"${args[@]}"}; do
  if [[ -n "$next" ]]; then
    case "$next" in scratch) scratch="$arg" ;; configuration) configuration="$arg" ;; esac
    next=""; continue
  fi
  next=""
  case "$arg" in
    --help|-h|list|--list-tests) inspect=true ;;
    --skip-build) skip_build=true ;;
  esac
  case "$arg" in
    --package-path) root_package=false; next=ignored ;; --package-path=*) root_package=false ;;
    --filter|--skip) next=ignored ;;
    --scratch-path) next=scratch ;; --scratch-path=*) scratch="${arg#*=}" ;;
    -c|--configuration) next=configuration ;; --configuration=*) configuration="${arg#*=}" ;;
    --skip-build|--list-tests|list|--help|-h|--parallel|--no-parallel|--maximum-repetitions|--maximum-repetitions=*|--repeat-until|--repeat-until=*|--xunit-output|--xunit-output=*|--enable-code-coverage|--enable-xctest|--disable-xctest) custom=true ;;
  esac
done
if $root_package; then
  case "$scratch" in /*) ;; *) scratch="$PWD/$scratch" ;; esac
  export ARSENAL_TEST_STUB_WORKER="${ARSENAL_TEST_STUB_WORKER:-$scratch/$configuration/arsenal-stub-worker}"
  export ARSENAL_TEST_SERVER="${ARSENAL_TEST_SERVER:-$scratch/$configuration/arsenal-testserver}"
  # Explicit skip-build consumes existing output unchanged; other commands validate local ABI consumers.
  if ! $skip_build && ! $inspect; then /usr/bin/python3 scripts/incremental.py prepare "$scratch" "$configuration"; fi
fi
build_args=(); test_args=(); filter_value=false; has_filter=false
for arg in ${args[@]+"${args[@]}"}; do
  if $filter_value; then test_args+=("$arg"); filter_value=false; continue; fi
  case "$arg" in
    --filter|--skip) test_args+=("$arg"); filter_value=true; [[ "$arg" != --filter ]] || has_filter=true ;;
    --filter=*|--skip=*) test_args+=("${arg%%=*}" "${arg#*=}"); [[ "$arg" != --filter=* ]] || has_filter=true ;;
    *) build_args+=("$arg") ;;
  esac
done
$filter_value && { echo 'test: filter needs a pattern' >&2; exit 1; }

# The native graph: build it (worker, lab tools, every family's tests) with its dev helpers, then run its suites in
# isolated per-target processes. Arguments: [--changed] [--filter/--skip pairs].
native_suites() {
  local selection=() targets
  if [[ "${1:-}" == --changed ]]; then
    shift
    targets="$(/usr/bin/python3 scripts/test-targets.py --native --affected)"
    if [[ -z "$targets" ]]; then echo 'Tests: no native suite affected'; return 0; fi
    selection=(--changed)
  fi
  (( $# == 0 )) || selection=(--filtered)
  native_debug_build
  export ARSENAL_TEST_METALLIB="${ARSENAL_TEST_METALLIB:-$NATIVE_HELPERS/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib}"
  export ARSENAL_MLX_METALLIB="${ARSENAL_MLX_METALLIB:-$ARSENAL_TEST_METALLIB}"
  export ARSENAL_TEST_NATIVE_WORKER="${ARSENAL_TEST_NATIVE_WORKER:-$NATIVE_HELPERS/$NATIVE_EXE}"
  timed native-tests /usr/bin/python3 scripts/test-targets.py --native ${selection[@]+"${selection[@]}"} -- \
    "$CLT/usr/libexec/swift/pm/swiftpm-testing-helper" --test-bundle-path "$(native_test_binary)" \
    --testing-library swift-testing --experimental-maximum-parallelization-width "${ARSENAL_TEST_WIDTH:-8}" "$@"
}

if $native; then
  if ! $root_package || $custom || $inspect || (( ${#build_args[@]} > 0 )); then
    echo 'test: --native accepts only --changed, --filter and --skip' >&2; exit 1
  fi
  # Native suites run the app's headless validator (.build/debug/Arsenal) on their fixtures.
  timed app-build swift_debug build --product Arsenal --jobs "${ARSENAL_BUILD_JOBS:-2}"
  /usr/bin/python3 scripts/incremental.py record "$scratch" "$configuration"
  lint_wait
  native_suites ${changed[@]+"${changed[@]}"} ${test_args[@]+"${test_args[@]}"}
  exit 0
fi
if ! $root_package || $custom; then
  lint_wait
  inspection=(); if $inspect; then inspection=(--inspect); fi
  /usr/bin/python3 scripts/test-run-checked.py ${inspection[@]+"${inspection[@]}"} -- env DEVELOPER_DIR="$CLT" "$CLT/usr/bin/swift" test \
    --build-system native -Xswiftc -F -Xswiftc "$FRAMEWORKS" -Xlinker -F -Xlinker "$FRAMEWORKS" \
    -Xlinker -rpath -Xlinker "$FRAMEWORKS" -Xlinker -rpath -Xlinker "$CLT/Library/Developer/usr/lib" \
    --jobs "${ARSENAL_BUILD_JOBS:-2}" ${args[@]+"${args[@]}"}
  if $root_package && ! $skip_build && ! $inspect; then /usr/bin/python3 scripts/incremental.py record "$scratch" "$configuration"; fi
else
  timed app-build swift_debug test list --jobs "${ARSENAL_BUILD_JOBS:-2}" ${build_args[@]+"${build_args[@]}"} > "$PWD/.build/test-list.txt"
  /usr/bin/python3 scripts/incremental.py record "$scratch" "$configuration"
  lint_wait
  test_binary=$(/usr/bin/python3 - "$scratch" "$configuration" <<'PYBINARY'
import pathlib, sys
from scripts.incremental import package_graph
metadata = package_graph(pathlib.Path.cwd())
name = metadata['package_name'] + 'PackageTests'
print(pathlib.Path(sys.argv[1]) / sys.argv[2] / (name + '.xctest') / 'Contents/MacOS' / name)
PYBINARY
)
  # SwiftPM keeps its build lock during test execution. Its own helper runs the same compiled tests
  # without another build session, so independent target processes can actually overlap.
  selection=()
  if (( ${#changed[@]} > 0 )); then selection=(--changed); fi
  if $has_filter; then selection=(--filtered); fi
  # A filtered run names app-package suites; complete and affected runs also run the native graph. Its build (the one
  # compile in flight, 2 jobs) and short suites run beside the app suites; ARSENAL_PREBUILD_RELEASE=1 (scripts/ready.sh)
  # then compiles the release configuration there too. Compiles stay one at a time; only test execution overlaps them.
  native_pid=""
  if ! $has_filter; then
    native_log="${ARSENAL_TEST_LOG_DIR:-$PWD/.build/test-results}/native-build.log"
    mkdir -p "$(dirname "$native_log")"
    set -m
    (
      set +m
      native_suites ${changed[@]+"${changed[@]}"}
      if [[ "${ARSENAL_PREBUILD_RELEASE:-0}" == 1 ]]; then timed release-compile env ARSENAL_PKG_COMPILE_ONLY=1 scripts/build.sh; fi
    ) > "$native_log" 2>&1 &
    native_pid=$!
    set +m
    background+=("$native_pid")
  fi
  app_status=0
  timed app-tests /usr/bin/python3 scripts/test-targets.py ${selection[@]+"${selection[@]}"} -- \
    "$CLT/usr/libexec/swift/pm/swiftpm-testing-helper" --test-bundle-path "$test_binary" \
    --testing-library swift-testing --experimental-maximum-parallelization-width "${ARSENAL_TEST_WIDTH:-8}" ${test_args[@]+"${test_args[@]}"} \
    || app_status=$?
  native_status=0
  if [[ -n "$native_pid" ]]; then
    wait "$native_pid" || native_status=$?
    echo "--- native graph (built and tested beside the app suites; log $native_log)"
    cat "$native_log"
  fi
  (( app_status == 0 )) || { echo 'test: app-package suites failed' >&2; exit "$app_status"; }
  (( native_status == 0 )) || { echo 'test: native graph build or suites failed' >&2; exit "$native_status"; }
fi
