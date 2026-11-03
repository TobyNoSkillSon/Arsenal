#!/bin/bash
# Ready to install: every suite of both packages (scripts/test.sh), then the strict signed candidate in dist/
# (ARSENAL_STRICT=1 scripts/build.sh: assemble, sign, smokes). Installs nothing; scripts/install-for-review.sh does that.
#   <with-release-keychain.sh> scripts/ready.sh
# The release compile runs beside the app suites (after the debug compiles, one compile at a time), so the signed build
# finds its compile done; nothing is assembled or published unless every suite passed.
# Takes this worktree's work lock (a second run waits), writes a new log directory .build/logs/ready-<time>-<id>/
# (run.log, per-suite logs, phases.tsv) and ends with each phase's wall time.
set -euo pipefail
# The physical path: mlock names the locked root by it, so a symlinked checkout cannot loop re-taking the lock.
cd "$(dirname "$0")/.." && cd -P .
if [[ -z "${ARSENAL_ENTRY_LOCKED:-}" && "${ARSENAL_WORK_LOCKED_ROOT:-}" != "$PWD" ]]; then
  exec env ARSENAL_ENTRY_LOCKED=1 nice -n 19 scripts/mlock work ready -- "$PWD/scripts/ready.sh" "$@"
fi
(( $# == 0 )) || { echo 'ready: takes no arguments; it always runs every suite' >&2; exit 2; }
source scripts/dev-lib.sh
entry_log ready
entry_jobs
# Fail before the suites, not after them, when the signing identity is unavailable.
scripts/check-identity.sh
timed all-suites env ARSENAL_PREBUILD_RELEASE=1 scripts/test.sh
timed signed-bundle env ARSENAL_STRICT=1 scripts/build.sh
