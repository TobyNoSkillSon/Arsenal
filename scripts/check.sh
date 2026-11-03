#!/bin/bash
# Check my change (the day loop): lint, build, then the complete suites the change affects in both packages
# (scripts/test.sh --changed). Fast, impact-selected; it never stands in for scripts/ready.sh before an install.
#   scripts/check.sh [--native] [extra scripts/test.sh arguments]
# Takes this worktree's work lock (a second run waits), writes a new log directory .build/logs/check-<time>-<id>/
# (run.log, per-suite logs, phases.tsv) and ends with each phase's wall time.
set -euo pipefail
# The physical path: mlock names the locked root by it, so a symlinked checkout cannot loop re-taking the lock.
cd "$(dirname "$0")/.." && cd -P .
if [[ -z "${ARSENAL_ENTRY_LOCKED:-}" && "${ARSENAL_WORK_LOCKED_ROOT:-}" != "$PWD" ]]; then
  exec env ARSENAL_ENTRY_LOCKED=1 nice -n 19 scripts/mlock work check -- "$PWD/scripts/check.sh" "$@"
fi
source scripts/dev-lib.sh
entry_log check
entry_jobs
timed check scripts/test.sh --changed "$@"
