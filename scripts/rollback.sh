#!/bin/bash
# SPDX-License-Identifier: MIT
# Restore the previous bundle AND its instruction snapshot. No weight or user-config changes.
# Internal install mode: --snapshot <previous-version>; writes the paired snapshot path to stdout.
# --check runs every restore check (signature, snapshot record, version, hashes) and changes nothing.
set -euo pipefail
source "$(dirname "$0")/pkg-lib.sh"
DEST="${ARSENAL_DESTINATION_APP:-$HOME/Applications/Arsenal.app}"
SUPPORT="$(profile_dir)"
if [[ "${1:-}" != --snapshot ]]; then
  [[ -d "$DEST.previous/Contents" ]] || die "no previous app at $DEST.previous"
  [[ -f "$DEST.previous.rollback-state" ]] || die "previous app has no paired instruction snapshot; nothing restored"
  codesign --verify --deep --strict "$DEST.previous" || die "previous app signature does not verify; nothing restored"
fi
if pgrep -f "$DEST/Contents/MacOS/Arsenal" >/dev/null; then die "Arsenal is running from $DEST; quit it before changing instructions"; fi
/usr/bin/python3 - "$DEST" "$SUPPORT" "${1:-restore}" "${2:-}" <<'PY'
import hashlib, json, os, pathlib, plistlib, shutil, sys, uuid
app, support = map(pathlib.Path, sys.argv[1:3])
mode, version = sys.argv[3:5]
# These are instructions and verification records only. Weights, envs, config and voices stay in place.
paths = ('catalog', 'models', 'state/installed-catalog')
support = support.resolve()
def copy_tree(source, dest):
    # Never follow a model-folder symlink into weights or another user directory.
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
        raise RuntimeError('instruction snapshot refuses symlinks: ' + str(source))
    shutil.copytree(source, dest)
def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p != root / 'snapshot.json'}
def snapshot(destination):
    destination.mkdir(parents=True)
    present = []
    for relative in paths:
        source = support / relative
        if source.exists():
            copy_tree(source, destination / relative)
            present.append(relative)
    return present
if mode == '--snapshot':
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_' for c in version):
        raise RuntimeError('invalid previous version')
    final = support / 'rollback' / (version + '-' + uuid.uuid4().hex)
    stage = final.with_name('.stage-' + final.name)
    try:
        present = snapshot(stage)
        record = {'support': str(support), 'version': version, 'paths': present, 'sha256': hashes(stage)}
        (stage / 'snapshot.json').write_text(json.dumps(record))
        stage.rename(final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(final)
else:
    paired = pathlib.Path(str(app) + '.previous.rollback-state')
    # The pointer names the snapshot folder. The snapshot is looked up in the profile folder given here.
    saved = support / 'rollback' / pathlib.Path(paired.read_text().strip()).name
    record = json.loads((saved / 'snapshot.json').read_text())
    with (pathlib.Path(str(app) + '.previous') / 'Contents/Info.plist').open('rb') as handle:
        previous_version = plistlib.load(handle)['CFBundleShortVersionString']
    if record['version'] != previous_version or set(record['paths']) - set(paths):
        raise RuntimeError('snapshot does not match previous app')
    if hashes(saved) != record['sha256']:
        raise RuntimeError('instruction snapshot checksums differ; nothing restored')
    if mode == '--check':
        sys.exit(0)
    forward = support / 'rollback' / ('rolled-forward-' + uuid.uuid4().hex)
    stage = forward / 'restore'
    stage.mkdir(parents=True)
    for relative in record['paths']:
        copy_tree(saved / relative, stage / relative)
    moved = []
    forward_app = pathlib.Path(str(app) + '.rolled-forward-' + forward.name.removeprefix('rolled-forward-'))
    app_moved = False
    try:
        for relative in paths:
            current = support / relative
            if current.exists():
                target = forward / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                current.rename(target)
            moved.append(relative)
            if relative in record['paths']:
                current.parent.mkdir(parents=True, exist_ok=True)
                (stage / relative).rename(current)
        # The app being replaced may already be gone.
        if app.exists():
            app.rename(forward_app)
            app_moved = True
        pathlib.Path(str(app) + '.previous').rename(app)
    except BaseException:
        if app_moved:
            forward_app.rename(app)
        for relative in reversed(moved):
            current = support / relative
            if current.exists(): shutil.rmtree(current)
            if (forward / relative).exists(): (forward / relative).rename(current)
        raise
    shutil.rmtree(stage)
    paired.unlink()
    print('restored ' + str(app) + ' and instructions for ' + previous_version)
    print('newer app retained at ' + str(forward_app) + '; newer instructions at ' + str(forward))
PY
