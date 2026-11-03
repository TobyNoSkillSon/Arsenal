#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Copy/check only signed model and shared-kind instructions, never weights."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def inventory(root):
    catalog = json.loads((root / ("Resources/catalog/catalog.json" if (root / "Resources/catalog/catalog.json").is_file() else "catalog/catalog.json")).read_text())
    files = {}
    for model in catalog["models"]:
        for file in model["files"]:
            files[f'models/{model["manifest"]["id"]}/{file["path"]}'] = file
    for file in catalog["kind_files"]:
        files[f'kinds/{file["path"]}'] = file
    return catalog, files


def verify(root, exact=True):
    """exact: the tree holds the signed files and nothing else (a bundle). A source checkout may also hold unshipped
    folders (models the catalog excludes, such as LLM chat); every signed file must still be present and match."""
    catalog, files = inventory(root)
    present = set()
    for tree in ("models", "kinds"):
        base = root / ("Resources/kinds" if tree == "kinds" and (root / "Resources/kinds").is_dir() else tree)
        assert base.is_dir() and not base.is_symlink(), f"missing/linked {base}"
        for path in base.rglob("*"):
            assert not path.is_symlink(), f"symlink: {path}"
            if path.is_file():
                present.add(tree + "/" + path.relative_to(base).as_posix())
    assert set(files) <= present and (not exact or present == set(files)), \
        f"signed inventory mismatch: missing={set(files)-present}, extra={present-set(files) if exact else set()}"
    for relative, file in files.items():
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts, f"unsafe path: {relative}"
        source = root / ("Resources" if path.parts[0] == "kinds" and (root / "Resources/kinds").is_dir() else "") / path
        data = source.read_bytes()
        assert len(data) == file["bytes"], f"size mismatch: {relative}"
        assert hashlib.sha256(data).hexdigest() == file["sha256"].lower(), f"hash mismatch: {relative}"
    return catalog, files


def main():
    command, source = sys.argv[1], Path(sys.argv[2])
    catalog, files = verify(source, exact=command != "copy")
    if command == "copy":
        destination = Path(sys.argv[3])
        for relative in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / ("Resources" if relative.startswith("kinds/") and (source / "Resources/kinds").is_dir() else "") / relative, target)
        verify(destination)
    elif command != "verify":
        raise ValueError("usage: bundle-inventory.py copy ROOT RESOURCES | verify RESOURCES")
    print(f'catalog inventory: sequence {catalog["sequence"]}, {len(catalog["models"])} model folders, '
          f'{len(catalog["kind_files"])} kind files, {len(files)} files verified (including kernels/logs/results)')


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, ValueError, OSError) as error:
        sys.exit(f"catalog inventory: {error}")
