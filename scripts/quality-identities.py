#!/usr/bin/env python3
"""Build receipt for quality gates, scoped to each native worker's actual source closure."""
import hashlib, json, os, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON = ['Workers/Transport', 'Workers/Kernels', 'Packages/ArsenalWire', 'Packages/ModelPrecision', 'Workers/Package.resolved']
FAMILIES = {
    'voices': ['Workers/Voices'],
    'sound': ['Workers/Sound', 'Workers/Voices/Sources/VireoCore', 'Workers/Voices/Sources/SmallMGEMM'],
    'decisions': ['Workers/Decisions'],
    'd1': ['Workers/D1', 'models/d1-3b/runners/swift'],
    'elephant': ['Workers/Elephant', 'models/elephant-1/runners/swift', 'Workers/IntegrationKit'],
    'flux': ['Workers/Flux', 'models/flux2-klein-4b/runners/swift'],
    'chat': ['Workers/Chat'],
    'language': ['Workers/Language', 'Workers/IntegrationKit'],
    'embed': ['Workers/Embed', 'Workers/IntegrationKit'],
    'audio': ['Workers/Audio', 'Workers/IntegrationKit'],
    'music': ['Workers/Music', 'Workers/IntegrationKit'],
    'restoration': ['Workers/Restoration', 'Workers/IntegrationKit'],
    'perception': ['Workers/Perception', 'Workers/IntegrationKit'],
    'geometry': ['Workers/Geometry', 'Workers/IntegrationKit'],
    'birefnet': ['Workers/BiRefNetEngine', 'models/birefnet/runners/swift'],
    'qwenimage': ['Workers/QwenImage', 'Workers/IntegrationKit'],
}
def kit_families(root=ROOT):
    """Kit models (Workers/Kit): family kit-<id> is its own module plus the kit package manifest."""
    families = {}
    for entry in sorted((root / 'Workers/Kit/Sources').glob('*/Entry.swift')):
        match = re.search(r'^    public static let modelID = "(.+)"$', entry.read_text(), re.M)
        if match:
            families['kit-' + match.group(1)] = [str(entry.parent.relative_to(root)), 'Workers/Kit/Package.swift']
    return families
def identities(root=ROOT):
    result = {}
    for family, roots in {**FAMILIES, **kit_families(root)}.items():
        digest = hashlib.sha256()
        files = set()
        for relative in COMMON + roots:
            path = root / relative
            candidates = [path] if path.is_file() else []
            if path.is_dir():
                for directory, dirs, names in os.walk(path):
                    dirs[:] = [name for name in dirs if not name.startswith('.') and name != 'Tests']
                    candidates.extend(pathlib.Path(directory) / name for name in names)
            for file in candidates:
                if file.is_file() and file.suffix in ['.swift', '.metal', '.h', '.cpp', '.c', '.resolved', '.patch']:
                    files.add(file)
        for file in sorted(files):
            digest.update(str(file.relative_to(root)).encode()); digest.update(b'\0'); digest.update(file.read_bytes()); digest.update(b'\0')
        result[family] = digest.hexdigest()
    return result
if __name__ == '__main__':
    pathlib.Path(sys.argv[1]).write_text(json.dumps(identities(), sort_keys=True) + '\n')
