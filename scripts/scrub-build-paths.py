#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Keep the build machine's paths out of shipped binaries (scripts/build.sh, release builds).

  scrub-build-paths.py scrub ROOT APP     rewrite ROOT inside every Mach-O file of APP, before strip and signing
  scrub-build-paths.py check ROOT HOME APP  list every Mach-O or metallib in APP that still names ROOT or HOME; exit 1 if any

Why a scrub as well as prefix maps: build.sh passes -file-prefix-map / -ffile-prefix-map / -oso_prefix, which cover
debug info, C/C++ __FILE__ and the linker's debug map, but no compiler flag reaches two kinds of Swift string literal:
`#filePath` (a development-checkout lookup) and the build-directory fallback that SwiftPM generates into every
`Bundle.module` accessor. Both are dead paths on any other Mac. Each occurrence of ROOT (followed by "/" or the end of
the string) is replaced in place by a neutral path of the same length, "/arsenal" padded with "/." segments, so string
lengths compiled into the code stay valid and the path still means the same relative location.
"""
import os, pathlib, re, sys

MACHO_MAGICS = {b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xfe\xed\xfa\xcf', b'\xbe\xba\xfe\xca'}


def neutral(length):
    base = '/arsenal'
    if length < len(base) + 1:
        sys.exit(f'scrub: build root of {length} bytes is too short to replace')
    rest = length - len(base)
    return (base + '/' * (rest % 2) + '/.' * (rest // 2)).encode()


def binaries(app, metallibs=False):
    for directory, _, names in os.walk(app):
        for name in names:
            path = pathlib.Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            if metallibs and name.endswith('.metallib'):
                yield path; continue
            with open(path, 'rb') as file:
                if file.read(4) in MACHO_MAGICS:
                    yield path


def scrub(root, app):
    root_bytes = root.rstrip('/').encode()
    replacement = neutral(len(root_bytes))
    pattern = re.compile(re.escape(root_bytes) + rb'(?=[/\0])')
    total = 0
    for path in binaries(app):
        data = path.read_bytes()
        new, count = pattern.subn(replacement, data)
        if count:
            assert len(new) == len(data)
            mode = path.stat().st_mode
            path.chmod(mode | 0o200)
            path.write_bytes(new)
            path.chmod(mode)
            print(f'scrub: {path.relative_to(app)}: {count} build-root path(s) neutralised')
            total += count
    print(f'scrub: {total} build-root path(s) neutralised in {app}')


def check(root, home, app):
    needles = [n.rstrip('/').encode() for n in (root, home) if n.rstrip('/')]
    hits = 0
    for path in binaries(app, metallibs=True):
        data, seen = path.read_bytes(), set()
        for needle in needles:
            for match in re.finditer(re.escape(needle), data):
                start = data.rfind(b'\0', 0, match.start()) + 1
                if start in seen:
                    continue
                seen.add(start)
                end = data.find(b'\0', match.end())
                text = data[start:end if end != -1 else match.end() + 80][:200].decode('utf-8', 'replace')
                print(f'path in binary: {path.relative_to(app)}: {text}')
                hits += 1
    if hits:
        sys.exit(f'check: {hits} build-machine path(s) remain in shipped binaries')
    print(f'check: no build root or home folder in the Mach-O files and metallibs of {app}')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'scrub':
        scrub(sys.argv[2], pathlib.Path(sys.argv[3]))
    elif len(sys.argv) == 5 and sys.argv[1] == 'check':
        check(sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4]))
    else:
        sys.exit(__doc__)
