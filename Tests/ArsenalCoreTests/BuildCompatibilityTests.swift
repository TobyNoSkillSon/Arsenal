// SPDX-License-Identifier: MIT
import Foundation
import Testing

private func compatibilityShell(_ script: String, arguments: [String], environment: [String: String]) throws -> (Int32, String) {
    let child = Process(), pipe = Pipe()
    child.executableURL = URL(fileURLWithPath: "/usr/bin/perl")
    child.arguments = ["-e", "alarm 10; exec @ARGV", "/bin/bash", "-c", script, "test"] + arguments
    child.environment = environment; child.standardOutput = pipe; child.standardError = pipe
    try child.run(); let output = pipe.fileHandleForReading.readDataToEndOfFile(); child.waitUntilExit()
    return (child.terminationStatus, String(decoding: output, as: UTF8.self))
}
@Test func v3BorrowGuardAcceptsWeakRejectsStrongAndFailsClosed() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let directory = URL(fileURLWithPath: "/tmp/arsenal-v3-borrow-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let mock = directory.appendingPathComponent("nm")
    try Data(
        """
        #!/bin/sh
        case "$3" in
          weak) echo '(undefined) weak external _swift_initBorrow (from libswiftCore)' ;;
          strong) echo '(undefined) external _swift_initBorrow (from libswiftCore)' ;;
          end) echo '(undefined) external _swift_endBorrow (from libswiftCore)' ;;
          unreadable) exit 1 ;;
        esac
        """.utf8
    ).write(to: mock)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: mock.path)
    let environment = ["PATH": directory.path + ":/usr/bin:/bin", "HOME": directory.path]
    let script = "set -euo pipefail; source \"$1\"; reject_borrow_symbols \"$2\""
    #expect(try compatibilityShell(script, arguments: [root.appendingPathComponent("scripts/pkg-lib.sh").path, "weak"], environment: environment).0 == 0)
    for symbol in ["strong", "end", "unreadable"] {
        #expect(try compatibilityShell(script, arguments: [root.appendingPathComponent("scripts/pkg-lib.sh").path, symbol], environment: environment).0 != 0)
    }
}
@Test func v3NativeLinkLeavesSwift63UnchangedAndRejectsUnknownCompiler() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let build = try String(contentsOf: root.appendingPathComponent("scripts/build.sh"), encoding: .utf8)
    let start = try #require(build.range(of: "  link=()")),
        end = try #require(build.range(of: "  clt_swift build --package-path \"$NATIVE_DIR\"", range: start.upperBound..<build.endIndex))
    let directory = URL(fileURLWithPath: "/tmp/arsenal-v3-compiler-" + UUID().uuidString)
    let bin = directory.appendingPathComponent("usr/bin")
    try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let clang = bin.appendingPathComponent("clang")
    try Data(
        """
        #!/bin/sh
        cat > "$CLANG_INPUT"
        while [ "$#" -gt 0 ]; do
          if [ "$1" = "-o" ]; then printf 'fixture object' > "$2"; break; fi
          shift
        done
        """.utf8
    ).write(to: clang)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: clang.path)
    let input = directory.appendingPathComponent("shim.c")
    let script = """
        set -euo pipefail
        die() { echo "$*"; exit 1; }
        clt_swift() { printf '%s\\n' "$TEST_SWIFT_VERSION"; }
        CLT="$1"; NATIVE_DIR="$1"
        cd "$2"
        \(build[start.lowerBound..<end.lowerBound])
        printf '%s' "${link[@]+"${link[@]}"}"
        """
    var environment = ["PATH": "/usr/bin:/bin", "HOME": directory.path, "CLANG_INPUT": input.path, "TEST_SWIFT_VERSION": "Apple Swift version 6.3.3"]
    let old = try compatibilityShell(script, arguments: [directory.path, root.path], environment: environment)
    #expect(old.0 == 0 && old.1.isEmpty && !FileManager.default.fileExists(atPath: input.path))
    environment["TEST_SWIFT_VERSION"] = "Apple Swift version 6.4"
    let current = try compatibilityShell(script, arguments: [directory.path, root.path], environment: environment)
    #expect(current.0 == 0 && current.1.contains("-weak_reference_mismatches") && current.1.hasSuffix("weak"))
    #expect(try String(contentsOf: input, encoding: .utf8).contains("weak_import"))
    try FileManager.default.removeItem(at: input)
    let reused = try compatibilityShell(script, arguments: [directory.path, root.path], environment: environment)
    #expect(reused.0 == 0 && reused.1 == current.1 && !FileManager.default.fileExists(atPath: input.path))
    environment["TEST_SWIFT_VERSION"] = "unknown compiler"
    #expect(try compatibilityShell(script, arguments: [directory.path, root.path], environment: environment).0 != 0)
}

@Test func v3LockPathsStaySharedForMainWorktreeAndCleanClone() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let script = """
        exec /usr/bin/python3 - "$1" <<'PY'
        import pathlib, sys
        program = pathlib.Path(sys.argv[1]).read_text().rsplit('\\nmain()', 1)[0]
        base = '/tmp/vault/Projects/Arsenal'
        for root in [base, base + '/.worktrees/worker', base + '/.build/v3-candidate']:
            context = {'__file__': root + '/scripts/mlock'}
            exec(compile(program, 'mlock', 'exec'), context)
            assert context['MAIN'] == base
            assert context['VELLA'] == '/tmp/vault/Projects/Vella'
            assert context['HOLD'] == base + '/.gpu-hold'
        print('main, worktree, clean clone share locks and hold')
        PY
        """
    let result = try compatibilityShell(script, arguments: [root.appendingPathComponent("scripts/mlock").path], environment: ["PATH": "/usr/bin:/bin"])
    #expect(result.0 == 0 && result.1.contains("share locks and hold"))
}
