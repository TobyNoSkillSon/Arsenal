// SPDX-License-Identifier: MIT
import Foundation
import Testing

/// The add-a-model kit ships inside the signed app (Contents/Resources/kit). Running it must leave that folder unchanged:
/// a __pycache__ beside folder_check.py (Homebrew's python3 writes one; Apple's keeps it elsewhere) breaks the app's seal,
/// and the updater then refuses every later version (fresh review, 2 Nov).
@Suite struct BundledKitTests {
    private static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    @Test func runningTheKitWritesNothingBesideIt() throws {
        let fm = FileManager.default
        let copy = fm.temporaryDirectory.appendingPathComponent("arsenal-bundled-kit-" + UUID().uuidString)
        defer { try? fm.removeItem(at: copy) }
        try fm.createDirectory(at: copy, withIntermediateDirectories: true)
        for name in ["new-model.py", "folder_check.py"] {
            try fm.copyItem(at: Self.repo.appendingPathComponent("kit/" + name), to: copy.appendingPathComponent(name))
        }
        // Behave like a python3 that caches beside the source (no pycache prefix), then run the kit's help.
        let script =
            "import runpy, sys; sys.pycache_prefix = None; sys.argv = ['new-model.py', '--help']\n"
            + "try:\n    runpy.run_path(sys.argv[0], run_name='__main__')\nexcept SystemExit:\n    pass\n"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", script]
        process.currentDirectoryURL = copy
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONDONTWRITEBYTECODE"] = nil; environment["PYTHONPYCACHEPREFIX"] = nil
        process.environment = environment
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        try process.run(); process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        #expect(try fm.contentsOfDirectory(atPath: copy.path).sorted() == ["folder_check.py", "new-model.py"])
    }
}
