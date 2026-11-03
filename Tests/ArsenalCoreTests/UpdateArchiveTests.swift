// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct UpdateArchiveTests {
    func fixture(extra: [(String, String)] = []) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-archive-\(UUID().uuidString)")
        var complete = false
        defer {
            if !complete { unregisterFixtureBundles(in: root); try? FileManager.default.removeItem(at: root) }
        }
        let app = root.appendingPathComponent("Arsenal.app")
        let helpers = app.appendingPathComponent("Contents/Helpers")
        let licences = app.appendingPathComponent("Contents/Resources/Licenses")
        try FileManager.default.createDirectory(at: helpers, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: licences.appendingPathComponent("native"), withIntermediateDirectories: true)
        for path in ["Contents/MacOS/Arsenal", "Contents/Helpers/ArsenalUpdateHelper", "Contents/Helpers/arsenal-worker-native"] {
            let file = app.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("fixture executable".utf8).write(to: file)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: file.path)
        }
        let info: [String: String] = ["CFBundleIdentifier": "dev.arsenal.archive-fixture", "CFBundleExecutable": "Arsenal", "CFBundleShortVersionString": "3.2.0"]
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0).write(to: app.appendingPathComponent("Contents/Info.plist"))
        try Data("fixture licence".utf8).write(to: licences.appendingPathComponent("LICENSE-A"))
        // Read the assembly source of truth so new families are automatically exercised.
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let families = try UpdateVerification.tool(
            "/bin/bash", ["-c", "source \"$1\"; printf '%s\\n' \"${NATIVE_FAMILIES[@]}\"", "fixture", repo.appendingPathComponent("scripts/pkg-lib.sh").path])
        #expect(families.split(whereSeparator: \.isNewline).contains("d1"))
        #expect(families.split(whereSeparator: \.isNewline).contains("flux"))
        for family in families.split(whereSeparator: \.isNewline) {
            try FileManager.default.createSymbolicLink(atPath: helpers.appendingPathComponent("arsenal-worker-" + family).path, withDestinationPath: "arsenal-worker-native")
        }
        let links = [("Contents/Resources/Licenses/LICENSE-B", "LICENSE-A"), ("Contents/Resources/Licenses/native/dependency-LICENSE", "../LICENSE-A")] + extra
        for (name, target) in links {
            let link = app.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: target)
        }
        let archive = root.appendingPathComponent("fixture.zip")
        // docs/UPDATES.md's exact release ZIP command, applied to fake products.
        try UpdateVerification.tool("/usr/bin/ditto", ["-c", "-k", "--keepParent", "--norsrc", app.path, archive.path])
        complete = true
        return archive
    }

    @Test func documentedReleaseArchiveAllowsAssemblyLinks() throws {
        let archive = try fixture()
        defer { unregisterFixtureBundles(in: archive.deletingLastPathComponent()); try? FileManager.default.removeItem(at: archive.deletingLastPathComponent()) }
        try UpdateVerification.archiveEntries(archive)
        let extracted = archive.deletingLastPathComponent().appendingPathComponent("extracted")
        try UpdateVerification.tool("/usr/bin/ditto", ["-x", "-k", archive.path, extracted.path])
        let licences = extracted.appendingPathComponent("Arsenal.app/Contents/Resources/Licenses")
        #expect(try String(contentsOf: licences.appendingPathComponent("native/dependency-LICENSE"), encoding: .utf8) == "fixture licence")
    }

    @Test func statsSubcommandLayoutPassesUnchanged31Verifier() throws {
        let archive = try fixture()
        defer { unregisterFixtureBundles(in: archive.deletingLastPathComponent()); try? FileManager.default.removeItem(at: archive.deletingLastPathComponent()) }
        try UpdateVerification.archiveEntries(archive)
        let extracted = archive.deletingLastPathComponent().appendingPathComponent("stats-extracted")
        try UpdateVerification.tool("/usr/bin/ditto", ["-x", "-k", archive.path, extracted.path])
        let bundle = extracted.appendingPathComponent("Arsenal.app"), app = extracted.appendingPathComponent("verification-fixture")
        #expect(throws: UpdateFailure.self) { try UpdateVerification.bundle(bundle, version: "3.2.0") }
        // Exercise the production metadata validator outside a .app bundle; the archive itself has a fixture ID.
        try FileManager.default.moveItem(at: bundle, to: app)
        let plist = app.appendingPathComponent("Contents/Info.plist")
        var info = try #require(PropertyListSerialization.propertyList(from: Data(contentsOf: plist), format: nil) as? [String: String])
        info["CFBundleIdentifier"] = "dev.arsenal.app"
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0).write(to: plist)
        try UpdateVerification.bundle(app, version: "3.2.0")
        #expect(!FileManager.default.fileExists(atPath: app.appendingPathComponent("Contents/Helpers/arsenal-stats").path))
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let assembly = try String(contentsOf: repo.appendingPathComponent("scripts/assemble.sh"), encoding: .utf8)
        #expect(!assembly.contains("Contents/Helpers/arsenal-stats"))
    }

    @Test(arguments: [
        ("Contents/Resources/Licenses/escape", "../../../../../outside"),
        ("Contents/Resources/Licenses/absolute", "/tmp/outside"),
        ("Contents/Resources/Licenses/missing", "absent"),
        ("Contents/Resources/Licenses/cycle", "cycle"),
        ("Contents/Resources/Licenses/chain", "LICENSE-B"),
        ("Contents/Resources/Licenses/wrong-target", "../../Helpers/arsenal-worker-native"),
        ("Contents/Helpers/unexpected", "arsenal-worker-native"),
        ("Contents/Helpers/arsenal-stats", "../MacOS/Arsenal"),
        ("Contents/Helpers/arsenal-worker-escape", "../../../../outside")
    ])
    func unsafeOrUnexpectedLinksAreRejected(name: String, target: String) throws {
        let archive = try fixture(extra: [(name, target)])
        defer { unregisterFixtureBundles(in: archive.deletingLastPathComponent()); try? FileManager.default.removeItem(at: archive.deletingLastPathComponent()) }
        #expect(throws: UpdateFailure.self) { try UpdateVerification.archiveEntries(archive) }
    }
}
