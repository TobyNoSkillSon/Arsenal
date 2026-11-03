// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct UpdateTests {
    @Test func semanticOrdering() throws {
        let ordered = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "1.1.0", "2.0.0", "10.0.0"]
        for (old, new) in zip(ordered, ordered.dropFirst()) {
            let oldVersion = try #require(UpdateVersion(old)), newVersion = try #require(UpdateVersion(new))
            #expect(oldVersion < newVersion)
        }
        #expect(UpdateVersion("v3.0.0+build.7") == UpdateVersion("3.0.0+build.8"))
        for invalid in ["1", "1.2", "01.2.3", "1.2.3-01", "1.2.3-a..b", "1.2.3+", "1.2.3+..", "1.2.3.4", " 1.2.3", "../3.0.0"] {
            #expect(UpdateVersion(invalid) == nil)
        }
        #expect(UpdateVersion("1.0.0-9999999999999999999999999")! > UpdateVersion("1.0.0-999999999999999999999999")!)
    }
    @Test func checkCadence() {
        let now = Date(timeIntervalSince1970: 1000000)
        #expect(UpdateController.checkIsDue(last: nil, now: now))
        #expect(!UpdateController.checkIsDue(last: now.addingTimeInterval(-259199), now: now))
        #expect(UpdateController.checkIsDue(last: now.addingTimeInterval(-259200), now: now))
        #expect(!UpdateController.checkIsDue(last: now.addingTimeInterval(1), now: now))
    }
    @Test func hashesMustBeUniqueAndNamed() throws {
        let hash = String(repeating: "a", count: 64)
        #expect(try UpdateVerification.expectedHash("\(hash)  Arsenal-3.0.0-arm64.zip\n", name: "Arsenal-3.0.0-arm64.zip") == hash)
        #expect(throws: UpdateFailure.self) { try UpdateVerification.expectedHash("\(hash) x.zip", name: "Arsenal-3.0.0-arm64.zip") }
        #expect(throws: UpdateFailure.self) { try UpdateVerification.expectedHash("\(hash) x.zip\n\(hash) x.zip", name: "x.zip") }
        #expect(throws: UpdateFailure.self) { try UpdateVerification.expectedHash("short x.zip", name: "x.zip") }
    }
    @Test func releaseLayout() throws {
        let json = Data(
            #"{"tag_name":"v3.0.0","body":"Together","draft":false,"prerelease":false,"assets":[{"name":"Arsenal-3.0.0-arm64.zip","size":123,"browser_download_url":"https://example.com/app.zip"},{"name":"SHA256SUMS","size":90,"browser_download_url":"https://example.com/SHA256SUMS"}]}"#
                .utf8)
        let parsed = try UpdateRelease.parse(json)
        let release = try #require(parsed)
        #expect(release.version == "3.0.0"); #expect(release.notes == "Together"); #expect(release.size == 123)
        let draft = Data(String(decoding: json, as: UTF8.self).replacingOccurrences(of: #""draft":false"#, with: #""draft":true"#).utf8)
        #expect(try UpdateRelease.parse(draft) == nil)
        let mislabeledPrerelease = Data(String(decoding: json, as: UTF8.self).replacingOccurrences(of: "v3.0.0", with: "v3.0.0-rc.1").utf8)
        #expect(try UpdateRelease.parse(mislabeledPrerelease) == nil)
    }
    @Test func allowedSources() throws {
        let source = try UpdateSource(repository: "example/Arsenal")
        #expect(source.latest.absoluteString == "https://api.github.com/repos/example/Arsenal/releases/latest")
        #expect(!source.allows(URL(string: "http://127.0.0.1/x")!))
        #expect(!source.allows(URL(string: "https://example.com/asset.zip")!))
        #expect(!source.allows(URL(string: "https://github.com.example.com/asset.zip")!))
        #expect(source.allows(URL(string: "https://release-assets.githubusercontent.com/asset.zip")!))
        #expect(source.allows(URL(string: "https://github.com/owner/repo/releases/download/v3.0.0/asset.zip")!))
        #expect(!source.allows(URL(string: "https://user:secret@example.com/x")!))
        #expect(throws: UpdateFailure.self) { try UpdateSource(repository: "owner/repo?redirect=x") }
        #expect(throws: UpdateFailure.self) { try UpdateSource(testServer: URL(string: "http://example.com/x")!) }
    }
    @MainActor @Test func productionDeadlinesAndAtomicExchange() throws {
        let limits = UpdateInstaller.Limits()
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent(
            "Sources/Arsenal/App/AppDelegate.swift")
        let declaration = try #require(String(contentsOf: source, encoding: .utf8).components(separatedBy: "\n").first { $0.contains("static let quitLimit: Duration =") })
        let expression = declaration.components(separatedBy: "=").last!.trimmingCharacters(in: .whitespaces)
        #expect(expression.hasPrefix(".seconds("))
        let seconds = try #require(Double(expression.dropFirst(".seconds(".count).dropLast()))
        #expect(Duration.seconds(limits.exit) > .seconds(seconds)); #expect(limits.lateExit > 0); #expect(limits.launch == 60); #expect(limits.restore == 1800)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-swap-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("installed-file"), previous = root.appendingPathComponent("previous-file")
        try Data("installed".utf8).write(to: destination); try Data("new".utf8).write(to: previous)
        try UpdateInstaller.exchange(previous, destination)
        #expect(try String(contentsOf: destination, encoding: .utf8) == "new")
        #expect(try String(contentsOf: previous, encoding: .utf8) == "installed")
        try UpdateInstaller.exchange(previous, destination)
        #expect(try String(contentsOf: destination, encoding: .utf8) == "installed")
    }
    @Test func toolsHaveBoundedEventWait() throws {
        #expect(throws: UpdateFailure.self) { try UpdateVerification.tool("/bin/sleep", ["10"], timeout: 0.05) }
        #expect(try UpdateVerification.tool("/usr/bin/true", []) == "")
    }
    @MainActor @Test func earlyAcknowledgementDoesNotNeedRestoreRecord() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-ack-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let token = UUID().uuidString, args = ["Arsenal", "--arsenal-update", token]
        try UpdateController.acknowledgeRelaunch(support: root, arguments: args)
        let ack = try JSONDecoder().decode(UpdateAcknowledgement.self, from: Data(contentsOf: UpdateAcknowledgement.url(support: root, token: token)))
        #expect(ack.token == token); #expect(ack.pid == getpid()); #expect(ack.phase == .launched)
        let defaults = UserDefaults(suiteName: root.lastPathComponent)!
        defer { defaults.removePersistentDomain(forName: root.lastPathComponent) }
        let hooks = UpdateServiceHooks(acquireIdle: { [] }, releaseIdle: {}, restore: { _ in throw UpdateFailure("Must not restore missing state") })
        let controller = UpdateController(support: root, hooks: hooks, repository: nil, currentVersion: "3.0.0", defaults: defaults, quit: {})
        await controller.restoreAfterRelaunch(arguments: args)
        if case .failed = controller.state {} else { Issue.record("Missing restore state must report a model restore failure.") }
        #expect(FileManager.default.fileExists(atPath: UpdateAcknowledgement.url(support: root, token: token).path))
    }
    @MainActor @Test func restoreCleanupAndLateLaunchAreTokenScoped() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-token-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let token = UUID().uuidString, other = UUID().uuidString
        let record = root.appendingPathComponent(UpdateRestore.fileName)
        try JSONEncoder().encode(UpdateRestore(token: other, version: "3.0.0", models: [])).write(to: record)
        UpdateRestore.remove(support: root, token: token)
        #expect(FileManager.default.fileExists(atPath: record.path))
        UpdateRestore.remove(support: root, token: other)
        #expect(!FileManager.default.fileExists(atPath: record.path))
        let failure = root.appendingPathComponent("update-failure.txt")
        try Data("Unconfirmed [update:\(other)]".utf8).write(to: failure)
        try UpdateController.acknowledgeRelaunch(support: root, arguments: ["Arsenal", "--arsenal-update", token])
        #expect(FileManager.default.fileExists(atPath: failure.path))
        try Data("Unconfirmed [update:\(token)]".utf8).write(to: failure)
        try UpdateController.acknowledgeRelaunch(support: root, arguments: ["Arsenal", "--arsenal-update", token])
        #expect(!FileManager.default.fileExists(atPath: failure.path))
        try Data("Unconfirmed [update:\(token)]".utf8).write(to: failure)
        let hooks = UpdateServiceHooks(acquireIdle: { [] }, releaseIdle: {}, restore: { _ in })
        let controller = UpdateController(support: root, hooks: hooks, repository: nil, currentVersion: "3.0.0", quit: {})
        controller.start()
        #expect(controller.state == .failed(reason: "Unconfirmed"))
    }
    @MainActor @Test func reconciliationPreservesPreV3AndUnverifiedBackups() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-reconcile-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        func bundle(_ name: String, version: String) throws -> URL {
            let app = root.appendingPathComponent(name), contents = app.appendingPathComponent("Contents")
            try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
            try PropertyListSerialization.data(fromPropertyList: ["CFBundleIdentifier": "dev.arsenal.app", "CFBundleShortVersionString": version], format: .xml, options: 0)
                .write(to: contents.appendingPathComponent("Info.plist"))
            return app
        }
        // Production identity validation uses metadata directories, never a LaunchServices app bundle.
        let app = try bundle("Arsenal.fixture", version: "2.2.0")
        let backup = try bundle(".Arsenal.app.previous-\(UUID().uuidString)", version: "1.0.0")
        let previous = try bundle("Arsenal.fixture.previous", version: "3.0.0")
        UpdateInstaller.reconcile(app)
        #expect(!FileManager.default.fileExists(atPath: backup.path))
        let info = try Data(contentsOf: previous.appendingPathComponent("Contents/Info.plist"))
        #expect(String(decoding: info, as: UTF8.self).contains("1.0.0"))
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasPrefix(".Arsenal.app.retained-") })
        let unverified = try bundle(".Arsenal.app.previous-\(UUID().uuidString)", version: "0.9.0")
        UpdateInstaller.reconcile(app)
        #expect(FileManager.default.fileExists(atPath: unverified.path))
        UpdateInstaller.reconcile(root.appendingPathComponent("absent.app")) // Inspection failure is best effort.
        #expect(FileManager.default.fileExists(atPath: unverified.path))
    }
    @MainActor @Test func unsetRepositoryDoesNothing() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-unit-\(UUID().uuidString)")
        let defaults = UserDefaults(suiteName: root.lastPathComponent)!
        let hooks = UpdateServiceHooks(acquireIdle: { throw UpdateFailure("Must not acquire") }, releaseIdle: {}, restore: { _ in })
        let controller = UpdateController(support: root, hooks: hooks, repository: nil, currentVersion: "2.2.0", defaults: defaults, quit: {})
        controller.start(); await controller.checkForUpdates(); await controller.installAvailableUpdate()
        #expect(!controller.enabled); #expect(!controller.schedulerArmed); #expect(controller.state == .none)
        #expect(!FileManager.default.fileExists(atPath: root.path))
        defaults.removePersistentDomain(forName: root.lastPathComponent)
    }
}
