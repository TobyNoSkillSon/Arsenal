// SPDX-License-Identifier: MIT
import Foundation
import Darwin

/// Historical regression inputs are test fixtures. Build a private merged tree so validators
/// exercise their ordinary filesystem path without requiring local lab evidence.
let intakeFixtureRepository: URL = {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    // One root per global: ArsenalCoreTests and ArsenalRunnerCheckTests both define this, and they share one test process.
    let root = repo.appendingPathComponent(".build/repository-fixtures/" + UUID().uuidString)
    let fm = FileManager.default
    do {
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        try fm.copyItem(at: repo.appendingPathComponent("models"), to: root.appendingPathComponent("models"))
        // Withdrawn folders keep their historical descriptors in retired-models/; the legacy checks still read them.
        for folder in try fm.contentsOfDirectory(at: repo.appendingPathComponent("retired-models"), includingPropertiesForKeys: nil)
        where !fm.fileExists(atPath: root.appendingPathComponent("models/" + folder.lastPathComponent).path) {
            try fm.copyItem(at: folder, to: root.appendingPathComponent("models/" + folder.lastPathComponent))
        }
        let fixtures = try fm.contentsOfDirectory(at: repo.appendingPathComponent("Tests/Fixtures/Intake"), includingPropertiesForKeys: nil)
        for fixture in fixtures where fixture.pathExtension == "json" {
            let files = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: fixture))
            for (path, contents) in files {
                guard !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else { throw CocoaError(.fileReadCorruptFile) }
                let target = root.appendingPathComponent("models/" + fixture.deletingPathExtension().lastPathComponent + "/" + path)
                try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data(contents.utf8).write(to: target)
            }
        }
        try fm.createDirectory(at: root.appendingPathComponent("Resources"), withIntermediateDirectories: true)
        try fm.copyItem(at: repo.appendingPathComponent("Resources/kinds"), to: root.appendingPathComponent("Resources/kinds"))
        atexit { try? FileManager.default.removeItem(at: intakeFixtureRepository) }
        return root
    } catch { preconditionFailure("Cannot materialize intake fixtures: \(error)") }
}()
