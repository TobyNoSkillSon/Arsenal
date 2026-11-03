// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalCommands

/// A quick-check table result binds to its quick-check record by run and value; it can only be a quick check of peak
/// memory or request time, with no quality verdict.
struct QuickCheckValidationTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private let table = "results/quick-m5-max-fp16-native-exact.json"
    private let record = "results/quick-checks/night-2026-10-22.json"

    /// A scratch copy of a tracked operation-kind folder, so mutations never touch the repository.
    private func folder() throws -> (URL, ModelManifest) {
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-quick-check-\(UUID().uuidString)/parakeet-v3-ultra")
        try FileManager.default.createDirectory(at: scratch.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("models/parakeet-v3-ultra"), to: scratch)
        return (scratch, try ModelManifest.load(scratch))
    }
    private func validate(_ root: JSONObject, _ path: String, _ folder: URL, _ manifest: ModelManifest) throws {
        try CatalogResultValidation.validate(data: WireJSON.encode(root), path: path, folder: folder, manifest: manifest)
    }
    private func cell(_ root: JSONObject, _ metric: String, _ change: (inout JSONObject) -> Void) -> JSONObject {
        var root = root, stats = root["stats"]!.object!, metrics = stats["metrics"]!.object!, cell = metrics[metric]!.object!
        change(&cell); metrics[metric] = .object(cell); stats["metrics"] = .object(metrics); root["stats"] = .object(stats)
        return root
    }

    @Test func boundQuickCheckValidatesAndForgeriesFail() throws {
        let (folder, manifest) = try folder()
        defer { try? FileManager.default.removeItem(at: folder.deletingLastPathComponent()) }
        let original = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(table)))
        try validate(original, table, folder, manifest)
        try validate(try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(record))), record, folder, manifest)

        let forged: [String: JSONObject] = [
            "peak differs": cell(original, "peakMemoryGB") { $0["value"] = .number(1.5) },
            "time differs": cell(original, "secondsPerRequest") { $0["value"] = .number(0.01) },
            "relabelled measured": cell(original, "peakMemoryGB") { $0["state"] = .string("measured") },
            "other run": cell(original, "peakMemoryGB") { cell in
                var provenance = cell["provenance"]!.object!; provenance["run"] = .string("night-2026-10-21-fp16"); cell["provenance"] = .object(provenance)
            },
            "quality verdict": {
                var root = original, stats = root["stats"]!.object!
                stats["qualityCheckPassed"] = .bool(true); root["stats"] = .object(stats); return root
            }()
        ]
        for (name, root) in forged {
            #expect(throws: (any Error).self, "\(name)") { try validate(root, table, folder, manifest) }
        }
    }

    @Test func quickCheckRecordNeedsTheFolderRevisionAndACompletedRequest() throws {
        let (folder, manifest) = try folder()
        defer { try? FileManager.default.removeItem(at: folder.deletingLastPathComponent()) }
        let original = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(record)))
        var stale = original; stale["revision"] = .string(String(repeating: "0", count: 40))
        var failed = original, rows = original["rows"]!.array!, row = rows[0].object!, ops = row["ops"]!.object!
        for (name, op) in ops { var op = op.object!; op["status"] = .number(422); ops[name] = .object(op) }
        row["ops"] = .object(ops); rows[0] = .object(row); failed["rows"] = .array(rows)
        var errored = original, errorRows = original["rows"]!.array!, errorRow = errorRows[0].object!
        errorRow["error"] = .string("peak.py exit 1"); errorRows[0] = .object(errorRow); errored["rows"] = .array(errorRows)
        for (name, root) in [("stale", stale), ("no completed request", failed), ("error row", errored)] {
            #expect(throws: (any Error).self, "\(name)") { try validate(root, record, folder, manifest) }
        }
    }
}
