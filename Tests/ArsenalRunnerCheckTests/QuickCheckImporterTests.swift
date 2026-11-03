// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalCommands

/// scripts/import-quick-checks.py and catalog validation agree: whatever the importer writes validates, including an
/// image model's seconds per image for an operation that is not generation. The importer keeps an earlier good row over
/// a later failure, never overwrites a file it did not write, and leaves a precision alone when another result (any
/// mode) already has figures.
struct QuickCheckImporterTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    private func scratch(_ models: [String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-importer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("models"), withIntermediateDirectories: true)
        for model in models {
            let folder = root.appendingPathComponent("models/" + model)
            try FileManager.default.copyItem(at: shippedOrRetiredFolder(repo, model), to: folder)
            try? FileManager.default.removeItem(at: folder.appendingPathComponent("results"))
        }
        return root
    }
    private func row(_ model: String, _ dtype: String, op: String, status: Int = 200, seconds: Double = 12.5, error: String? = nil) throws -> String {
        var row: JSONObject = [
            "model": .string(model), "dtype": .string(dtype), "date": "2026-10-22T04:52:20+0200", "chip": "m5-max", "machine": "Mac in use (fixture)",
            "peak_footprint_mb": 30_000, "ops": [op: ["seconds_warm": .number(seconds), "seconds_first": 14, "status": .number(Double(status))]]
        ]
        if let error { row["error"] = .string(error) }
        return String(decoding: try WireJSON.encode(row), as: UTF8.self)
    }
    private func importer(_ root: URL, _ lines: [String]) throws {
        let jsonl = root.appendingPathComponent("lab/night/peak-results.jsonl")
        try FileManager.default.createDirectory(at: jsonl.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (lines.joined(separator: "\n") + "\n").write(to: jsonl, atomically: true, encoding: .utf8)
        let process = Process(), output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "-I", repo.appendingPathComponent("scripts/import-quick-checks.py").path, jsonl.path, "--repo", root.path, "--gpu-cores", "40"]
        process.standardOutput = output; process.standardError = output
        try process.run(); let data = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        #expect(process.terminationStatus == 0, "\(String(decoding: data, as: UTF8.self))")
    }

    @Test func imageOperationSecondsPerImageValidatesAndLaterFailureKeepsTheGoodRow() throws {
        let root = try scratch(["qwen-image-2.1-turbo"])
        defer { try? FileManager.default.removeItem(at: root) }
        try importer(root, [try row("qwen-image-2.1-turbo", "bf16", op: "edit"), try row("qwen-image-2.1-turbo", "bf16", op: "edit", error: "peak.py exit 4")])
        let folder = root.appendingPathComponent("models/qwen-image-2.1-turbo")
        let result = try PrecisionStatsResult.decode(Data(contentsOf: folder.appendingPathComponent("results/quick-m5-max-bf16-native-exact.json")))
        #expect(result.stats.value(.secondsPerImage).state == .quickCheck && result.stats.value(.secondsPerImage).value == 12.5)
        #expect(result.stats.value(.peakMemoryGB).basis == "Load and two Edit requests on a short fixed test input.")
        #expect(result.stats.value(.peakMemoryGB).provenance?.method.contains("all Arsenal worker processes (only the measured model was loaded)") == true)
        let kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
        let errors = CatalogCommand.validation(folder: folder, kinds: kinds)
        #expect(errors.isEmpty, "\(errors)")
    }

    @Test func importerNeverOverwritesOrDuplicatesOtherResults() throws {
        let root = try scratch(["parakeet-v3-ultra", "smart-turn-v3"])
        defer { try? FileManager.default.removeItem(at: root) }
        // A hand-written file under the importer's name.
        let parakeet = root.appendingPathComponent("models/parakeet-v3-ultra/results")
        try FileManager.default.createDirectory(at: parakeet, withIntermediateDirectories: true)
        let hand = Data(#"{"note": "hand-written"}"#.utf8)
        try hand.write(to: parakeet.appendingPathComponent("quick-m5-max-fp16-native-exact.json"))
        // Another result with a figure in a different mode.
        let turn = root.appendingPathComponent("models/smart-turn-v3/results")
        try FileManager.default.createDirectory(at: turn, withIntermediateDirectories: true)
        var stats = PrecisionStatSet(kind: .audio, dtype: "fp16", runner: "native")
        let provenance = StatProvenance(chip: "m5-max", date: "2026-10-16T00:00:00Z", method: "intake", source: "results/intake/run.json", revision: "x", run: "run-fp16")
        stats.availability = StatValue(state: .quickCheck, provenance: provenance)
        stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: 0.3, provenance: provenance))
        try JSONEncoder().encode(PrecisionStatsResult(model: "smart-turn-v3", mode: "fast", stats: stats)).write(to: turn.appendingPathComponent("intake-fast.json"))

        try importer(root, [try row("parakeet-v3-ultra", "fp16", op: "transcribe", seconds: 0.04), try row("smart-turn-v3", "fp16", op: "end_of_turn", seconds: 0.03)])
        #expect(try Data(contentsOf: parakeet.appendingPathComponent("quick-m5-max-fp16-native-exact.json")) == hand)
        #expect(try FileManager.default.contentsOfDirectory(atPath: turn.path) == ["intake-fast.json"])
    }

    /// A rerun under the same run once rewrote the record with only its own rows, so the earlier precision's result
    /// pointed at a row that was gone and catalog validation reported an identity mismatch.
    @Test func rerunUnderTheSameRunKeepsEarlierRows() throws {
        // A model that still declares int8 (BiRefNet's int8 was withdrawn on 22 Oct, so its rows no longer validate).
        let root = try scratch(["ovis-vl-embedding-2b"])
        defer { try? FileManager.default.removeItem(at: root) }
        try importer(root, [try row("ovis-vl-embedding-2b", "fp16", op: "embed", seconds: 0.5), try row("ovis-vl-embedding-2b", "int8", op: "embed", seconds: 0.4)])
        try importer(root, [try row("ovis-vl-embedding-2b", "int8", op: "embed", seconds: 0.3)])
        let folder = root.appendingPathComponent("models/ovis-vl-embedding-2b")
        let record = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("results/quick-checks/night-2026-10-22.json")))
        let rows = record["rows"]?.array ?? []
        #expect(rows.count == 2)
        #expect(rows.compactMap { $0["precision"]?.string } == ["fp16", "int8"])
        #expect(rows.last?["ops"]?["embed"]?["seconds_warm"]?.number == 0.3)
        let kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
        let errors = CatalogCommand.validation(folder: folder, kinds: kinds)
        #expect(errors.isEmpty, "\(errors)")
    }

    /// Test night 1 Nov: a kit folder whose own kind declares no table figures was skipped ("kind … has no table
    /// figures"). Its figures now go to Arsenal's Resources/figures, bound to its pins; the kit folder keeps the record
    /// and gets no table file, and a model with only an installed folder is read, never written.
    @Test func kitKindFiguresGoToArsenalsOwnFileNotTheFolder() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-importer-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let kit = root.appendingPathComponent("models/chronos-2"), installed = root.appendingPathComponent("installed/t0-alpha")
        try FileManager.default.createDirectory(at: kit.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: installed.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("models/chronos-2"), to: kit)
        // A shipped kit model's kind is signed (Resources/kinds), not the folder's own.
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Resources"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("Resources/kinds"), to: root.appendingPathComponent("Resources/kinds"))
        try FileManager.default.copyItem(at: repo.appendingPathComponent("models/t0-alpha"), to: installed)
        try? FileManager.default.removeItem(at: kit.appendingPathComponent("results"))
        func digest(_ folder: URL) throws -> String { try LocalModelFolders.discoverSnapshot(folder).contentDigest }
        let before = try digest(installed)
        let jsonl = root.appendingPathComponent("lab/night/peak-results.jsonl")
        try FileManager.default.createDirectory(at: jsonl.deletingLastPathComponent(), withIntermediateDirectories: true)
        try ([try row("chronos-2", "fp16", op: "forecast", seconds: 0.073), try row("t0-alpha", "fp16", op: "forecast", seconds: 0.07)].joined(separator: "\n") + "\n")
            .write(to: jsonl, atomically: true, encoding: .utf8)
        func run() throws {
            let process = Process(), output = Pipe()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [
                "python3", "-I", repo.appendingPathComponent("scripts/import-quick-checks.py").path, jsonl.path, "--repo", root.path, "--gpu-cores", "40",
                "--local-models", installed.deletingLastPathComponent().path
            ]
            process.standardOutput = output; process.standardError = output
            try process.run(); let log = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
            #expect(process.terminationStatus == 0, "\(String(decoding: log, as: UTF8.self))")
        }
        try run()
        let store = root.appendingPathComponent("Resources/figures/folder-quick-checks.json"), first = try Data(contentsOf: store)
        // A rerun of the same night changes nothing.
        try run()
        #expect(try Data(contentsOf: store) == first)

        let figures = FolderReferenceFigures.load(resources: root.appendingPathComponent("Resources"))
        #expect(figures.entries.map(\.model) == ["chronos-2", "t0-alpha"])
        let chronos = try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: kit.appendingPathComponent("model.toml"), encoding: .utf8))
        let entry = try #require(figures.figures(chronos, runner: "kit")["fp16"])
        #expect(entry.secondsPerRequest.value == 0.073 && entry.secondsPerRequest.basis == "One warm Forecast request on a short fixed test input.")
        #expect(entry.peakMemoryGB.provenance?.source == "models/chronos-2/results/quick-checks/night-2026-10-22.json; fp16")
        // The kit folder holds the record only; the installed folder is untouched.
        #expect(try FileManager.default.contentsOfDirectory(atPath: kit.appendingPathComponent("results").path) == ["quick-checks"])
        #expect(try digest(installed) == before)
        // An entry without a repo folder cites its measurements file by a path that exists (here inside the repo).
        #expect(figures.entries.last?.peakMemoryGB.provenance?.source == "lab/night/peak-results.jsonl; fp16")
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("lab/night/peak-results.jsonl").path))
    }
}

/// The shipped folder, or the withdrawn one in retired-models/ (non-open licences are kept there, not shipped).
private func shippedOrRetiredFolder(_ repo: URL, _ id: String) -> URL {
    let shipped = repo.appendingPathComponent("models/" + id)
    return FileManager.default.fileExists(atPath: shipped.path) ? shipped : repo.appendingPathComponent("retired-models/" + id)
}
