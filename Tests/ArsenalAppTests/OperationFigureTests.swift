// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// The Models table's two figures for models whose kind has no generative speed unit (Parakeet's transcription) and for
/// an image model whose operation is not generation (BiRefNet), read from the result files in their model folders.
@MainActor @Suite(.menuState, .serialized) final class OperationFigureTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-operation-figures-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// A row as the service builds it: the folder's schema-2 results through the reader.
    private func row(_ id: String, name: String, params: String) throws -> ModelRow {
        let folder = repo.appendingPathComponent("models/" + id)
        let manifest = try ModelManifest.load(folder)
        let results = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("results"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }.compactMap { try? PrecisionStatsResult.decode(Data(contentsOf: $0)) }
            .filter { PrecisionEvidence.current($0.stats, revision: manifest.weights.revision) }
        var row = ModelRow(
            id: id, name: name, kind: manifest.kind, params: params, licence: "Apache-2.0", precisions: manifest.precisions, selectedPrecision: manifest.basePrecision,
            fast: false, sizeBytes: 1_000_000_000, summary: "", publisher: "", fastExact: false)
        row.runner = "native"
        for precision in manifest.precisions { row.states[precision] = PrecisionState(state: .ready) }
        row.precisionStats = PrecisionStatsReader.sets(
            kind: manifest.kind, baseDtype: manifest.basePrecision, runner: "native", supported: manifest.precisions, results: results, chip: "m5-max", gpuCores: 40)
        return row
    }

    @Test func quickCheckPeakAndRequestTimeFillAnOperationRow() throws {
        let parakeet = try row("parakeet-v3-ultra", name: "Parakeet v3 Ultra", params: "627M")
        let memory = TableFigures.figure(parakeet, precision: "fp16", column: .memory, host: "M5 Max")
        // The 1 Nov test night: 1740.8 MiB = 1.83 GB (10^9), shown in Activity Monitor's GB.
        #expect(memory.value == "1.70 GB" && memory.reference == nil)
        #expect(memory.help.contains("Quick check on M5 Max") && memory.help.contains(TableFigures.quickCheckHelp))
        #expect(!memory.help.contains("Measured"))
        let speed = TableFigures.figure(parakeet, precision: "fp16", column: .speed, host: "M5 Max")
        #expect(speed.value == "95 ms" && speed.delta == nil)
        #expect(speed.help.hasPrefix("fp16 · 95 ms per request"))
        #expect(speed.help.contains("One warm Transcribe request on a 3.3 s speech-and-music clip."))
        // Another Mac: the speed names its chip; peak memory stands as it is.
        let elsewhere = TableFigures.figure(parakeet, precision: "fp16", column: .speed, host: "M3 Pro")
        #expect(elsewhere.value == "95 ms" && elsewhere.reference == "M5 Max")
        #expect(RowPresentation.hasFigures(parakeet))
        #expect(TableFigures.headerHelp(.speed, kind: .audio) == "Time for one request of the model's own task. Lower is faster.")
    }

    @Test func requestTimeFormats() {
        #expect(TableFigures.format(.secondsPerRequest, 0.0004) == "1 ms")
        #expect(TableFigures.format(.secondsPerRequest, 0.897) == "897 ms")
        #expect(TableFigures.format(.secondsPerRequest, 2.866) == "2.9 s")
        #expect(TableFigures.format(.secondsPerRequest, 42) == "42 s")
    }

    @Test func intakeFiguresStayForBackgroundRemoval() throws {
        let birefnet = try row("birefnet", name: "BiRefNet", params: "220M")
        let speed = TableFigures.figure(birefnet, precision: "fp16", column: .speed, host: "M5 Max")
        #expect(speed.value == "0.4 s" && speed.help.contains("Quick check"))
        #expect(TableFigures.figure(birefnet, precision: "fp16", column: .memory, host: "M5 Max").value == "5.37 GB")
    }

    /// Review render: Parakeet (audio), BiRefNet (images, background removal) and TeleOCR (a chat-kind reader, peak only).
    /// Run with ARSENAL_TABLE_FIGURE_RENDERS=<dir>.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_TABLE_FIGURE_RENDERS"] != nil))
    func renderOperationFigures() async throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_TABLE_FIGURE_RENDERS"]!)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { Appearance.shared.set(.dark); NSApp.appearance = nil }
        let rows = [
            try row("parakeet-v3-ultra", name: "Parakeet v3 Ultra", params: "627M"), try row("birefnet", name: "BiRefNet", params: "220M"),
            try row("teleocr", name: "TeleOCR", params: "2B"), try row("ui-venus-1.5-2b", name: "UI-Venus 1.5 2B", params: "2B")
        ]
        let kinds = try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).filter { ["audio", "images", "chat", "perception"].contains($0.id) }
        let c = TableController(service: FixtureService(rows: rows, supportDirectory: support), chip: "M5 Max")
        c.apply(rows: rows, kinds: kinds, catalog: CatalogStatus(source: "local", pinnedCommit: "f600b7d4c1e2", verified: true))
        Appearance.shared.set(.dark); NSApp.appearance = Appearance.shared.nsAppearance
        try await Task.sleep(for: .milliseconds(100))
        await QA.renderTable(c, to: directory.appendingPathComponent("operation-figures-dark.png"))
        try RenderDelegate.tooltips(c).write(to: directory.appendingPathComponent("operation-figures-tooltips.txt"), atomically: true, encoding: .utf8)
        c.stop()
    }
}
