// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// The Models table's Precision column at real size with the user's own model set (25 Oct): a catalog snapshot read from the
/// running app's local API (`GET /arsenal/v1/catalog`, saved as JSON), the precisions on disk as his store holds them
/// (Laya's int8 and int4 are hardlinks to its fp16 file, so the store counts the fp16 bytes for each), and Kokoro loaded
/// at int4. Nothing is read from or written to his support folder here.
/// Run with ARSENAL_PRECISION_RENDERS=<dir> ARSENAL_PRECISION_SNAPSHOT=<catalog.json>.
@MainActor @Suite(.menuState, .serialized) final class PrecisionRenders {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-precision-render-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// The precisions on disk in the user's store, by model (25 Oct).
    static let onDisk: [String: [String]] = {
        var disk: [String: [String]] = ["kokoro-82m": ["fp16", "int4"]]
        for id in ["laya-english", "laya-multilingual", "laya-typed-decisions", "index-translate-2b", "ovis-vl-embedding-2b", "teleocr"] {
            disk[id] = ["fp16", "int8", "int4"]
        }
        for id in ["pocket-tts", "von-1.1", "von-1.2"] { disk[id] = ["bf16", "int8", "int4"] }
        for id in [
            "audiosr-basic", "big-lama", "birefnet", "da3-small", "ddcolor-modelscope", "harrier-oss-v1-0.6b", "htdemucs-ft",
            "moge-3-vitl", "parakeet-v3-ultra", "pyannote-community-1", "qwen3-asr-0.6b", "qwen3-asr-1.7b", "rf-detr-nano",
            "rife-4.25", "smart-turn-v3", "tipsv2-l14"
        ] { disk[id] = ["fp16"] }
        return disk
    }()
    /// Models whose quantized copies are hardlinks to the 16-bit file (quantized at load): the store counts its bytes for each.
    static let sharedFile: Set<String> = ["laya-english", "laya-multilingual", "laya-typed-decisions"]
    /// index-translate-2b's int4 is a tier the catalog withdrew; its files are still on disk (1,386,497,670 bytes, measured).
    static let withdrawn: [String: [String: Int64]] = ["index-translate-2b": ["int4": 1_386_497_670]]

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_PRECISION_RENDERS"] != nil))
    func renderUsersTable() async throws {
        let env = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: env["ARSENAL_PRECISION_RENDERS"]!)
        let snapshot = URL(fileURLWithPath: try #require(env["ARSENAL_PRECISION_SNAPSHOT"], "ARSENAL_PRECISION_SNAPSHOT=<catalog.json>"))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let entries = try JSONDecoder().decode([CatalogEntry].self, from: Data(contentsOf: snapshot))
        #expect(entries.count == 42)
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let kinds = try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds"))

        var rows: [ModelRow] = [], disk: [OnDiskModel] = [], sizes: [String: [String: Int64]] = [:]
        for entry in entries {
            guard let dtypes = Self.onDisk[entry.id] else { continue }
            var row = entry.row
            row.states = [:]
            for dtype in dtypes where row.precisions.contains(dtype) {
                row.states[dtype] = PrecisionState(state: entry.id == "kokoro-82m" && dtype == "int4" ? .loaded : .ready)
            }
            rows.append(row)
            sizes[entry.id] = entry.sizesByDtype
            let base = entry.sizesByDtype[dtypes[0]] ?? 0
            disk.append(
                OnDiskModel(
                    id: entry.id,
                    precisions: dtypes.map { dtype in
                        let bytes = Self.withdrawn[entry.id]?[dtype] ?? (Self.sharedFile.contains(entry.id) ? base : entry.sizesByDtype[dtype] ?? 0)
                        return OnDiskPrecision(dtype: dtype, bytes: bytes, inUse: entry.id == "kokoro-82m" && dtype == "int4")
                    }, cacheBytes: 0, offered: true))
        }
        #expect(rows.count == Self.onDisk.count)
        let service = FixtureService(rows: rows, kinds: kinds, supportDirectory: support)
        await service.setOnDisk(disk)
        await service.setCatalogSizes(sizes)
        let c = TableController(service: service, chip: "M5 Max")
        await c.refresh()
        defer { c.stop() }
        // The whole table, unscrolled, so every row is in the picture.
        let maxHeight = ModelTable.maxHeight
        ModelTable.maxHeight = 10_000
        defer { ModelTable.maxHeight = maxHeight }

        var report: [String] = []
        for row in c.groups.flatMap(\.rows) {
            let lines = c.precisionLines(row)
            report.append(
                "\(row.id)\t" + lines.map { "\($0.copy.label) \($0.copy.size)\($0.selected ? " [chosen]" : "")\($0.loaded ? " [loaded]" : "")" }.joined(separator: " | "))
        }
        try report.joined(separator: "\n").write(to: directory.appendingPathComponent("models-table-lines.txt"), atomically: true, encoding: .utf8)
        let laya = try #require(c.rows.first { $0.id == "laya-english" })
        #expect(c.precisionLines(laya).map(\.copy.size) == ["846 MB", "450 MB", "238 MB"])

        Appearance.shared.set(.dark); NSApp.appearance = Appearance.shared.nsAppearance
        defer { NSApp.appearance = nil }
        try await Task.sleep(for: .milliseconds(100))
        await QA.renderTable(c, to: directory.appendingPathComponent("models-table.png"))
        try "table \(ModelTable.width(c)) x \(ModelTable.height(c)) pt\n".write(
            to: directory.appendingPathComponent("models-table-size.txt"), atomically: true, encoding: .utf8)
    }
}
