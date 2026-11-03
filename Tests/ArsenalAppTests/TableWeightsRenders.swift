// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// Review renders for the weights in the Models table: one 16-bit copy on disk; 16-bit and int4
/// with both sizes; a loaded row whose in-use copy cannot be deleted; a model no catalog lists; the unlisted-weights row;
/// a copy whose delete is running. Each at the menu's real width and height, dark and light, first frame.
/// Run with ARSENAL_TABLE_WEIGHTS_RENDERS=<dir> (the table's renders go to lab/renders/table).
@MainActor @Suite(.menuState, .serialized) final class TableWeightsRenders {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-table-weights-renders-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    private func row(_ id: String, _ states: [String: ModelState], selected: String? = nil, bytes: [String: Int64] = [:]) -> ModelRow {
        var r = Fixtures.rows(.firstLaunch).first { $0.id == id }!
        for (dtype, state) in states { r.states[dtype] = PrecisionState(state: state) }
        for (dtype, size) in bytes { r.downloadBytes[dtype] = size }
        if let selected { r.selectedPrecision = selected }
        return r
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_TABLE_WEIGHTS_RENDERS"] != nil))
    func renderWeights() async throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_TABLE_WEIGHTS_RENDERS"]!)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { Appearance.shared.set(.dark); NSApp.appearance = nil }

        let kokoro = row("kokoro-82m", ["fp16": .ready], selected: "fp16", bytes: ["fp16": 341_742_463])
        let flux = row("flux2-klein-4b", ["bf16": .notInstalled], selected: "bf16", bytes: ["bf16": 15_980_140_843])
        func qwen(_ states: [String: ModelState], selected: String) -> ModelRow {
            row("qwen3.8-27b", states, selected: selected, bytes: ["bf16": 54_000_000_000, "int4": 15_000_000_000])
        }
        func controller(_ rows: [ModelRow], unlisted: [OnDiskModel] = [], orphans: Int64 = 0) async -> (TableController, FixtureService) {
            let service = FixtureService(rows: rows, supportDirectory: support.appendingPathComponent(UUID().uuidString))
            await service.setUnlisted(unlisted)
            await service.setOrphanBytes(orphans)
            let c = TableController(service: service, chip: "M5 Max")
            c.confirmDeleteWeights = { _, _ in true }
            await c.refresh()
            return (c, service)
        }

        // (a) One 16-bit copy on disk, the 4-bit cell not yet made.
        let (a, _) = await controller([kokoro, flux, qwen([:], selected: "bf16")])
        // (b) 16-bit and int4 on disk: both lines, sizes shown.
        let (b, _) = await controller([kokoro, qwen(["bf16": .ready, "int4": .ready], selected: "bf16")])
        // (c) Loaded at int4: its 4-bit line's trash is off ("Unload to delete"); the row's trash is hovered to show it.
        let (cc, _) = await controller([kokoro, qwen(["bf16": .ready, "int4": .loaded], selected: "int4")])
        cc.renderHover = "qwen3.8-27b"
        // (d) A model no catalog lists, with weights on disk: its own group, last.
        let retired = [
            OnDiskModel(id: "chandra-ocr-2", precisions: [OnDiskPrecision(dtype: "bf16", bytes: 3_100_000_000, inUse: false)], cacheBytes: 0, offered: false),
            OnDiskModel(
                id: "ui-venus-2-9b",
                precisions: [OnDiskPrecision(dtype: "bf16", bytes: 18_500_000_000, inUse: false), OnDiskPrecision(dtype: "int4", bytes: 5_200_000_000, inUse: false)],
                cacheBytes: 0, offered: false)
        ]
        let (d, _) = await controller([kokoro, flux], unlisted: retired)
        // (e) The unlisted-weights row: weights no model refers to, one button.
        let (e, _) = await controller([kokoro, qwen(["bf16": .ready, "int4": .ready], selected: "bf16")], orphans: 3_100_000_000)
        // (f) A copy's delete running: its trash is off with "Deleting…", its line still there.
        let (f, fService) = await controller([kokoro, qwen(["bf16": .ready, "int4": .ready], selected: "bf16")])
        // Long enough to hold through every frame below; the delete is awaited at the end.
        await fService.setDelay("deletePrecision", .seconds(60))
        f.deleteCopy(f.rows.first { $0.id == "qwen3.8-27b" }!, "int4")
        try await Task.sleep(for: .milliseconds(150))

        let shots: [(String, TableController)] = [
            ("a-16-only", a), ("b-16-and-4", b), ("c-loaded-in-use", cc), ("d-not-in-catalog", d), ("e-unlisted-weights", e), ("f-delete-in-flight", f)
        ]
        var notes = ""
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            try await Task.sleep(for: .milliseconds(100))
            for (name, c) in shots {
                await QA.renderTable(c, to: directory.appendingPathComponent("\(name)-\(mode.rawValue).png"))
            }
            notes += "mode \(mode.rawValue) rendered; in flight: \(f.removing.sorted())\n"
        }
        for (name, c) in shots {
            notes += "# \(name)\n"
            for r in c.rows {
                let w = c.weights(r)
                let copies = w.copies.map { "\($0.label) \($0.size) trash=\($0.enabled ? "on" : "off") tip=\($0.help)" }
                notes += "- \(r.id): \(copies.isEmpty ? "no copies" : copies.joined(separator: "; ")); delete=\(w.deleteBlocked ?? "on")\n"
            }
            if c.orphanBytes > 0 { notes += "- unlisted weights: \(formatBytes(c.orphanBytes))\n" }
            notes += "\n"
        }
        try notes.write(to: directory.appendingPathComponent("weights-states.txt"), atomically: true, encoding: .utf8)
        // Let the running delete finish (its 60 s is only a hold for the frames above).
        try await waitUntil(timeout: 90) { f.removing.isEmpty }
    }
}
