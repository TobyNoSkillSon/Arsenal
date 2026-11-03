// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// Native runner changes reload installed weights; Get is for missing weights and assets.
@MainActor @Suite(.menuState, .serialized) final class V21TableTests {
    private func row(state: ModelState, resident: String? = nil, runner: String = "mlx-audio") -> ModelRow {
        var r = ModelRow(
            id: "k", name: "Kokoro 82M", kind: "voices", precisions: ["fp16", "int8"], selectedPrecision: "fp16", runner: runner,
            downloadBytes: ["fp16": 342_000_000], runnerLanguage: "swift")
        r.states["fp16"] = .init(state: state); r.defaultRunner = "native"; r.residentRunner = resident
        if resident != nil, resident != runner { r.needsReload = true }
        return r
    }

    @Test func nativeRunnersLoadAndReloadWhileMissingWeightsUseGet() {
        let available = row(state: .ready)
        let ready = RowPresentation(available)
        #expect(ready.action == .load && ready.actionTitle == "Load" && !ready.pending)
        #expect(ModelHelp.action(available, presentation: ready) == "Loads 16-bit")
        let loaded = row(state: .loaded, resident: "native")
        let p = RowPresentation(loaded)
        #expect(p.pending && p.action == .reload && p.actionTitle == "Reload")
        #expect(ModelHelp.action(loaded, presentation: p) == "Reloads at 16-bit")
        let fresh = row(state: .notInstalled)
        let f = RowPresentation(fresh)
        #expect(f.action == .get && f.actionTitle == "Get \u{00b7} 342 MB")
        #expect(ModelHelp.action(fresh, presentation: f) == "Downloads 16-bit from Hugging Face.")
    }

    @Test func theRunnerRowShowsTheOnlyRunnerAndOffersEligibleAlternatives() throws {
        let one = [CatalogRunner(name: "native", language: "swift", engine: "voices", acceptedDefault: true, selected: true)]
        #expect(TableController.runnerSetting(row(state: .ready, runner: "native"), one)?.choices == ["native"])
        let two = one + [CatalogRunner(name: "mlx-audio", language: "swift", engine: "voices"), CatalogRunner(name: "far", language: "swift", engine: "x", eligible: false)]
        let s = try #require(TableController.runnerSetting(row(state: .loaded, resident: "native"), two))
        #expect(s.id == TableController.runnerKey && s.title == "Runner" && s.applies == "load" && s.choices == ["native", "mlx-audio"])
        #expect(s.value == "mlx-audio" && s.defaultValue == "native" && s.pending && s.control == .list)
    }

    @Test func theRunnerRowSetsTheRunnerAndRedRowsNameIt() async throws {
        // On the table: its weights are on this Mac.
        var r = ModelRow(
            id: "k", name: "Kokoro 82M", kind: "voices", precisions: ["fp16"], states: ["fp16": PrecisionState(state: .ready)], selectedPrecision: "fp16")
        r.defaultRunner = "native"
        let service = FixtureService(rows: [r], metadata: ["k": ["runner": .array([["name": "native", "language": "swift"], ["name": "alt", "language": "swift"]])]])
        let c = TableController(service: service, chip: "M5 Max")
        await c.refresh()
        #expect(c.settings["k"]?.map(\.id).prefix(2) == [TableController.keepKey, TableController.runnerKey])
        c.set(c.rows[0], TableController.runnerKey, "alt")
        try await waitUntil { await service.calls.contains("runner k alt") }
        c.set(c.rows[0], TableController.runnerKey, "native")
        try await waitUntil { await service.calls.contains("runner k default") }
        var red = r; red.runner = "alt"; red.residentRunner = "native"; red.states["fp16"] = .init(state: .loaded); red.needsReload = true
        c.apply(rows: [red], kinds: Fixtures.kinds)
        #expect(c.pendingChanges(red).first == "runner alt")
    }

    @Test func theRunnerTooltipNamesAChosenRunnersDefault() {
        var r = ModelRow(id: "k", name: "K", kind: "voices", runner: "native", runnerLanguage: "swift")
        #expect(ModelHelp.runnerTooltip(r) == "Runner: Native (Swift \u{00b7} MLX)")
        r.defaultRunner = "native"; r.runner = "mlx-audio"; r.runnerLanguage = "swift"
        #expect(ModelHelp.runnerTooltip(r) == "Runner: mlx-audio (Swift \u{00b7} MLX), chosen; default native")
    }

    @Test func thePaletteKeepsItsContrast() {
        func luminance(_ c: NSColor) -> CGFloat {
            let s = c.usingColorSpace(.sRGB)!
            func ch(_ v: CGFloat) -> CGFloat { v <= 0.03928 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4) }
            return 0.2126 * ch(s.redComponent) + 0.7152 * ch(s.greenComponent) + 0.0722 * ch(s.blueComponent)
        }
        func ratio(_ a: NSColor, _ b: NSColor) -> CGFloat { let x = luminance(a), y = luminance(b); return (max(x, y) + 0.05) / (min(x, y) + 0.05) }
        for p in [ArsenalPalette.dark, .light] {
            #expect(ratio(p.t1, p.canvas) >= 15 && ratio(p.t2, p.canvas) >= 7)
        }
    }
}
