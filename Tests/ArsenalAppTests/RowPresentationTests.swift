// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) final class RowPresentationTests {
    private func row(_ states: [String: PrecisionState] = [:], fast: Bool = true, badge: ModelBadge = .unverified, precisions: [String] = ["bf16", "int8"]) -> ModelRow {
        ModelRow(
            id: "m", name: "M", kind: "voices", params: "82M", licence: "Apache-2.0", precisions: precisions, states: states,
            selectedPrecision: "bf16", fast: fast, badge: badge, sizeBytes: 4_500_000_000)
    }

    @Test func withdrawnCatalogRowOffersRemoveEvenWhenSelectedDtypeIsNotInstalled() {
        var row = row(["bf16": .init(state: .notInstalled, phase: "catalogRemoved"), "int8": .init(state: .ready, phase: "catalogRemoved")])
        row.verification = "No longer in the catalog"
        let p = RowPresentation(row)
        // Its int8 copy is on this Mac, so the action deletes it: Delete (Remove only takes a model off the list).
        #expect(p.action == .remove && p.actionTitle == "Delete")
        #expect(p.status.text == "No longer in the catalog" && p.status.tone != .error)
        #expect(p.locked && !p.cancellable && !p.pending)
        #expect(ModelTableRow.hasStatusLine(row, p))
        #expect(TableController(service: StubModelsService()).hasWeights(row))
    }

    /// Every model with weights on disk shows in the table, even one an agent runs off the table (its weights
    /// are deleted from here). A model with no weights that is off the table stays out of it.
    @Test func modelsRunningOffTheTableWithWeightsShowInIt() async throws {
        var agent = row(["bf16": .init(state: .ready)]); agent.id = "agent"; agent.name = "Agent"
        var idle = row(); idle.id = "idle"; idle.name = "Idle"
        let service = StubModelsService(rows: [row(["bf16": .init(state: .ready)]), agent, idle])
        try await service.remove(id: "agent")
        try await service.load(id: "agent", precision: "bf16", fast: nil)
        try await service.remove(id: "idle")
        let c = TableController(service: service)
        await c.refresh()
        #expect(c.rows.map(\.id) == ["m", "agent"])
        #expect(c.rows.first { $0.id == "agent" }?.kind != Unlisted.kind)
        await c.refreshRows()
        #expect(c.rows.map(\.id) == ["m", "agent"])
    }

    /// A withdrawn row's Remove is the model's Delete: one question with the size, then every weight goes and the row
    /// leaves the table. Declining keeps everything.
    @Test func withdrawnRowRemoveButtonAsksOnceWithTheSizeThenDeletesEverything() async throws {
        var row = row(["bf16": .init(state: .notInstalled, phase: "catalogRemoved"), "int4": .init(state: .ready, phase: "catalogRemoved")], precisions: ["bf16", "int4"])
        row.verification = "No longer in the catalog"
        let service = StubModelsService(rows: [row])
        let c = TableController(service: service)
        await c.refresh()
        var asked: [Int64] = []
        c.confirmDeleteWeights = { _, bytes in
            asked.append(bytes); return false
        }
        let action = RowActionView()
        action.spec = RowAction(title: "Remove", enabled: true, deletable: false, help: "Remove", onPerform: { c.perform(row) }, onDelete: {})
        #expect(action.accessibilityPerformPress())
        #expect(asked == [4_500_000_000])
        let onDisk = try await service.onDiskModels()
        #expect(c.rows.count == 1 && onDisk.count == 1)
        c.confirmDeleteWeights = { _, bytes in
            asked.append(bytes); return true
        }
        #expect(action.accessibilityPerformPress())
        try await waitUntil { c.rows.isEmpty }
        #expect(asked.count == 2)
        let remaining = try await service.onDiskModels()
        #expect(remaining.isEmpty)
        #expect(c.lastError == nil)
    }

    @Test func notInstalledGetShowsTheDownloadSizeBeforeStarting() {
        let p = RowPresentation(row())
        #expect(p.action == .get)
        #expect(p.actionTitle == "Get \u{00b7} 4.5 GB")
        #expect(p.busyText == nil)
        #expect(!(p.locked))
    }

    @Test func derivedPrecisionDownloadsOnlyWhenTheSixteenBitIsMissing() {
        // The derived cell is int4 (int8 has no cell).
        #expect(RowPresentation(row(precisions: ["bf16", "int4"]), previewPrecision: "int4").downloadBytes == 4_500_000_000)
        let have16 = RowPresentation(row(["bf16": .init(state: .ready)], precisions: ["bf16", "int4"]), previewPrecision: "int4")
        #expect(have16.downloadBytes == 0)
        #expect(have16.actionTitle == "Get") // made here, nothing to download
    }

    @Test func downloadingShowsPercentBytesAndCancel() {
        let p = RowPresentation(row(["bf16": .init(state: .downloading, progress: 0.237, bytes: 1_070_000_000, totalBytes: 4_500_000_000)]))
        #expect(p.busyText == "23%")
        #expect(p.cancellable)
        #expect(p.progress == 0.237)
        #expect(p.status.text == "Downloading \u{00b7} 1.1 of 4.5 GB")
        #expect(p.locked)
    }

    @Test func derivingAndLoadingAreBusyWithoutCancel() {
        let deriving = RowPresentation(row(["bf16": .init(state: .ready), "int8": .init(state: .deriving)]), previewPrecision: "int8")
        #expect(deriving.busyText == "\u{2026}")
        #expect(!(deriving.cancellable))
        #expect(deriving.status.text == "Making 8-bit\u{2026}")
        let loading = RowPresentation(row(["bf16": .init(state: .loading)]))
        #expect(loading.status.text == "Loading 16-bit\u{2026}")
        #expect(loading.locked)
    }

    @Test func aBusyPrecisionShowsItsProgressWhateverCellIsSelected() {
        let r = row(["int8": .init(state: .downloading, progress: 0.5)])
        let p = RowPresentation(r)
        #expect(p.selected == "bf16")
        #expect(p.busyText == "50%")
        #expect(p.cancellable)
    }

    @Test func loadedUnloadsAndAnotherChoiceReloads() {
        let loaded = row(["bf16": .init(state: .loaded), "int4": .init(state: .ready)], precisions: ["bf16", "int4"])
        let p = RowPresentation(loaded)
        #expect(p.action == .unload)
        #expect(p.loadedPrecision == "bf16")
        #expect(p.hot)
        #expect(RowPresentation(loaded, previewPrecision: "int4").action == .reload)
        #expect(RowPresentation(loaded, previewFast: false).action == .reload)
        #expect(RowPresentation(loaded, previewPrecision: "int4").actionTitle == "Reload")
        // An 8-bit choice names no cell here: nothing is pending.
        #expect(RowPresentation(loaded, previewPrecision: "int8").action == .unload)
    }

    @Test func readyLoads() {
        #expect(RowPresentation(row(["bf16": .init(state: .ready)])).action == .load)
    }

    @Test func failureShowsItsMessageAndOffersGetAgain() {
        let p = RowPresentation(row(["bf16": .init(state: .failed, error: "Get failed: stalled.")]))
        #expect(p.action == .get)
        #expect(p.status == StatusLine(text: "Get failed: stalled.", tone: .error, help: "Get failed: stalled."))
    }

    // Round 1 (stub-table-get-failed): "Get failed: Hugging Face sto…" was cut beside Try Again. The reason reads in
    // plain words under its lead, up to two lines as a purpose takes, inside the name column and the row's height; the
    // hover keeps the whole message.
    @Test func aFailedGetSaysItsReasonWholeBesideTryAgain() {
        let message = "Get failed: Hugging Face stopped sending data for 2 minutes. Get resumes from 6.1 GB."
        let p = RowPresentation(row(["bf16": .init(state: .failed, error: message)]))
        #expect(p.line == StatusLine(text: "Get failed: the connection stalled", tone: .error, help: message))
        #expect(ModelTableRow.purposeLines(p.line) == 2)
        let width = ModelTable.W.model - ModelTable.W.icon - ModelTable.W.iconGap
        let font = NSFont.systemFont(ofSize: ModelTable.deltaSize, weight: .medium)
        let line = ceil(font.ascender - font.descender + font.leading)
        let height = (p.line.text as NSString).boundingRect(with: NSSize(width: width, height: 1000), options: [.usesLineFragmentOrigin], attributes: [.font: font]).height
        #expect(height <= 2 * line + 0.5, "\(p.line.text) needs \(height) pt at \(width) pt")
        #expect(ErrorWording.plain("The Internet connection appears to be offline.") == "this Mac is offline")
        // A reason with no plain wording shows its first sentence; a decimal ("6.1 GB") is not a sentence's end.
        let other = RowPresentation(row(["bf16": .init(state: .failed, error: "Hash mismatch on model.safetensors. Get again to replace it.")]))
        #expect(other.line.text == "Hash mismatch on model.safetensors." && other.line.help.hasSuffix("replace it."))
        #expect(ErrorWording.lead("Get failed: 6.1 GB of 9.0 GB.") == "Get failed: 6.1 GB of 9.0 GB.")
    }

    @Test func badgesExplainThemselvesAndUnverifiedSaysNotTried() {
        // Its hover names the check of each offered cell: 16-bit, then int4 (int8 has no cell).
        let unverified = RowPresentation(row(precisions: ["bf16", "int4"])).status
        #expect(unverified.text == "Unverified")
        #expect(unverified.help == "bf16 has no quality check yet. int4 has no quality check yet. No figures yet.")
        let measured = RowPresentation(row(badge: .measured), chip: "M5 Max").status
        #expect(measured.tone == .measured)
        #expect(measured.help.contains("M5 Max"))
        let other = RowPresentation(row(badge: .otherChip)).status
        #expect(other.text == "Measured on another chip")
        #expect(other.help.contains("another chip"))
    }

    @Test func aPreviewOfAPrecisionTheModelDoesNotOfferIsIgnored() {
        #expect(RowPresentation(row(), previewPrecision: "int4").selected == "bf16")
    }

    @Test func precisionColumnsLabelTheDTypeThatRuns() {
        let fp16 = ModelRow(id: "r", name: "R", kind: "voices", precisions: ["fp16"], selectedPrecision: "fp16")
        #expect(PrecisionColumn.all.map { PrecisionColumn.label(fp16, $0) } == ["fp16", "int4"])
        let bf16 = ModelRow(id: "b", name: "B", kind: "images", precisions: ["bf16", "int8", "int4"], selectedPrecision: "bf16")
        #expect(PrecisionColumn.all.map { PrecisionColumn.label(bf16, $0) } == ["bf16", "int4"])
        // Two cells only: 16-bit and 4-bit. The model with no 4-bit format greys its one quantized cell.
        #expect(Set(ModelHelp.unavailable(fp16).keys) == ["4"])
    }
}

@MainActor @Suite(.menuState, .serialized) final class MetricTextTests {
    private let speed = KindColumn(id: "speed", title: "Speed", unit: "×")

    @Test func stringsFromResultsAreShownAndSplitAtTheDot() {
        var r = ModelRow(id: "f", name: "F", kind: "images", badge: .measured)
        r.metrics = ["seconds_per_image": "4.1 s \u{00b7} 1024\u{00d7}1024"]
        let m = MetricText(r, KindColumn(id: "seconds_per_image", title: "Seconds per image", unit: "s"))
        #expect(m.value == "4.1 s")
        #expect(m.detail == "1024\u{00d7}1024")
        #expect(!(m.estimate))
    }

    @Test func anotherChipIsAnEstimateAndUnverifiedIsADash() {
        var r = ModelRow(id: "k", name: "K", kind: "voices", badge: .otherChip, metrics: ["speed": 42.0])
        #expect(MetricText(r, speed).value == "~42×")
        #expect(MetricText(r, speed).estimate)
        r.badge = .unverified
        #expect(MetricText(r, speed).value == "\u{2014}")
        r.badge = .measured; r.metrics = [:]
        #expect(MetricText(r, speed).value == "\u{2014}")
    }

    @Test func peakRAMComesLastOnceAndFromTheRowWhenNoMetric() {
        let kind = KindSpec(id: "chat", title: "Chat", order: 3, columns: [KindColumn(id: "context", title: "Context", unit: "tokens")])
        #expect(MetricText.columns(kind).map(\.id) == ["context", "peak_ram"])
        let r = ModelRow(id: "c", name: "C", kind: "chat", badge: .measured, metrics: ["context": 262_144], peakRAMMB: 8_600)
        #expect(MetricText(r, KindColumn(id: "context", title: "Context", unit: "tokens")).value == "256K")
        #expect(MetricText(r, MetricText.peakRAM).value == "8.01 GB")
    }

    @Test func quickCheckFiguresShowLighterWithoutABadge() {
        let r = ModelRow(
            id: "l", name: "L", kind: "decisions", badge: .unverified, metrics: ["accuracy": "90.9%"],
            verification: "quick check", quickMetrics: ["accuracy": true])
        let accuracy = KindColumn(id: "accuracy", title: "Accuracy", unit: "%")
        let m = MetricText(r, accuracy)
        #expect(m.value == "90.9%")
        #expect(m.estimate)
        let status = RowPresentation(r).status
        #expect(status.text == "Quick check")
        #expect(status.help == QuickCheck.help && QuickCheck.help.contains("uncontrolled"))
        #expect(ColumnHelp.cell(r, accuracy, chip: "M5 Max").contains(QuickCheck.help))
    }

    @Test func noFastPathPinsTheSwitchAtExact() {
        let flux = Fixtures.base.first { $0.id == "flux2-klein-4b" }!
        #expect(!flux.fastAvailable)
        // Fast kernels in the folder make Fast available even for a runner without its own Fast mode.
        var kernels = flux; kernels.hasFastKernels = true
        #expect(kernels.fastAvailable)
        // Without a Fast path the row runs Exact whatever was saved, and that is never a pending change.
        var loaded = flux; loaded.states["bf16"] = .init(state: .loaded); loaded.fast = true
        let p = RowPresentation(loaded)
        #expect(!p.fast && !p.pending && p.action == .unload)
        #expect(Fixtures.base.first { $0.id == "kokoro-82m" }!.fastAvailable)
    }

    @Test func onlyTheQuickCheckFiguresAreLighter() {
        let r = ModelRow(
            id: "v", name: "V", kind: "decisions", badge: .unverified, metrics: ["accuracy": "88.0%", "latency": "12 ms"],
            verification: "quick check", quickMetrics: ["accuracy": true])
        #expect(MetricText(r, KindColumn(id: "accuracy", title: "Accuracy", unit: "%")).estimate)
        #expect(MetricText(r, KindColumn(id: "latency", title: "Latency", unit: "ms")).value == "\u{2014}")
    }

    @Test func aCellNamesWhatItsGetStillDownloads() {
        let r = ModelRow(
            id: "m", name: "M", kind: "images", precisions: ["bf16", "int8"], states: ["int8": .init(state: .ready)],
            downloadBytes: ["bf16": 15_980_140_843, "int8": 0])
        let p = RowPresentation(r)
        #expect(ModelHelp.precision(r, column: "16", presentation: p).hasSuffix("Get downloads 16.0 GB"))
        #expect(!ModelHelp.precision(r, column: "8", presentation: p).contains("Get downloads"))
        #expect(p.actionTitle == "Get \u{00b7} 16.0 GB")
    }

    @Test func aChatModelsContextShowsBeforeAnyMeasurement() {
        let r = ModelRow(id: "q", name: "Q", kind: "chat", badge: .unverified, metrics: ["context": 262_144, "tokens_per_s": 30])
        #expect(MetricText(r, KindColumn(id: "context", title: "Context", unit: "tokens")).value == "256K")
        #expect(MetricText(r, KindColumn(id: "tokens_per_s", title: "Tokens/s", unit: "tokens/s")).value == "\u{2014}")
        #expect(ColumnHelp.cell(r, KindColumn(id: "context", title: "Context", unit: "tokens"), chip: nil).hasSuffix("Arsenal runs up to 256K tokens."))
    }

    /// The vendor's larger maximum (`metrics["native_context"]`) follows what Arsenal runs; an equal one adds nothing.
    @Test func contextTooltipNamesTheModelsOwnMaximum() {
        let column = KindColumn(id: "context", title: "Context", unit: "tokens")
        let r = ModelRow(id: "q", name: "Q", kind: "chat", metrics: ["context": 32_768, "native_context": 262_144])
        #expect(ColumnHelp.cell(r, column, chip: nil).hasSuffix("Arsenal runs up to 32K tokens; the model supports 256K."))
        #expect(MetricText(r, column).value == "32K")
        let same = ModelRow(id: "q", name: "Q", kind: "chat", metrics: ["context": 32_768, "native_context": 32_768])
        #expect(ColumnHelp.cell(same, column, chip: nil).hasSuffix("Arsenal runs up to 32K tokens."))
    }

    @Test func unitsFormat() {
        #expect(MetricText.format(0.18, unit: "s", id: "x") == "180 ms")
        #expect(MetricText.format(4.12, unit: "s", id: "x") == "4.1 s")
        #expect(MetricText.format(0.914, unit: "%", id: "x") == "91.4%")
        #expect(MetricText.format(96, unit: "tokens/s", id: "x") == "96.0")
        #expect(MetricText.format(1.3, unit: "GB", id: "peak_ram") == "1.21 GB")
    }
}

/// Local copy, verification of it, and the released 16-bit source: what the row, button and tooltips say.
@MainActor @Suite(.menuState) final class LocalCopyAndRetentionTests {
    private func qwen(_ scene: Fixtures.Scene) -> ModelRow { Fixtures.rows(scene).first { $0.id == "qwen3.8-27b" }! }

    @Test func aLocalCopyMakesGetSayItDownloadsNothing() {
        let row = qwen(.localCopy)
        let p = RowPresentation(row)
        #expect(p.action == .get && p.getUsesLocalCopy)
        // Nothing to download: a plain Get, its tooltip saying where the weights come from.
        #expect(p.actionTitle == "Get")
        let help = ModelHelp.action(row, presentation: p)
        #expect(help == "Adds the copy already on this Mac (55.6 GB). Nothing to download.")
        #expect(ModelHelp.precision(row, column: "16", presentation: p).hasSuffix("Get uses the copy on this Mac"))
        // Without a local copy, a zero download is a plain Get (made here from what is installed).
        var plain = row; plain.localCopyBytes = [:]
        #expect(RowPresentation(plain).actionTitle == "Get")
        // A partial copy still downloads the rest: the size wins.
        var partial = row; partial.downloadBytes["bf16"] = 1_200_000_000
        #expect(RowPresentation(partial).actionTitle == "Get \u{00b7} 1.2 GB")
    }

    @Test func verifyingTheLocalCopyIsBusyAndCancellable() {
        let row = qwen(.verifying)
        let p = RowPresentation(row)
        #expect(p.status.text == "Checking local copy")
        #expect(p.status.help == "Checking the hashes of the copy on this Mac (22.8 GB of 55.6 GB); nothing is downloaded.")
        #expect(p.busyText == "41%" && p.cancellable && p.locked)
        #expect(ModelHelp.action(row, presentation: p) == "Checking the copy on this Mac")
        #expect(ModelHelp.precision(row, column: "16", presentation: p).contains("\nChecking the copy on this Mac"))
        // The same state without the phase is an ordinary download.
        var downloading = row; downloading.states["bf16"]?.phase = "downloading"
        #expect(RowPresentation(downloading).status.text.hasPrefix("Downloading \u{00b7}"))
    }

    @Test func theModelTooltipSaysTheSourceWasRemovedOnce() {
        let row = qwen(.retention)
        let p = RowPresentation(row)
        let tooltip = ModelHelp.text(row, presentation: p)
        let note = "16-bit source removed to save 55.6 GB; Get re-downloads it"
        #expect(tooltip.hasSuffix("\n" + note))
        #expect(tooltip.components(separatedBy: note).count == 2)
        #expect(p.action == .load)
        let sixteen = RowPresentation(row, previewPrecision: "bf16")
        #expect(sixteen.actionTitle == "Get \u{00b7} 55.6 GB")
        #expect(!ModelHelp.text(qwen(.firstLaunch), presentation: RowPresentation(qwen(.firstLaunch))).contains("removed"))
    }

    @Test func wholeGetBytesDriveTheRowText() {
        let row = ModelRow(
            id: "kokoro-82m", name: "Kokoro 82M", kind: "voices", precisions: ["fp16"],
            states: ["fp16": .init(state: .downloading, progress: 0.5, bytes: 2_000_000, totalBytes: 4_000_000, bytesPerSecond: 1_000_000)], selectedPrecision: "fp16")
        let p = RowPresentation(row)
        #expect(p.status.text == "Downloading \u{00b7} 2 of 4 MB")
        #expect(p.busyText == "50%" && p.cancellable)
    }
}

@MainActor @Test func aOneModeRunnerTooltipNamesNoFastOrExact() {
    let loaded = Fixtures.rows(.loaded)
    let qwen = loaded.first { $0.id == "qwen3.8-27b" }!, kokoro = loaded.first { $0.id == "kokoro-82m" }!
    // v2: Exact is always a mode; a runner without Fast is loaded at Exact.
    // The loaded precision is the selected segment, not tooltip text; the name shows only when the cell truncates it.
    #expect(!ModelHelp.text(qwen, presentation: RowPresentation(qwen)).contains("Loaded at"))
    #expect(!ModelHelp.text(kokoro, presentation: RowPresentation(kokoro)).hasPrefix(kokoro.name))
    #expect(ModelHelp.text(kokoro, presentation: RowPresentation(kokoro), name: true).hasPrefix(kokoro.name + "\n"))
    #expect(
        ModelHelp.action(kokoro, presentation: RowPresentation(kokoro, previewFast: false), changes: ["Exact"])
            == "Reload applies: Exact. Until then the loaded settings serve requests.")
}
