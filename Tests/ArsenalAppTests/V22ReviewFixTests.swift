// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal
import ArsenalCore

/// rv22 review fixes on the app side: Get plans fail closed, the Get pop-up says only what the row and plan know, and the
/// browser's filters pick the same dtype and Fits as the service (`CatalogMatch`).
@MainActor @Suite(.menuState, .serialized) final class V22ReviewFixTests {
    /// rv22-astra B2 (UI half), rv22-opus N1: a plan error used to become an invented weights-only plan, so a small
    /// model that also needed an environment could start downloading with no pop-up. Now nothing starts and the footer
    /// says why.
    /// The table holds only models with weights or a Get under way: Kokoro is on it as a failed Get the user retries.
    static var failedKokoro: [ModelRow] {
        var rows = Fixtures.rows(.firstLaunch)
        rows[rows.firstIndex { $0.id == "kokoro-82m" }!].states["fp16"] = PrecisionState(state: .failed, error: "Get failed.")
        return rows
    }

    @Test func aFailedGetPlanStartsNothingAndSaysWhy() async throws {
        let service = FixtureService(rows: Self.failedKokoro)
        await service.setPlanError("kokoro-82m", WorkerFailure(code: "weights", message: "The weight inventory could not be checked."))
        let c = TableController(service: service)
        await c.refresh()
        var asked = 0
        c.confirmGet = { _, _ in
            asked += 1; return true
        }
        c.perform(c.rows.first { $0.id == "kokoro-82m" }!)
        try await waitUntil { c.lastError != nil }
        #expect(c.lastError == "Kokoro 82M: could not check what Get needs, so nothing was downloaded. The weight inventory could not be checked.")
        try await Task.sleep(for: .milliseconds(100))
        #expect(asked == 0)
        #expect(c.rows.first { $0.id == "kokoro-82m" }!.state == .failed)
        // Once the service can plan again, the same click gets as usual.
        await service.setPlanError("kokoro-82m", nil)
        c.perform(c.rows.first { $0.id == "kokoro-82m" }!)
        try await waitUntil { c.presentation(c.rows.first { $0.id == "kokoro-82m" }!).action == .load }
    }

    private func row(local: Bool = false, formats: [ModelFormat]) -> ModelRow {
        var r = ModelRow(id: "m", name: "M", kind: "chat", precisions: formats.map(\.id), selectedPrecision: formats.first?.id ?? "bf16")
        r.local = local; r.formats = formats
        return r
    }
    private func plan(_ dtype: String, localCopy: Int64 = 0) -> GetPlan {
        GetPlan(
            id: "m", dtype: dtype, runner: "native",
            items: [GetPlanItem(id: "weights", name: "M weights", kind: "weights", downloadBytes: 3_000_000_000, localCopyBytes: localCopy)])
    }

    /// rv22-astra S3, rv22-opus N2: derivation follows the format's source (not its bit width), and "reviewed" is said
    /// only of a signed catalog's pin.
    @Test func theGetPopUpNamesOnlyTheProvenanceItKnows() {
        let formats = [
            ModelFormat(id: "bf16", bitsPerWeight: 16, source: .vendor, runners: ["native"]),
            ModelFormat(id: "mxfp4", bitsPerWeight: 4.25, source: .vendor, runners: ["native"]),
            ModelFormat(id: "int4", bitsPerWeight: 4.5, source: .derived, runners: ["native"]),
            ModelFormat(id: "q3", bitsPerWeight: 3, source: .runnerNative, runners: ["native"])
        ]
        let signed = row(formats: formats), local = row(local: true, formats: formats)
        let reviewed = "The weights come from Hugging Face, pinned to the revision Arsenal reviewed, and are checked by hash."
        let keep = " Arsenal keeps them until you delete them."
        // A vendor quantization is downloaded ready-made: no claim that it is made here.
        #expect(ModelsMenu.weightsProvenance(signed, plan: plan("mxfp4")) == reviewed + keep)
        #expect(ModelsMenu.weightsProvenance(signed, plan: plan("bf16")) == reviewed + keep)
        #expect(ModelsMenu.weightsProvenance(signed, plan: plan("int4")) == reviewed + " int4 is then made from them on this Mac." + keep)
        #expect(ModelsMenu.weightsProvenance(signed, plan: plan("q3")) == reviewed + " The runner turns them into q3 when it loads." + keep)
        // A local folder's pins are its own, not Arsenal's review.
        let unsigned = "The weights come from Hugging Face, pinned to the revision its local folder names (Arsenal did not review it), and are checked by hash."
        #expect(ModelsMenu.weightsProvenance(local, plan: plan("mxfp4")) == unsigned + keep)
        // Part of it already on this Mac: says how much comes from that copy.
        #expect(
            ModelsMenu.weightsProvenance(signed, plan: plan("bf16", localCopy: 1_000_000_000)) == reviewed + " 1.0 GB of them is taken from a copy already on this Mac." + keep)
        // An undeclared format makes no derivation claim either way.
        #expect(ModelsMenu.weightsProvenance(row(formats: []), plan: plan("int4")) == reviewed + keep)
        #expect(ModelsMenu.getAlert(signed, plan: plan("mxfp4")).informativeText == reviewed + keep)
    }

    /// rv22-astra S2, rv22-opus S5: the app compared decimal GB against rounded GiB without headroom. On a 32 GiB Mac
    /// a model needing 33,000 MB fits for the service (33,512 <= 34,359.7 MB) but the browser hid it and drew it red.
    /// Since 22 Oct both compare under the Mac-wide line (100/128 of 34,359.7 MB = 26,843.5 MB) with the 512 MB headroom.
    @Test func fitsAgreesWithTheServiceAtTheMemoryBoundary() {
        let physical: UInt64 = 32 * 1_073_741_824
        func entry(_ id: String, ramMB: Double) -> CatalogEntry {
            let row = ModelRow(id: id, name: id, kind: "chat", precisions: ["bf16"], selectedPrecision: "bf16", peakRAMMB: ramMB, downloadBytes: ["bf16": 30_000_000_000])
            return CatalogEntry(row: row, sizesByDtype: ["bf16": 30_000_000_000], memoryMB: Double(physical) / 1e6)
        }
        let entries = [entry("inside", ramMB: 26_000), entry("edge", ramMB: 26_331), entry("over", ramMB: 26_332)]
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: physical, chip: "M5")
        c.apply(entries: entries, kinds: BrowserTests.kinds)
        c.state.fits = true
        var q = CatalogQuery(); q.fitsRAM = true
        let core = CatalogEntry.query(entries, q).map(\.id)
        #expect(core.sorted() == ["edge", "inside"])
        #expect(c.shown.map(\.id).sorted() == core.sorted())
        for e in entries {
            let item = c.items.first { $0.id == e.id }!
            #expect(item.ramExceeds(physicalBytes: physical) == (e.fitsRAMByDtype["bf16"] != true), "\(e.id)")
        }
        #expect(c.items.first { $0.id == "over" }!.ramHelp(physicalBytes: physical)?.hasSuffix("Needs about 25 GB; Arsenal keeps this 32 GB Mac under 25 GB.") == true)
    }

    /// rv22-opus S3: core ordered candidates by the folder's list (base, int8, int4, then others) and the app by bits,
    /// so where a vendor int4 (4 bits, as `ModelManifest.formats` gives it) and a declared mxfp4 (4.25) both matched,
    /// the browser showed mxfp4 and `arsenal models` int4. Both now take `CatalogMatch`'s order.
    @Test func theMatchedDtypeIsTheServicesWhenTwoFormatsMatch() {
        let sizes: [String: Int64] = ["bf16": 27_000_000_000, "int8": 16_000_000_000, "int4": 7_600_000_000, "mxfp4": 7_400_000_000]
        var row = ModelRow(
            id: "q", name: "Q", kind: "chat", precisions: ["bf16", "int8", "int4", "mxfp4"], selectedPrecision: "bf16", peakRAMMB: 30_000, downloadBytes: sizes)
        row.formats = [
            ModelFormat(id: "bf16", bitsPerWeight: 16, source: .vendor, runners: ["native"]),
            ModelFormat(id: "int8", bitsPerWeight: 8, source: .vendor, runners: ["native"]),
            ModelFormat(id: "int4", bitsPerWeight: 4, source: .vendor, runners: ["native"]),
            ModelFormat(id: "mxfp4", bitsPerWeight: 4.25, source: .vendor, runners: ["native"], storage: "mxfp4")
        ]
        let entries = [CatalogEntry(row: row, sizesByDtype: sizes, memoryMB: 128_000)]
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        c.apply(entries: entries, kinds: BrowserTests.kinds)
        // The browser lists the service's order without the 8-bit formats, which are not offered.
        #expect(c.items[0].dtypes == CatalogMatch.orderedDtypes(entries[0]).filter { !PrecisionColumn.isEightBit($0, formats: row.formats) })
        c.state.toggle(.ram, "8\u{2013}16 GB")
        var q = CatalogQuery(); q.minRAMMB = 8_000; q.maxRAMMB = 15_999
        let core = CatalogEntry.query(entries, q)
        #expect(core.first?.matchingDtype == "mxfp4")
        #expect(c.shown.first?.matchedDtype == core.first?.matchingDtype)
    }

    /// rv22-astra B2: a Get runs under the plan the user saw. When the service finds it needs more (here a local copy
    /// that turned out wrong, so 342 MB must now be downloaded) it returns a revised plan; the app asks about it even
    /// though a small weights-only Get alone would not ask, and only then gets under the revised plan.
    @Test func aRevisedPlanIsAskedAgainBeforeAnythingMoreIsDownloaded() async throws {
        let service = FixtureService(rows: Self.failedKokoro)
        let weights = { (bytes: Int64, local: Int64) in
            GetPlan(
                id: "kokoro-82m", dtype: "fp16", runner: "native",
                items: [GetPlanItem(id: "weights", name: "Kokoro 82M weights", kind: "weights", downloadBytes: bytes, localCopyBytes: local)])
        }
        await service.setPlan(weights(0, 342_000_000))
        await service.setRevision(weights(342_000_000, 0))
        let c = TableController(service: service)
        await c.refresh()
        var asked: [GetPlan] = []
        var answer = false
        c.confirmGet = { _, plan in
            asked.append(plan); return answer
        }
        func kokoro() -> ModelRow { c.rows.first { $0.id == "kokoro-82m" }! }
        // Declined: the first plan needed no question; the revision is asked and refused, so nothing more happens.
        c.perform(kokoro())
        try await waitUntil { !asked.isEmpty }
        try await Task.sleep(for: .milliseconds(100))
        #expect(asked == [weights(342_000_000, 0)])
        #expect(await service.approvedGets == [weights(0, 342_000_000)])
        #expect(kokoro().state == .failed)
        // Accepted: the Get runs again under the revised plan and completes.
        asked = []; answer = true
        await service.setRevision(weights(342_000_000, 0))
        c.perform(kokoro())
        try await waitUntil { c.presentation(kokoro()).action == .load }
        #expect(asked == [weights(342_000_000, 0)])
        #expect(await service.approvedGets.suffix(2) == [weights(0, 342_000_000), weights(342_000_000, 0)])
        #expect(c.lastError == nil)
    }

    /// A part of unknown size bounds nothing (`GetPlan.covers` refuses it), so asking would only loop: one plain error
    /// with Cancel, no Get, no question. Also for a revised plan.
    @Test func anUnknownSizeIsOneErrorAndNoGet() async throws {
        let service = FixtureService(rows: Self.failedKokoro)
        let unknown = GetPlan(
            id: "kokoro-82m", dtype: "fp16", runner: "mlx-audio",
            items: [
                GetPlanItem(id: "weights", name: "Kokoro 82M weights", kind: "weights", downloadBytes: 342_000_000),
                GetPlanItem(id: "mlx-audio", name: "mlx-audio", kind: "library", downloadBytes: nil)
            ])
        await service.setPlan(unknown)
        let c = TableController(service: service)
        await c.refresh()
        var asked = 0
        var errors: [GetPlan] = []
        c.confirmGet = { _, _ in
            asked += 1; return true
        }
        c.unknownGetSize = { _, plan in errors.append(plan) }
        func kokoro() -> ModelRow { c.rows.first { $0.id == "kokoro-82m" }! }
        c.perform(kokoro())
        try await waitUntil { !errors.isEmpty }
        try await Task.sleep(for: .milliseconds(100))
        #expect(errors == [unknown] && asked == 0)
        #expect(await service.approvedGets.isEmpty)
        #expect(kokoro().state == .failed)
        let alert = ModelsMenu.unknownSizeAlert(kokoro(), plan: unknown)
        #expect(alert.messageText == "Arsenal can\u{2019}t tell how much Kokoro 82M needs to download")
        #expect(alert.informativeText == "\u{2022} Kokoro 82M weights \u{00b7} 342 MB\n\u{2022} mlx-audio \u{00b7} size unknown\nNothing was downloaded.")
        #expect(alert.buttons.map(\.title) == ["Cancel"])
        // A revision to an unknown size: the same error, no question, no second Get.
        let known = GetPlan(
            id: "kokoro-82m", dtype: "fp16", runner: "native",
            items: [GetPlanItem(id: "weights", name: "Kokoro 82M weights", kind: "weights", downloadBytes: 0, localCopyBytes: 342_000_000)])
        await service.setPlan(known)
        await service.setRevision(unknown)
        errors = []
        c.perform(kokoro())
        try await waitUntil { !errors.isEmpty }
        try await Task.sleep(for: .milliseconds(100))
        #expect(errors == [unknown] && asked == 0)
        #expect(await service.approvedGets == [known])
        #expect(kokoro().state == .failed)
    }
}
