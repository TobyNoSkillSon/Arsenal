// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// v2.2 items 4–5: dtype, Size, RAM and Fits hold at one offered dtype together, the row then reads at that dtype, and
/// any format label a folder declares is listed, ordered and filterable. Checked against core's own catalog query.
@MainActor @Suite(.menuState, .serialized) final class V22FilterTests {
    static let gb: Int64 = 1_000_000_000
    /// A 27B chat model: bf16 54 GB (the download), int8/int4 made here; the vendor's mxfp4 is its own 14 GB download.
    static func big() -> CatalogEntry {
        var row = ModelRow(
            id: "big", name: "Big 27B", kind: "chat", params: "27B", precisions: ["bf16", "int8", "int4", "mxfp4"], selectedPrecision: "bf16", peakRAMMB: 58_000,
            downloadBytes: ["bf16": 54 * gb, "int8": 54 * gb, "int4": 54 * gb, "mxfp4": 14 * gb])
        row.sizeBytes = 54 * gb
        var e = CatalogEntry(
            row: row, metadata: ["derive": ["int8": "mlx-affine-g64", "int4": "mlx-affine-g64"], "weights": ["dtype": "bf16"]],
            sizesByDtype: ["bf16": 54 * gb, "int8": 28 * gb, "int4": 15 * gb, "mxfp4": 14 * gb], memoryMB: 128_000)
        e.ramMBByDtype = ["bf16": 58_000, "int8": 30_000, "int4": 16_500, "mxfp4": 15_500]
        e.formats = [
            ModelFormat(id: "bf16", bitsPerWeight: 16, source: .vendor, runners: ["native"]),
            ModelFormat(id: "int8", bitsPerWeight: 8.5, source: .derived, runners: ["native"]),
            ModelFormat(id: "int4", bitsPerWeight: 4.5, source: .derived, runners: ["native"]),
            ModelFormat(id: "mxfp4", bitsPerWeight: 4.25, source: .vendor, runners: [], storage: "mxfp4")
        ]
        return e
    }
    static func small() -> CatalogEntry {
        let row = ModelRow(
            id: "small", name: "Small 82M", kind: "voices", params: "82M", precisions: ["fp16", "int8"], selectedPrecision: "fp16", peakRAMMB: 800,
            downloadBytes: ["fp16": 342_000_000, "int8": 342_000_000])
        var e = CatalogEntry(row: row, metadata: ["derive": ["int8": "mlx-affine-g64"]], sizesByDtype: ["fp16": 171_000_000, "int8": 95_000_000], memoryMB: 128_000)
        e.ramMBByDtype = ["fp16": 800, "int8": 600]
        return e
    }
    /// The entry with its 8-bit formats removed, as the browser offers it (its dtypes, sizes, RAM and formats).
    static func offered(_ e: CatalogEntry) -> CatalogEntry {
        var c = e
        let keep: (String) -> Bool = { !PrecisionColumn.isEightBit($0, formats: e.formats) }
        c.row.precisions = c.row.precisions.filter(keep)
        c.sizesByDtype = c.sizesByDtype.filter { keep($0.key) }
        c.ramMBByDtype = c.ramMBByDtype.filter { keep($0.key) }
        c.fitsRAMByDtype = c.fitsRAMByDtype.filter { keep($0.key) }
        c.formats = c.formats.filter { keep($0.id) }
        return c
    }
    private func controller(_ entries: [CatalogEntry] = [big(), small()]) -> BrowserController {
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        c.apply(entries: entries, kinds: BrowserTests.kinds)
        return c
    }

    @Test func anyDeclaredFormatIsListedWidestFirstAndOfferedAsAFilter() {
        let c = controller()
        let big = c.items.first { $0.id == "big" }!
        // 8-bit is not listed for any model: the list and the filter hold the 16-bit and int4 formats only.
        #expect(big.dtypes == ["bf16", "int4", "mxfp4"])
        #expect(big.dtypeText == "bf16 \u{00b7} int4 \u{00b7} mxfp4")
        #expect(BrowserFacet.dtype.values(c.items) == ["bf16", "fp16", "int4", "mxfp4"])
        #expect(!c.items.contains { $0.dtypes.contains("int8") })
        // 8-bit has no Precision column: fp8 is named as 8-bit, never placed.
        #expect(PrecisionColumn.of("mxfp4") == "4" && PrecisionColumn.of("q3_k") == "4" && PrecisionColumn.of("fp32") == nil)
        #expect(PrecisionColumn.of("fp8") == nil && PrecisionColumn.isEightBit("fp8"))
    }

    /// A saved 8-bit choice is the 16-bit copy, both as the default and as the row's dtype; no 8-bit dtype is listed.
    @Test func aSavedEightBitChoiceDefaultsToSixteenBitAndIsNotListed() {
        var row = ModelRow(id: "eight", name: "Eight", kind: "voices", params: "1B", precisions: ["bf16", "int8", "int4"], selectedPrecision: "int8")
        row.states["bf16"] = PrecisionState(state: .ready)
        let item = BrowserItem(CatalogEntry(row: row, metadata: [:], added: true), kind: nil)
        #expect(item.dtypes == ["bf16", "int4"])
        #expect(item.defaultDtype == "bf16")
        #expect(!item.dtypes.contains("int8"))
    }

    @Test func int4AndOneToEightGBIsOneConstraintAtOneDtype() {
        let c = controller()
        // At its default bf16 the big model weighs 54 GB; at int4 15 GB: 8–32 GB matches only through int4.
        c.state.toggle(.dtype, "int4"); c.state.toggle(.size, "8\u{2013}32 GB")
        #expect(c.shown.map(\.id) == ["big"])
        let row = c.shown[0]
        #expect(row.matchedDtype == "int4" && row.ramGB == 16.5 && row.downloadBytes == 54 * Self.gb)
        // No peak memory measured at int4: the table's triangle, never the service's estimate (Fits still uses it).
        #expect(row.ramText.isEmpty && row.peakHelp.contains("Not measured"))
        // int4 and 1–8 GB: neither model's int4 is in range (small has no int4).
        c.state.values[.size] = ["1\u{2013}8 GB"]
        #expect(c.shown.isEmpty)
        // A size alone looks at every format the browser offers: int4 (15 GB) puts the big model into 8–32 GB, first of
        // the two that match (its 8-bit copy is not offered).
        c.state.values = [.size: ["8\u{2013}32 GB"]]
        #expect(c.shown.map(\.id) == ["big"] && c.shown[0].matchedDtype == "int4")
        // RAM < 8 GB: the small model at its default dtype.
        c.state.values = [.ram: ["< 8 GB"]]
        #expect(c.shown.map(\.id) == ["small"] && c.shown[0].matchedDtype == "fp16")
    }

    @Test func fitsAloneProjectsOnlyWhenTheDefaultDoesNotFit() {
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 32 * 1_073_741_824, chip: "M5")
        c.apply(entries: [Self.big(), Self.small()], kinds: BrowserTests.kinds)
        c.state.fits = true
        let big = c.shown.first { $0.id == "big" }
        // 22 Oct: Fits compares under the Mac-wide line (25 GiB of 32), so int8's 30 GB no longer fits; int4 does.
        #expect(big?.matchedDtype == "int4" && big?.ramGB == 16.5)
        #expect(c.shown.first { $0.id == "small" }?.matchedDtype == nil)
    }

    @Test func theAppAgreesWithCoresCatalogQuery() {
        let entries = [Self.big(), Self.small()]
        let c = controller(entries)
        let cases: [(dtype: String?, size: String?, ram: String?, query: CatalogQuery)] = [
            (
                "int4", "8\u{2013}32 GB", nil,
                {
                    var q = CatalogQuery(); q.dtypes = ["int4"]; q.minSizeBytes = 8 * Self.gb; q.maxSizeBytes = 32 * Self.gb; return q
                }()
            ),
            (
                "int8", nil, "16\u{2013}32 GB",
                {
                    var q = CatalogQuery(); q.dtypes = ["int8"]; q.minRAMMB = 16_000; q.maxRAMMB = 31_999; return q
                }()
            ),
            (
                nil, "< 1 GB", nil,
                {
                    var q = CatalogQuery(); q.maxSizeBytes = Self.gb - 1; return q
                }()
            ),
            (
                "mxfp4", nil, nil,
                {
                    var q = CatalogQuery(); q.dtypes = ["mxfp4"]; return q
                }()
            )
        ]
        for t in cases {
            c.state.values = [:]
            if let d = t.dtype { c.state.toggle(.dtype, d) }
            if let s = t.size { c.state.toggle(.size, s) }
            if let r = t.ram { c.state.toggle(.ram, r) }
            // Core's query over the formats the browser offers: the 8-bit formats are not offered.
            let core = CatalogEntry.query(entries.map(Self.offered), t.query)
            #expect(Set(c.shown.map(\.id)) == Set(core.map(\.id)), "\(t)")
            for e in core { #expect(c.shown.first { $0.id == e.id }?.matchedDtype == e.matchingDtype, "\(t) \(e.id)") }
        }
    }

    @Test func aFormatWithoutARunnerIsGreyedInTheTableWithThatReason() {
        var row = Self.big().row
        row.runner = "native"; row.formats = Self.big().formats
        let off = ModelHelp.unavailable(row)
        // int4 (derived, native runs it) is the 4-bit cell unless mxfp4 is selected.
        #expect(off.isEmpty)
        row.selectedPrecision = "mxfp4"
        #expect(PrecisionColumn.dtype(row, "4") == "mxfp4")
        let greyed = ModelHelp.unavailable(row)
        #expect(greyed["4"] == "Unavailable. No runner in this Arsenal yet.")
        // The browser's detail says the same for that format.
        var entry = Self.big(); entry.row = row
        #expect(ModelDetail.sizes(entry).first { $0.dtype == "mxfp4" }?.note == "no runner in this Arsenal yet")
        #expect(ModelHelp.origin("mxfp4", native: "bf16", format: Self.big().formats[3]) == "mxfp4, a smaller version from the publisher")
    }
}
