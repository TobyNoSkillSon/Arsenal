// SPDX-License-Identifier: MIT
import ArsenalCore
import Testing

@testable import Arsenal

/// 25 Oct, installed build 1420: The user's table membership was empty, so every installed catalog model was listed under
/// "Not in catalog" with a Remove that deleted its weights. An installed catalog model off the table keeps its catalog row.
@Suite struct UnlistedRowsTests {
    static func disk(_ id: String) -> OnDiskModel {
        OnDiskModel(id: id, precisions: [OnDiskPrecision(dtype: "fp16", bytes: 10, inUse: false, dependents: [])], cacheBytes: 0, offered: true)
    }

    @Test func anInstalledCatalogModelOffTheTableKeepsItsCatalogRow() {
        let laya = ModelRow(id: "laya-english", name: "Laya · English", kind: "decisions")
        let rows = Unlisted.rows([], disk: ["laya-english": Self.disk("laya-english"), "gone": Self.disk("gone")], catalog: [laya])
        #expect(rows.map(\.id) == ["laya-english", "gone"])
        #expect(rows[0].kind == "decisions" && rows[0].name == "Laya · English")
        #expect(rows[1].kind == Unlisted.kind)
    }

    @Test func aCatalogModelWithNothingOnDiskStaysOffTheTable() {
        let rows = Unlisted.rows([], disk: [:], catalog: [ModelRow(id: "flux2-klein-4b", name: "FLUX.2", kind: "images")])
        #expect(rows.isEmpty)
    }
}
