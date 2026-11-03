// SPDX-License-Identifier: MIT
import Testing
import ArsenalCore
@testable import Arsenal

/// Old precisions still on this Mac that the catalog no longer offers: the Model hover's line and the settings card's Remove.
@MainActor @Suite struct WithdrawnTierTableTests {
    static func row(withdrawn: [String: Int64]) -> ModelRow {
        var row = ModelRow(id: "model", name: "Model", kind: "chat", precisions: ["bf16"])
        row.withdrawnTiers = withdrawn
        return row
    }

    @Test func oneLeftoverSaysWhatIsOnDiskAndWhatTheModelRunsAt() {
        let row = Self.row(withdrawn: ["int8": 1_200_000_000])
        let line = "An old 8-bit copy (1.2 GB) is still on this Mac \u{2014} this model now runs at 16-bit only."
        #expect(ModelHelp.withdrawn(row) == line)
        #expect(ModelHelp.text(row, presentation: RowPresentation(row)).contains(line))
    }

    @Test func severalLeftoversListEachWithItsSizeLargestFirst() {
        let row = Self.row(withdrawn: ["int4": 600_000_000, "int8": 1_200_000_000])
        #expect(
            ModelHelp.withdrawn(row)
                == "Old copies are still on this Mac: 8-bit (1.2 GB), 4-bit (600 MB) \u{2014} this model now runs at 16-bit only.")
    }

    @Test func noLeftoverAddsNoLineAndNoRemoveFooter() {
        let row = Self.row(withdrawn: [:])
        #expect(ModelHelp.withdrawn(row) == nil)
        #expect(!ModelHelp.text(row, presentation: RowPresentation(row)).contains("still on this Mac"))
        #expect(SettingsPane.height([], loaded: false, withdrawn: false) == SettingsPane.height([], loaded: false))
    }

    @Test func removeFooterAddsItsRowToTheCard() {
        let plain = SettingsPane.height([], loaded: false)
        let withFooter = SettingsPane.height([], loaded: false, withdrawn: true)
        let expected = plain + SettingsPane.M.footerHeight
        #expect(abs(withFooter - expected) < 0.001, "\(withFooter) vs \(expected)")
    }

    @Test func removeHoverSaysWhatGoesAndWhatStays() {
        let help = ModelHelp.withdrawnHelp(Self.row(withdrawn: ["int8": 1_200_000_000]))
        #expect(help.hasPrefix("An old 8-bit copy (1.2 GB) is still on this Mac"))
        #expect(help.hasSuffix("Deletes the old copy from this Mac. The precisions the model offers stay."))
    }
}
