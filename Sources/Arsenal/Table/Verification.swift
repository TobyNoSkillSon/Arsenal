// SPDX-License-Identifier: MIT
import ArsenalCore

// The results file's verification label as the row carries it (`ModelRow.verification`: "Measured", "quick check",
// "Not tried on this Mac yet", …) and which figures are quick-check ones (`quickMetrics`).
extension ModelRow {
    /// Figures from a short uncontrolled run: shown lighter, no badge, never ranked.
    var isQuickCheck: Bool { verification == "quick check" || quickMetrics.values.contains(true) }
    /// One figure from a quick check (`quickMetrics` names them; a measured row has none).
    func isQuickCheck(_ column: String) -> Bool {
        badge != .measured && (quickMetrics[column] == true || (quickMetrics.isEmpty && verification == "quick check"))
    }
}

enum QuickCheck {
    static let title = "Quick check"
    static let help = "A short uncontrolled run on this Mac: rough figures, never ranked."
}
