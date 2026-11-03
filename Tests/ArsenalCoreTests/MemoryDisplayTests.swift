// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Mac memory on screen is Activity Monitor's binary GB (GiB); the wire and stored figures stay decimal GB.
@Suite struct MemoryDisplayTests {
    /// 128 GiB and 100 GiB (the 100/128 line) in decimal GB, as the probe and the wire carry them.
    static let total = 137.438953472, limit = 107.374182400

    @Test func theLineOnThisMacReadsAsActivityMonitorsGB() {
        #expect(MemoryDisplay.text(decimalGB: Self.limit) == "100 GB")
        #expect(MemoryDisplay.text(decimalGB: Self.total) == "128 GB")
        #expect(abs(MacMemoryGuard.safeInUseMB(totalMB: Self.total * 1000) / 1000 - Self.limit) < 1e-6)
    }

    @Test func figuresChangeSizeAtTheirPrecision() {
        #expect(MemoryDisplay.text(decimalGB: 0.52) == "0.48 GB")
        #expect(MemoryDisplay.text(decimalGB: 1.3) == "1.21 GB")
        #expect(MemoryDisplay.text(decimalGB: 20.4) == "19.0 GB")
        #expect(MemoryDisplay.text(decimalGB: 86.9) == "80.9 GB")
        #expect(MemoryDisplay.text(decimalGB: 108.2) == "101 GB")
        #expect(MemoryDisplay.text(decimalGB: 107.3) == "99.9 GB")
        // Just under 100 GiB rounds to the whole figure rather than reading "100.0 GB".
        #expect(MemoryDisplay.text(decimalGB: 107.36) == "100 GB")
        #expect(MemoryDisplay.text(decimalGB: 0) == "0.00 GB")
    }

    @Test func roomUnderTheLineIsNoSwapRoomLessWhatLiesAboveIt() {
        // 86.9 GB usable without swap, with 30.06 GB already above the line: 56.8 GB of room (52.9 GiB).
        let room = MacMemoryGuard.roomUnderLineMB(totalMB: 137_439, noSwapAvailableMB: 86_900)
        #expect(abs(room - 56_835.5) < 1)
        #expect(MemoryDisplay.text(decimalGB: room / 1000) == "52.9 GB")
    }

    @Test func refusalCopyNamesTheLineAndTheRoomInGiB() {
        let refusal = LoadMemoryRefusal(
            model: "flux", name: "FLUX.2 Klein 4B", precision: "int8", neededGB: 12, freeGB: 9.5, limitGB: Self.limit, totalGB: Self.total)
        #expect(
            refusal.explanation
                == "Arsenal keeps the Mac at or under 100 GB of 128 GB in use, which keeps it out of swap. "
                + "This model needs about 11.2 GB at INT8; 8.85 GB fits under that limit now. Loading anyway lets macOS use swap, which slows the whole Mac.")
        let released = LoadMemoryRefusal(
            model: "flux", name: "FLUX.2 Klein 4B", precision: "int8", neededGB: 12, freeGB: 9.5, releasedGB: 4, limitGB: Self.limit, totalGB: Self.total)
        #expect(released.explanation.contains("8.85 GB fits under that limit after unloading the current model."))
        // The wire keeps decimal GB: the same refusal's details carry the decimal figures.
        #expect(refusal.details["limit_gb"] == .number(Self.limit) && refusal.details["total_gb"] == .number(Self.total))
        #expect(refusal.details["free_gb"] == .number(9.5))
    }
}
