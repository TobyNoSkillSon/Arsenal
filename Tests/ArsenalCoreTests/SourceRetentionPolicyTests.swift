// SPDX-License-Identifier: MIT
import Testing
@testable import ArsenalCore

// "Don't delete any models or 16-bit copies; they may be useful later." No 16-bit copy is retired after a derived
// load, even when every condition that used to qualify it (verified derive, derived tier in use, large enough) holds.
@Test func sourceRetentionNeverRetiresASource() {
    func eligible(selected: String = "int4", resident: String? = "int4", verified: Bool = true, size: Int64 = 9, getting: Bool = false) -> Bool {
        SourceRetentionPolicy.eligible(
            sourcePrecision: "bf16", derivedPrecision: "int4", selectedPrecision: selected, residentPrecision: resident, verifiedDerived: verified, sourceBytes: size,
            thresholdBytes: 8, getInFlight: getting)
    }
    #expect(!eligible())
    #expect(!eligible(selected: "bf16"))
    #expect(!eligible(resident: nil))
    #expect(!eligible(resident: "bf16"))
    #expect(!eligible(verified: false))
    #expect(!eligible(size: 8))
    #expect(!eligible(getting: true))
}
