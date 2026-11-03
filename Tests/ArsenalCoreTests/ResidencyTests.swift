// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@Test func residencyPolicies() {
    let manual = ResidentModel(id: "manual", residency: .manual, lastUsed: 1, memoryMB: 1000)
    let demand = ResidentModel(id: "demand", residency: .onDemand, lastUsed: 2, memoryMB: 1000)
    #expect(ResidencyPolicy.evictionOrder([manual, demand]).map(\.id) == ["demand", "manual"])
    #expect(ResidencyPolicy.admission(id: "new", estimateMB: 500, availableMB: 100, residents: [manual, demand], allowSwap: false) == .admit(evict: ["demand"]))
    #expect(ResidencyPolicy.admission(id: "new", estimateMB: 5000, availableMB: 100, residents: [manual, demand], allowSwap: false) == .refuse(needMB: 5512, freeMB: 100))
    #expect(ResidencyPolicy.deadline(demand, settings: .init(), minuteSeconds: 1) == 17)
    #expect(ResidencyPolicy.deadline(manual, settings: .init()) == nil)
    #expect(ResidencyPolicy.pressureVictims([manual, demand], order: ["demand", "manual"]) == ["demand"])
}

@Test func fullHostChipSlugMatchesResultFiles() {
    #expect(HostChip.slug("Apple M5 Max") == "m5-max")
    #expect(HostChip.slug("Apple M4 Pro") == "m4-pro")
    #expect(HostChip.slug("Apple M3") == "m3")
    #expect(HostChip.slug("Apple M2 Ultra") == "m2-ultra")
}

@Test func partialResidencySettingsKeepTheirDefaultsOnAFreshProfile() throws {
    let settings = try JSONDecoder().decode(ResidencySettings.self, from: Data(#"{"allowSwap":true}"#.utf8))
    #expect(settings.allowSwap && settings.manualIdleMinutes == 0 && settings.onDemandIdleMinutes == 15)
    let empty = try JSONDecoder().decode(ResidencySettings.self, from: Data("{}".utf8))
    #expect(empty == ResidencySettings())
}
