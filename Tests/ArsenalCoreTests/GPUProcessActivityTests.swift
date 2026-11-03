// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func legacyGPUActivityPayloadsStillDecode() throws {
    let legacy = try JSONDecoder().decode(GPUProcessActivity.self, from: Data(#"{"name":"WindowServer","pct":10}"#.utf8))
    #expect(legacy.pid == nil && legacy.name == "WindowServer" && legacy.pct == 10)
    let process = GPUProcessActivity(name: "Fixture", pct: 42, pid: 123)
    let decoded = try JSONDecoder().decode(MachineActivity.self, from: JSONEncoder().encode(MachineActivity(topOtherGPU: process)))
    #expect(decoded.topOtherGPU == process)
}
