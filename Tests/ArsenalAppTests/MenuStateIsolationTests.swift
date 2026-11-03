// SPDX-License-Identifier: MIT
import Testing

/// Exercise the same lease used by independent suites, including suspension and error recovery.
struct MenuStateIsolationTests {
    @Test func sharedLeaseSurvivesSuspensionAndReleasesAfterAThrow() async throws {
        let current = try #require(Test.current)
        let probe = Probe()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<8 {
                group.addTask {
                    do {
                        try await MenuStateIsolation().provideScope(for: current, testCase: nil) {
                            await probe.enter()
                            for _ in 0..<10 { await Task.yield() }
                            await probe.leave()
                            if index == 0 { throw FixtureError.expected }
                        }
                    } catch FixtureError.expected {}
                }
            }
            try await group.waitForAll()
        }
        #expect(await probe.peak == 1)
        #expect(await probe.entries == 8)
        #expect(await probe.active == 0)
    }

    private enum FixtureError: Error { case expected }
    private actor Probe {
        var active = 0, peak = 0, entries = 0
        func enter() { active += 1; entries += 1; peak = max(peak, active) }
        func leave() { active -= 1 }
    }
}
