// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

#if DEBUG
    @MainActor @Suite(.menuState, .serialized) struct MotionPredicateTests {
        @Test func pixelsRejectFlickerAndStaleFinalContent() {
            let old = Data([10, 20, 30, 255]), expected = Data([40, 50, 60, 255])
            #expect(MotionSequence.oneForwardSwitch([old, old, expected, expected], from: old, to: expected))
            #expect(!MotionSequence.oneForwardSwitch([old, expected, old, expected], from: old, to: expected))
            #expect(!MotionSequence.oneForwardSwitch([old, old, old], from: old, to: expected))
            #expect(!MotionSequence.oneForwardSwitch([old, Data([0, 0, 0, 0]), expected], from: old, to: expected))
            #expect(MotionSequence.oneForwardSwitch([old, old], from: old, to: old))
        }

        @Test func sizesRejectReversalsAndIntermediateAllocations() {
            let old = NSSize(width: 364, height: 150), expected = NSSize(width: 364, height: 240)
            #expect(MotionSequence.oneForwardSwitch([old, expected, expected], from: old, to: expected))
            #expect(!MotionSequence.oneForwardSwitch([old, expected, old, expected], from: old, to: expected))
            #expect(!MotionSequence.oneForwardSwitch([old, NSSize(width: 364, height: 200), expected], from: old, to: expected))
            #expect(!MotionSequence.oneForwardSwitch([old, old], from: old, to: expected))
        }

        @Test func explicitActivitySettersAreCountedEvenBetweenFrames() async throws {
            let table = TableController(service: FixtureService(rows: []))
            let menu = ActivityMenu(controller: table)
            let host = try #require(menu.item().submenu?.items.first?.view as? ActivityHostingView)
            let harness = MotionFrames(host: host)
            defer { harness.close(); menu.activity.hide(); table.stop() }
            await harness.settle()
            let old = host.frame.size
            let expected = NSSize(width: old.width, height: old.height + 80)
            let initial = harness.sizes.count
            host.setNaturalSize(expected)
            host.setNaturalSize(expected) // No allocation when the setter is repeated.
            host.setNaturalSize(old)
            host.setNaturalSize(expected) // All three occur before the next sampled frame.
            let allocations = Array(harness.sizes.dropFirst(initial))
            #expect(allocations == [expected, old, expected])
            #expect(!MotionSequence.oneForwardSwitch([old] + allocations, from: old, to: expected))
        }
    }
#endif
