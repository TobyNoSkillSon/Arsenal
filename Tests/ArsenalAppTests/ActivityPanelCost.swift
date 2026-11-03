// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import QuartzCore
import ArsenalCore
import Testing

@testable import Arsenal

/// `ARSENAL_MEASURE_PANEL=<seconds>`: what the open Activity panel costs, redraw included. The production Activity item
/// view is hosted in a borderless window at (-9000, -9000) (never ordered on screen), fed by a real aggregator with the
/// real machine sampler, as the app's own source does. Prints this process's CPU time over the window in four states:
/// panel closed (the process's baseline), closed while a tool window follows its own streaming request, open with a decision
/// queue draining one request every 0.19 s (a serial runner's pace), and open at rest after it, the first open's set-up
/// excluded. Not run by default.
@MainActor @Suite(.serialized) struct ActivityPanelCost {
    struct Source: ActivitySource {
        let aggregator: ActivityAggregator
        let rows: [ModelRow]
        func stream() async -> AsyncStream<ActivityFeed> {
            let upstream = await aggregator.stream { [aggregator, rows] in await aggregator.snapshot(rows: rows, memory: .init(models: ["laya": .init(footprintMB: 1_180)])) }
            return AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
                let task = Task {
                    for await snapshot in upstream { continuation.yield(ActivityFeed(snapshot)) }
                    continuation.finish()
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        }
    }

    /// One thread's CPU time (the main thread: the panel's SwiftUI updates and drawing).
    static func threadMS(_ thread: thread_act_t) -> Double {
        var info = thread_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<thread_basic_info>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { thread_info(thread, thread_flavor_t(THREAD_BASIC_INFO), $0, &count) }
        }
        guard status == KERN_SUCCESS else { return .nan }
        return Double(info.user_time.seconds + info.system_time.seconds) * 1000 + Double(info.user_time.microseconds + info.system_time.microseconds) / 1000
    }
    static func cpuMS() -> Double {
        var usage = rusage(); getrusage(RUSAGE_SELF, &usage)
        return Double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000 + Double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MEASURE_PANEL"] != nil))
    func measureTheOpenPanel() async throws {
        let seconds = Double(ProcessInfo.processInfo.environment["ARSENAL_MEASURE_PANEL"] ?? "") ?? 60
        let rows = [ModelRow(id: "laya", name: "Laya \u{00b7} Typed decisions", kind: "decisions", states: ["fp16": .init(state: .loaded)], selectedPrecision: "fp16")]
        let aggregator = ActivityAggregator()
        let table = TableController(service: FixtureService(rows: rows))
        defer { table.stop() }
        let menu = ActivityMenu(controller: table)
        menu.activity.rows = Dictionary(rows.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        menu.activity.source = Source(aggregator: aggregator, rows: rows)
        let host = try #require(menu.item().submenu?.items.first?.view as? ActivityHostingView)
        let harness = MotionFrames(host: host)
        defer { harness.close() }
        guard let panelMenu = menu.item().submenu else { return }

        // The window is never ordered on screen, so nothing would ask it to draw: a 60 Hz pump displays whatever the panel
        // marked dirty, as the window server does for an open menu, and the redraw is part of the measured cost.
        let mainThread = mach_thread_self()
        func window(_ label: String) async throws -> String {
            let mainStart = Self.threadMS(mainThread)
            let start = Self.cpuMS(), samples = await aggregator.sampleCount, publishes = await aggregator.publishCount
            let layouts = host.layoutPasses, draws = host.hostingDrawPasses
            let end = ContinuousClock.now + .seconds(seconds)
            while ContinuousClock.now < end {
                host.displayIfNeeded(); CATransaction.flush()
                try await Task.sleep(for: .milliseconds(16))
            }
            let ms = Self.cpuMS() - start, main = Self.threadMS(mainThread) - mainStart
            let read = await aggregator.sampleCount - samples, published = await aggregator.publishCount - publishes
            return String(
                format: "%@: %.0f ms CPU in %.0f s = %.2f%% of one core (main thread %.0f ms); %d machine readings, %d publishes, %d layouts, %d draws", label,
                ms, seconds, ms / seconds / 10, main, read, published, host.layoutPasses - layouts, host.hostingDrawPasses - draws)
        }
        try await Task.sleep(for: .seconds(2))
        var lines = [try await window("closed")]
        // Activity closed while a tool window follows its own request (a chat streaming a token every 50 ms): the window's
        // progress-only subscription, as OperationSession, ChatSession and ImageSession open it.
        let tool = await aggregator.stream(machine: false) { [aggregator, rows] in await aggregator.snapshot(rows: rows, memory: .init()) }
        let toolConsumer = Task { for await _ in tool {} }
        let streaming = Task {
            let id = aggregator.begin(model: "laya", op: "chat", requester: "Arsenal")
            var tokens = 0
            while !Task.isCancelled {
                tokens += 1
                aggregator.receive(id, .json(["event": "progress", "phase": "decode", "generated_tokens": .number(Double(tokens))]))
                try? await Task.sleep(for: .milliseconds(50))
            }
            aggregator.finish(id)
        }
        try await Task.sleep(for: .seconds(2))
        lines.append(try await window("closed, a tool window following its request"))
        streaming.cancel(); await streaming.value
        toolConsumer.cancel(); await toolConsumer.value
        try await Task.sleep(for: .seconds(1))
        menu.menuWillOpen(panelMenu)
        // The first open builds the panel's views and fonts once; that is not the cost of keeping it open.
        try await Task.sleep(for: .seconds(5))
        // A serial decision runner working through a queue: forty waiting, one started every 0.19 s.
        let queue = Task {
            var waiting = (0..<40).map { _ in aggregator.begin(model: "laya", op: "decide", requester: "Claude Code") }
            while !Task.isCancelled {
                let id = waiting.removeFirst()
                aggregator.receive(id, .json(["event": "progress", "phase": "prefill"]))
                try? await Task.sleep(for: .milliseconds(190))
                aggregator.finish(id, .finished(["answers": [:]]))
                waiting.append(aggregator.begin(model: "laya", op: "decide", requester: "Claude Code"))
            }
            for id in waiting { aggregator.finish(id, .cancelled) }
        }
        try await Task.sleep(for: .seconds(2))
        lines.append(try await window("open, decision queue draining"))
        queue.cancel(); await queue.value
        try await Task.sleep(for: .seconds(2))
        lines.append(try await window("open, at rest"))
        menu.menuDidClose(panelMenu)
        try await Task.sleep(for: .seconds(1))
        #expect(await aggregator.subscriberCount == 0)
        print(lines.joined(separator: "\n"))
    }
}
