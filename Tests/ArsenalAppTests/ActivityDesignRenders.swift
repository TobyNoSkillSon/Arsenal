// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// Review renders for Activity showing what each model is doing: the panel at real width, 2x
/// pixels, dark and light, on the plain menu material and over a busy blurred wallpaper. States: idle; one decision model
/// working through a queue of 40 (one at a time, as its runner does); a chat streaming; an image mid-generation; a voice
/// speaking with two waiting; another kind's operation (the layer fixture) mid-way; three models busy at once; the
/// decision queue drained a few seconds ago. Fixture feeds with figures the live panel can show (rate, time
/// each and wait agree), no worker, no GPU. Run with ARSENAL_ACTIVITY_RENDERS=<dir>.
@MainActor @Suite(.menuState, .serialized) final class ActivityDesignRenders {
    static let laya = "laya-typed-decisions", qwen = "qwen3.8-27b", flux = "flux2-klein-4b", kokoro = "kokoro-82m", layers = LayersFixture.id

    /// The fixture table's rows with all four loaded (the panel takes names and kinds from them).
    static func rows() -> [ModelRow] {
        var rows = Fixtures.rows(.loaded)
        if let i = rows.firstIndex(where: { $0.id == flux }) { rows[i].states["bf16"] = .init(state: .loaded) }
        rows.append(
            ModelRow(id: laya, name: "Laya \u{00b7} Typed decisions", kind: "decisions", precisions: ["fp16"], states: ["fp16": .init(state: .loaded)], selectedPrecision: "fp16"))
        rows.append(LayersFixture.row)
        return rows
    }
    static let machineIdle = ActivityFeed.Machine(cpuBusyPct: 4, gpuBusyPct: 2, cpuTempC: 41, gpuTempC: 38, memoryUsedMB: 52_400, memoryTotalMB: 137_439, arsenalMB: 310)
    static let machineBusy = ActivityFeed.Machine(cpuBusyPct: 31, gpuBusyPct: 96, cpuTempC: 58, gpuTempC: 74, memoryUsedMB: 71_800, memoryTotalMB: 137_439, arsenalMB: 330)
    /// A small model working: the GPU partly busy, the dies warm.
    static let machineVoice = ActivityFeed.Machine(cpuBusyPct: 12, gpuBusyPct: 38, cpuTempC: 49, gpuTempC: 55, memoryUsedMB: 54_900, memoryTotalMB: 137_439, arsenalMB: 320)
    static let memory: [String: Double] = [laya: 1_180, qwen: 15_420, flux: 13_240, kokoro: 410, layers: 2_150]

    /// A minute of busy share per second, oldest first: idle, then busy from `from` (seconds into the minute, fractional:
    /// the first second is partly busy), with `gaps` idle seconds.
    static func busy(from: Double, gaps: Set<Int> = []) -> [Double] {
        (0..<60).map { i in
            if gaps.contains(i) { return 0 }
            return min(1, max(0, Double(i + 1) - from))
        }
    }
    static func busyS(_ values: [Double]) -> Double { values.reduce(0, +) }

    static func idle(_ id: String, kind: String) -> ActivityFeed.Model {
        .init(id: id, kind: kind, state: "idle", requests: [], queued: 0, batch: nil, memoryMB: memory[id])
    }
    /// A one-at-a-time decision runner (as d1-3B and the decisions worker are) working through a queue: one running, 40
    /// waiting; flat out for the last 48 s at 0.19 s each, so 250 finished in the minute (5.2 a second) and the queue of
    /// 40 clears in 40 × 0.19 ≈ 8 s. Requests that waited behind a queue that size waited about as long.
    static func decisions() -> ActivityFeed.Model {
        let now = Date()
        let running = ActivityFeed.Request(
            id: "d0", requester: "Claude Code", op: "judge", phase: "prefill", startedAt: now.addingTimeInterval(-7.5), runningSince: now.addingTimeInterval(-0.08))
        let waiting = (0..<40).map { i in ActivityFeed.Request(id: "w\(i)", requester: "Claude Code", op: "judge", phase: "queued", startedAt: now) }
        let busy = busy(from: 11.6)
        return .init(
            id: laya, kind: "decisions", state: "prefill", requests: [running] + waiting, queued: 40, batch: nil, memoryMB: memory[laya], running: 1,
            recent: .init(finished: 250, failed: 0, meanRunS: 0.192, meanWaitS: 7.4, busyS: busyS(busy), busy: busy))
    }
    /// One stream writing at 30.8 tok/s for the last 14 s with one waiting behind it. Earlier in the minute one 15.8 s
    /// reply (438 tokens), then 3 s idle: 29.8 s of work, 850 tokens, so 28.5 tok/s on average (prefill included).
    static func chat() -> ActivityFeed.Model {
        let request = ActivityFeed.Request(
            id: "c", requester: "Claude Code", op: "chat", phase: "decode", promptTokens: 8_214, generatedTokens: 412, startedAt: Date().addingTimeInterval(-14),
            tps: 30.8, timing: ["decode_tps": 30.8, "ttft_s": 1.2], runningSince: Date().addingTimeInterval(-14))
        let busy = busy(from: 27.2, gaps: [43, 44, 45])
        return .init(
            id: qwen, kind: "chat", state: "decode", requests: [request], queued: 1, batch: 1, memoryMB: memory[qwen], running: 1,
            recent: .init(finished: 1, meanRunS: 15.8, meanWaitS: 0.1, tokens: 850, busyS: busyS(busy), busy: busy))
    }
    /// FLUX at step 3 of 4 with two more waiting: generating back to back for the last 16.8 s at 6.1 s an image, so two
    /// finished and the queue clears in about 12 s.
    static func image() -> ActivityFeed.Model {
        let started = Date().addingTimeInterval(-4.6)
        let request = ActivityFeed.Request(id: "i", requester: "Arsenal", op: "generate", phase: "step", step: 3, total: 4, startedAt: started, runningSince: started)
        let busy = busy(from: 43.2)
        return .init(
            id: flux, kind: "images", state: "step", requests: [request], queued: 2, batch: nil, memoryMB: memory[flux], running: 1,
            recent: .init(finished: 2, meanRunS: 6.1, meanWaitS: 3.2, steps: 11, images: 2, busyS: busyS(busy), busy: busy))
    }
    /// Kokoro reading a reply aloud, 1.6 s in (3.9 s of audio so far, about 2.4× real time), with two more waiting; six
    /// clips earlier in the minute, 7 s of audio each in 2.9 s.
    static func voice() -> ActivityFeed.Model {
        let now = Date()
        let speaking = ActivityFeed.Request(
            id: "v", requester: "Claude Code", op: "speak", phase: "step", startedAt: now.addingTimeInterval(-2.4), audioS: 3.9,
            runningSince: now.addingTimeInterval(-1.6))
        let waiting = (0..<2).map { i in ActivityFeed.Request(id: "vw\(i)", requester: "Claude Code", op: "speak", phase: "queued", startedAt: now) }
        let busy = busy(from: 38.4, gaps: [41, 42, 43, 44, 49, 50, 51, 55, 56])
        return .init(
            id: kokoro, kind: "voices", state: "step", requests: [speaking] + waiting, queued: 2, batch: nil, memoryMB: memory[kokoro], running: 1,
            recent: .init(finished: 6, meanRunS: 2.9, meanWaitS: 1.8, audioS: 42, busyS: busyS(busy), busy: busy))
    }
    /// Another kind (the layer fixture, drawn by the default card): separating layers 2 of 4, one more waiting; three
    /// separations earlier in the minute at 4.2 s each.
    static func layered() -> ActivityFeed.Model {
        let now = Date()
        let request = ActivityFeed.Request(
            id: "l", requester: "Arsenal", op: "separate", phase: "step", title: "Separate Layers", label: "Separating layers", unit: "layers", step: 2, total: 4,
            startedAt: now.addingTimeInterval(-2.1), runningSince: now.addingTimeInterval(-2.1))
        let waiting = ActivityFeed.Request(id: "lw", requester: "Arsenal", op: "separate", phase: "queued", startedAt: now)
        let busy = busy(from: 44.6, gaps: [49, 50, 51, 52])
        return .init(
            id: layers, kind: LayersFixture.kind.id, state: "step", requests: [request, waiting], queued: 1, batch: nil, memoryMB: memory[layers], running: 1,
            recent: .init(finished: 3, meanRunS: 4.2, meanWaitS: 2.6, steps: 14, busyS: busyS(busy), busy: busy))
    }
    /// The decision queue drained six seconds ago: nothing running or waiting, the machine quiet again, and the minute's
    /// last six seconds idle (42.4 s flat out at 0.19 s each: 220 finished).
    static func decisionsDrained() -> ActivityFeed.Model {
        var m = decisions()
        let busy = busy(from: 11.6, gaps: Set(54..<60))
        m.requests = []; m.queued = 0; m.running = 0; m.state = "idle"
        m.recent = .init(finished: 220, failed: 0, meanRunS: 0.192, meanWaitS: 7.4, busyS: busyS(busy), busy: busy)
        return m
    }

    static func feeds() -> [(String, ActivityFeed)] {
        [
            ("idle", ActivityFeed(models: [idle(laya, kind: "decisions"), idle(qwen, kind: "chat"), idle(kokoro, kind: "voices")], machine: machineIdle)),
            ("decisions", ActivityFeed(models: [decisions(), idle(qwen, kind: "chat")], machine: machineBusy)),
            ("chat", ActivityFeed(models: [chat()], machine: machineBusy)),
            ("image", ActivityFeed(models: [image()], machine: machineBusy)),
            ("voice", ActivityFeed(models: [voice(), idle(qwen, kind: "chat")], machine: machineVoice)),
            ("generic", ActivityFeed(models: [layered()], machine: machineVoice)),
            ("three-busy", ActivityFeed(models: [decisions(), chat(), image()], machine: machineBusy)),
            ("decisions-drained", ActivityFeed(models: [decisionsDrained(), idle(qwen, kind: "chat")], machine: machineIdle))
        ]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_ACTIVITY_RENDERS"] != nil))
    func renderActivityDesign() async throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_ACTIVITY_RENDERS"]!)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { Appearance.shared.set(.dark); NSApp.appearance = nil }
        let model = ActivityModel()
        model.rows = Dictionary(Self.rows().map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        var sizes: [String] = []
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            try await Task.sleep(for: .milliseconds(100))
            for name in Self.feeds().map(\.0) {
                for wallpaper in [false, true] {
                    // Built just before each capture: a running request's time left counts from now.
                    model.preview(Self.feeds().first { $0.0 == name }!.1)
                    let file = "activity-\(name)-\(mode.rawValue)\(wallpaper ? "-wallpaper" : "").png"
                    let size = await QA.renderActivity(model, to: directory.appendingPathComponent(file), wallpaper: wallpaper)
                    sizes.append("\(file) \(Int(size.width)) × \(Int(size.height)) pt")
                }
            }
        }
        // Every hover text the cards carry, as text (a PNG cannot hover).
        let tips = Self.feeds().flatMap { name, feed in
            feed.models.flatMap { m in
                [ActivityText.countsHelp(m), m.recent.map { ActivityText.minuteHelp($0, model.section(m)) }].compactMap { $0 }
                    .map { "\(name) / \(m.id): " + $0.replacingOccurrences(of: "\n", with: " | ") }
            }
        }
        try (sizes + [""] + tips).joined(separator: "\n").write(to: directory.appendingPathComponent("activity-renders.txt"), atomically: true, encoding: .utf8)
    }
}
