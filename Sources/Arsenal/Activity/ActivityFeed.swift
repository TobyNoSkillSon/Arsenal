// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import SwiftUI
import ArsenalCore

// What the activity panel shows (C3): per loaded model its state, the requests it serves and for whom, queue, batch and
// memory; and a machine strip (GPU busy, thermal state, the busiest other GPU process). Decoded from C3's JSON, so the
// panel reads the same document as any API client; the source is the service's own stream, subscribed only while the
// panel is on screen.

struct ActivityFeed: Equatable {
    var models: [Model] = []
    var machine = Machine()

    struct Model: Equatable, Identifiable {
        var id: String
        var kind: String
        var state: String
        var requests: [Request]
        var queued: Int
        var batch: Double?
        var memoryMB: Double?
        /// Being worked on now, and the last minute (nil when it was empty).
        var running = 0
        var recent: Recent?
    }
    /// A model's last minute, as C3 reports it: what finished, how long it took, and how much of each second the model
    /// was working.
    struct Recent: Equatable {
        var finished = 0
        var failed = 0
        var meanRunS: Double?
        var meanWaitS: Double?
        var tokens = 0
        var steps = 0
        var images = 0
        var audioS = 0.0
        /// Seconds of the minute with a request running, and per second the part that was (0…1), oldest first.
        var busyS = 0.0
        var busy: [Double] = []
    }
    struct Request: Equatable, Identifiable {
        var id: String
        var requester: String
        var op: String
        var phase: String
        /// The operation's title ("Separate Layers"), and the runner's own words for what it is doing now
        /// ("Separating layers") and what its steps count ("layers"); nil when not sent.
        var title: String?
        var label: String?
        var unit: String?
        var promptTokens: Int?
        var cachedTokens: Int?
        var generatedTokens: Int?
        var step: Int?
        var total: Int?
        var startedAt: Date?
        var tps: Double?
        /// Speech produced so far, in seconds of audio (once the model's sample rate is known).
        var audioS: Double?
        /// The request's C2 timing so far (ttft_s and prefill_tps once the first token is out, decode_tps, peak_mb, …).
        var timing: [String: Double] = [:]
        /// When its runner started it (nil while it waits, or from an older service).
        var runningSince: Date?
        /// The rate that belongs to the phase: prefill while reading the prompt, decode while writing; never one for the
        /// other.
        var phaseRate: Double? {
            switch phase {
            case "prefill": return timing["prefill_tps"]
            case "decode", "done": return timing["decode_tps"]
            default: return nil
            }
        }
    }
    /// The machine now (sampled every second while the panel is open): utilisation, die temperatures, memory.
    struct Machine: Equatable {
        var cpuBusyPct: Double?
        var gpuBusyPct: Double?
        var cpuTempC: Double?
        var gpuTempC: Double?
        var memoryUsedMB: Double?
        var memoryTotalMB: Double?
        var arsenalMB: Double?
    }
}

extension ActivityFeed {
    /// C3's JSON (snake case), as `GET /arsenal/v1/activity` returns it.
    init(json: [String: Any]) {
        func int(_ v: Any?) -> Int? { (v as? NSNumber)?.intValue }
        func double(_ v: Any?) -> Double? {
            guard let n = (v as? NSNumber)?.doubleValue, n.isFinite else { return nil }
            return n
        }
        models = (json["models"] as? [[String: Any]] ?? []).compactMap { m in
            guard let id = m["id"] as? String else { return nil }
            let requests = (m["requests"] as? [[String: Any]] ?? []).enumerated().map { i, r in
                Request(
                    id: r["id"] as? String ?? "\(id)-\(i)", requester: r["requester"] as? String ?? "API", op: r["op"] as? String ?? "",
                    phase: r["phase"] as? String ?? "queued", title: Self.words(r["title"], 60), label: Self.words(r["label"], 60), unit: Self.words(r["unit"], 24),
                    promptTokens: int(r["prompt_tokens"]), cachedTokens: int(r["cached_tokens"]),
                    generatedTokens: int(r["generated_tokens"]), step: int(r["step"]), total: int(r["total"]),
                    startedAt: Self.date(r["started_at"]), tps: double(r["tps"]), audioS: double(r["audio_s"]),
                    timing: (r["timing"] as? [String: Any] ?? [:]).compactMapValues { double($0) }, runningSince: Self.date(r["running_since"]))
            }
            let recent = (m["recent"] as? [String: Any]).map { r in
                Recent(
                    finished: int(r["finished"]) ?? 0, failed: int(r["failed"]) ?? 0, meanRunS: double(r["mean_run_s"]), meanWaitS: double(r["mean_wait_s"]),
                    tokens: int(r["tokens"]) ?? 0, steps: int(r["steps"]) ?? 0, images: int(r["images"]) ?? 0, audioS: double(r["audio_s"]) ?? 0,
                    busyS: double(r["busy_s"]) ?? 0, busy: (r["busy"] as? [Any] ?? []).map { double($0) ?? 0 })
            }
            return Model(
                id: id, kind: m["kind"] as? String ?? "", state: m["state"] as? String ?? "idle", requests: requests, queued: int(m["queued"]) ?? 0,
                batch: double(m["batch"]), memoryMB: double(m["memory_mb"]), running: int(m["running"]) ?? 0, recent: recent)
        }
        let machine = json["machine"] as? [String: Any] ?? [:]
        self.machine = Machine(
            cpuBusyPct: double(machine["cpu_busy_pct"]), gpuBusyPct: double(machine["gpu_busy_pct"]), cpuTempC: double(machine["cpu_temp_c"]),
            gpuTempC: double(machine["gpu_temp_c"]), memoryUsedMB: double(machine["memory_used_mb"]), memoryTotalMB: double(machine["memory_total_mb"]),
            arsenalMB: double(machine["arsenal_mb"]))
    }

    /// A runner's short text, bounded: empty is absent.
    static func words(_ v: Any?, _ limit: Int) -> String? {
        guard let s = (v as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else { return nil }
        return String(s.prefix(limit))
    }

    /// Dates arrive as ISO 8601 strings or as seconds (Foundation's default Date encoding: since 2001).
    static func date(_ v: Any?) -> Date? {
        if let s = v as? String { return ISO8601DateFormatter().date(from: s) }
        if let n = (v as? NSNumber)?.doubleValue { return n > 1_000_000_000 * 1.5 ? Date(timeIntervalSince1970: n) : Date(timeIntervalSinceReferenceDate: n) }
        return nil
    }

    /// The service's snapshot, through its JSON form (one decoding path for the in-process stream and HTTP).
    init?<T: Encodable>(encoded value: T) {
        guard let data = try? JSONEncoder().encode(value), let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        self.init(json: object)
    }
}

/// How a model's card is drawn: its folder's `activity` section (C1), else its kind's, else the default one. A new tool
/// kind adds a case here and its card in ActivityPanel.swift; models plug in by naming it in their folder.
enum ActivitySection: String, CaseIterable {
    case chat, image, voice, decision, `default`
    init(row: ModelRow?, kind: String) {
        if let named = row.flatMap({ ActivitySection(rawValue: $0.activity) }), named != .default { self = named; return }
        if let tool = row?.tools.first.flatMap(ActivitySection.init(rawValue:)) { self = tool; return }
        switch kind {
        case "chat": self = .chat
        case "images": self = .image
        case "voices": self = .voice
        case "decisions": self = .decision
        default: self = .default
        }
    }
}

/// Where snapshots come from: the service's stream in the app, fixed snapshots in tests and renders.
protocol ActivitySource: Sendable {
    func stream() async -> AsyncStream<ActivityFeed>
}

/// The panel's state: subscribed while visible, nothing running otherwise.
@MainActor final class ActivityModel: ObservableObject {
    @Published private(set) var feed = ActivityFeed()
    @Published private(set) var visible = false
    /// Names and folder sections for ids (from the table's rows).
    @Published var rows: [String: ModelRow] = [:]
    /// Each model's memory from the service's memory snapshot, until the activity stream reports its own.
    @Published var memory: [String: Double] = [:]
    /// A snapshot has arrived since the panel opened (a value still missing then is "not reported").
    @Published private(set) var sampled = false
    var source: ActivitySource?
    private var task: Task<Void, Never>?
    /// Subscriptions started so far (tests check that nothing subscribes while hidden).
    private(set) var subscriptions = 0

    func show() {
        guard !visible else { return }
        visible = true
        sampled = false
        // The first frame already lists what is loaded (idle until the first snapshot says otherwise), so the cards do
        // not appear a moment after the panel; nothing from an earlier opening is shown as current.
        let loaded = rows.values.filter { $0.residentDType != nil }.sorted { $0.name < $1.name }
        let previous = Dictionary(feed.models.map { ($0.id, $0.memoryMB) }, uniquingKeysWith: { a, _ in a })
        feed = ActivityFeed(
            models: loaded.map { .init(id: $0.id, kind: $0.kind, state: "idle", requests: [], queued: 0, batch: nil, memoryMB: previous[$0.id] ?? memory[$0.id]) },
            machine: .init())
        guard let source else { return }
        subscriptions += 1
        task = Task { [weak self] in
            for await feed in await source.stream() {
                guard let self, !Task.isCancelled else { return }
                if !self.sampled { self.sampled = true }
                if feed != self.feed { self.feed = feed }
            }
        }
    }
    func hide() {
        visible = false
        task?.cancel(); task = nil
    }
    /// Render harness and tests.
    func preview(_ feed: ActivityFeed, sampled: Bool = true) { self.feed = feed; self.sampled = sampled }

    func name(_ id: String) -> String { rows[id]?.name ?? id }

    func section(_ m: ActivityFeed.Model) -> ActivitySection { ActivitySection(row: rows[m.id], kind: m.kind) }

    /// What decides the panel's height: per card, its counts line, its request lines and which of them carry a bar, and
    /// its minute. Request identities and figures are left out, so a busy model's requests coming and going every second
    /// never resize the menu; the counts line stays (reading Idle) until the minute empties.
    func layout(_ feed: ActivityFeed) -> [String] {
        feed.models.map { m in
            let section = section(m)
            let lines = section == .decision ? [] : ActivityText.lines(m)
            let shown = lines.prefix(ActivityText.maxLines).map { ActivityText.showsBar($0.request, section) ? "bar" : "line" }
            return "\(m.id) counts:\(ActivityText.counts(m) != nil) \(shown.joined(separator: ",")) minute:\(m.recent != nil)"
        }
    }
    /// Every model on this Mac, so a model's colour does not depend on which others are loaded.
    var installed: [String] { rows.values.filter { $0.states.values.contains { $0.state != .notInstalled } }.map(\.id) }
}

/// C3 over the app's own API (`GET /arsenal/v1/activity/stream`): the service samples the machine only while this
/// stream is open, and it closes with the panel.
struct APIActivitySource: ActivitySource {
    let client: @MainActor () -> ArsenalClient?
    func stream() async -> AsyncStream<ActivityFeed> {
        guard let client = await client() else { return AsyncStream { $0.finish() } }
        return AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            let task = Task {
                do {
                    for try await data in client.events("GET", "/arsenal/v1/activity/stream") {
                        if let object = eventObject(data) { continuation.yield(ActivityFeed(json: object)) }
                    }
                } catch {}
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Fixed snapshots (tests and renders).
struct FixedActivitySource: ActivitySource {
    let feeds: [ActivityFeed]
    func stream() async -> AsyncStream<ActivityFeed> {
        AsyncStream { c in
            for f in feeds { c.yield(f) }
            c.finish()
        }
    }
}

/// C3 in-process: the service's own activity stream (it samples the machine only while someone is subscribed).
struct ServiceActivitySource: ActivitySource {
    let service: any ModelsService
    func stream() async -> AsyncStream<ActivityFeed> {
        let upstream = await service.activityStream()
        return AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            let task = Task {
                for await snapshot in upstream { continuation.yield(ActivityFeed(snapshot)) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

extension ActivityFeed {
    init(_ s: ActivitySnapshot) {
        models = s.models.map { m in
            Model(
                id: m.id, kind: m.kind, state: m.state,
                requests: m.requests.map {
                    Request(
                        id: $0.id.uuidString, requester: $0.requester, op: $0.op, phase: $0.phase, title: Self.words($0.title, 60), label: Self.words($0.label, 60),
                        unit: Self.words($0.unit, 24), promptTokens: $0.promptTokens, cachedTokens: $0.cachedTokens,
                        generatedTokens: $0.generatedTokens, step: $0.step, total: $0.total, startedAt: $0.startedAt, tps: $0.tps, audioS: $0.audioS,
                        timing: ($0.timing ?? [:]).compactMapValues { $0.number.flatMap { $0.isFinite ? $0 : nil } }, runningSince: $0.runningSince)
                }, queued: m.queued, batch: m.batch, memoryMB: m.memoryMB, running: m.running,
                recent: m.recent.map {
                    Recent(
                        finished: $0.finished, failed: $0.failed, meanRunS: $0.meanRunS, meanWaitS: $0.meanWaitS, tokens: $0.tokens, steps: $0.steps, images: $0.images,
                        audioS: $0.audioS, busyS: $0.busyS, busy: $0.busy)
                })
        }
        let m = s.machine
        machine = Machine(
            cpuBusyPct: m.cpuBusyPct, gpuBusyPct: m.gpuBusyPct, cpuTempC: m.cpuTempC, gpuTempC: m.gpuTempC, memoryUsedMB: m.memoryUsedMB,
            memoryTotalMB: m.memoryTotalMB, arsenalMB: m.arsenalMB)
    }
}
