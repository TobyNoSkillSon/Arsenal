// SPDX-License-Identifier: MIT
import Foundation

public struct ActivitySnapshot: Codable, Sendable, Equatable {
    public var models: [ModelActivity]
    public var machine: MachineActivity
    public init(models: [ModelActivity] = [], machine: MachineActivity = .init()) { self.models = models; self.machine = machine }
}
public struct ModelActivity: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var kind: String
    public var revision: ModelRevision?
    public var state: String
    public var requests: [RequestActivity]
    /// Accepted and not yet started by the runner: waiting for memory, for the model to load, or behind other requests.
    public var queued: Int
    /// Being worked on now: the runner reported starting it.
    public var running: Int
    /// The last minute of finished work; nil when the minute was empty.
    public var recent: RecentActivity?
    /// Sequences advanced together, not the number of connected HTTP clients. Absent until reported by a runner.
    public var batch: Double?
    public var memoryMB: Double?
    enum CodingKeys: String, CodingKey { case id, kind, revision, state, requests, queued, running, recent, batch; case memoryMB = "memory_mb" }
    public init(
        id: String, kind: String, state: String = "idle", requests: [RequestActivity] = [], queued: Int = 0, running: Int = 0, recent: RecentActivity? = nil,
        batch: Double? = nil, memoryMB: Double? = nil
    ) {
        self.id = id; self.kind = kind; self.state = state; self.requests = requests; self.queued = queued; self.running = running; self.recent = recent
        self.batch = batch; self.memoryMB = memoryMB
    }
}
/// A model's last minute, in whole seconds (the second under way is not counted yet): what finished, how long it took,
/// and how much of each second the model was working.
public struct RecentActivity: Codable, Sendable, Equatable {
    public var windowS: Int
    public var finished: Int
    public var failed: Int
    /// Mean time from the runner starting a request to its reply, and from acceptance to that start, over the finished
    /// requests whose start was reported.
    public var meanRunS: Double?
    public var meanWaitS: Double?
    public var tokens: Int
    public var steps: Int
    public var images: Int
    public var audioS: Double
    /// Seconds of the minute in which at least one request was running.
    public var busyS: Double
    /// One value per second of the window, oldest first: the part of that second the model was working (0…1).
    public var busy: [Double]
    enum CodingKeys: String, CodingKey {
        case finished, failed, tokens, steps, images, busy
        case windowS = "window_s", meanRunS = "mean_run_s", meanWaitS = "mean_wait_s", audioS = "audio_s", busyS = "busy_s"
    }
    public init(
        windowS: Int = 60, finished: Int = 0, failed: Int = 0, meanRunS: Double? = nil, meanWaitS: Double? = nil, tokens: Int = 0, steps: Int = 0, images: Int = 0,
        audioS: Double = 0, busyS: Double = 0, busy: [Double] = []
    ) {
        self.windowS = windowS; self.finished = finished; self.failed = failed; self.meanRunS = meanRunS; self.meanWaitS = meanWaitS; self.tokens = tokens
        self.steps = steps; self.images = images; self.audioS = audioS; self.busyS = busyS; self.busy = busy
    }
}
public struct RequestActivity: Codable, Sendable, Equatable, Identifiable {
    public var id: UUID
    public var requester: String
    public var op: String
    public var phase: String
    public var title: String?
    public var label: String?
    public var unit: String?
    public var promptTokens: Int?
    public var cachedTokens: Int?
    public var generatedTokens: Int?
    public var step: Int?
    public var total: Int?
    public var startedAt: Date
    /// When its runner reported starting it (absent while it waits).
    public var runningSince: Date?
    public var tps: Double?
    /// Audio produced so far (speech), once the request's sample rate is known (a stream announces it).
    public var audioS: Double?
    public var timing: JSONObject?
    enum CodingKeys: String, CodingKey {
        case id, requester, op, phase, title, label, unit, step, total, tps, timing
        case audioS = "audio_s"
        case promptTokens = "prompt_tokens", cachedTokens = "cached_tokens", generatedTokens = "generated_tokens", startedAt = "started_at"
        case runningSince = "running_since"
    }
    public init(id: UUID = UUID(), requester: String, op: String, phase: String = "queued", startedAt: Date = Date()) {
        self.id = id; self.requester = requester; self.op = op; self.phase = phase; self.startedAt = startedAt
    }
}
/// The machine right now, sampled every second while someone watches. Unknown values are absent, never zero.
public struct MachineActivity: Codable, Sendable, Equatable {
    public var gpuBusyPct: Double?
    public var thermal: String?
    /// Kept for older clients; no longer sampled (the busiest other GPU app is not useful).
    public var topOtherGPU: GPUProcessActivity?
    /// All cores' busy share since the previous sample.
    public var cpuBusyPct: Double?
    /// Mean of the SoC's CPU and GPU die sensors (°C), where this Mac's sensors are known.
    public var cpuTempC: Double?
    public var gpuTempC: Double?
    /// Memory in use as Activity Monitor counts it (app + wired + compressed), the machine's total, and Arsenal's own
    /// process footprint (model workers report their own in each model's `memory_mb`).
    public var memoryUsedMB: Double?
    public var memoryTotalMB: Double?
    public var arsenalMB: Double?
    enum CodingKeys: String, CodingKey {
        case gpuBusyPct = "gpu_busy_pct", thermal, topOtherGPU = "top_other_gpu", cpuBusyPct = "cpu_busy_pct", cpuTempC = "cpu_temp_c", gpuTempC = "gpu_temp_c"
        case memoryUsedMB = "memory_used_mb", memoryTotalMB = "memory_total_mb", arsenalMB = "arsenal_mb"
    }
    public init(
        gpuBusyPct: Double? = nil, thermal: String? = nil, topOtherGPU: GPUProcessActivity? = nil, cpuBusyPct: Double? = nil, cpuTempC: Double? = nil,
        gpuTempC: Double? = nil, memoryUsedMB: Double? = nil, memoryTotalMB: Double? = nil, arsenalMB: Double? = nil
    ) {
        self.gpuBusyPct = gpuBusyPct; self.thermal = thermal; self.topOtherGPU = topOtherGPU; self.cpuBusyPct = cpuBusyPct
        self.cpuTempC = cpuTempC; self.gpuTempC = gpuTempC; self.memoryUsedMB = memoryUsedMB; self.memoryTotalMB = memoryTotalMB; self.arsenalMB = arsenalMB
    }
}
public struct GPUProcessActivity: Codable, Sendable, Equatable {
    public var name: String
    public var pct: Double
    /// The sampled GPU process, even when its display name belongs to the responsible app.
    public var pid: Int32?
    public init(name: String, pct: Double, pid: Int32? = nil) { self.name = name; self.pct = pct; self.pid = pid }
}
