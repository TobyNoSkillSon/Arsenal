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
    /// What that memory is for, and the most the current or last job took (absent until the runner reports).
    public var memory: ModelMemoryActivity?
    enum CodingKeys: String, CodingKey { case id, kind, revision, state, requests, queued, running, recent, batch, memory; case memoryMB = "memory_mb" }
    public init(
        id: String, kind: String, state: String = "idle", requests: [RequestActivity] = [], queued: Int = 0, running: Int = 0, recent: RecentActivity? = nil,
        batch: Double? = nil, memoryMB: Double? = nil, memory: ModelMemoryActivity? = nil
    ) {
        self.id = id; self.kind = kind; self.state = state; self.requests = requests; self.queued = queued; self.running = running; self.recent = recent
        self.batch = batch; self.memoryMB = memoryMB; self.memory = memory
    }
}
/// A model's memory by what holds it, in MB (10⁶ bytes). Every figure is measured in the runner's process: the process
/// footprint (what Activity Monitor shows for it), MLX's own counters where the runner reports them, the supervisor's
/// footprint samples, twice a second during each job, and the kernel's own footprint maximum over the job. Nothing is
/// estimated; a part not reported is absent.
public struct ModelMemoryActivity: Codable, Sendable, Equatable {
    /// `mlx`: the runner reports MLX's counters, so its memory splits into weights, working memory and cache. `process`:
    /// only the process footprint is known (an engine that does not report MLX's counters).
    public var source: String
    /// The process footprint now (the supervisor's newest sample during a job, else the runner's reading).
    public var totalMB: Double
    /// MLX active memory right after the load: the weights (and whatever else the load keeps alive).
    public var weightsMB: Double?
    /// MLX active memory beyond the weights: activations, KV cache or context, and scratch.
    public var workingMB: Double?
    /// MLX active memory, when the weights were not measured apart (weights and working memory together).
    public var arraysMB: Double?
    /// MLX's buffer cache: freed buffers MLX keeps for reuse; macOS gets them back when the cache is cleared.
    public var cacheMB: Double?
    /// What the process grew by during the running job since the runner's last reading (the supervisor's samples), not
    /// yet split by MLX's counters: a runner that reports only between jobs. Absent when nothing is unsplit.
    public var unsplitMB: Double?
    /// The most RAM the process held during the current or last job (or load): the kernel's maximum footprint over that
    /// job, raised by the supervisor's samples and the runner's readings. Never MLX's peak counter, which also counts
    /// allocator memory the process does not hold (41.2 GB against the kernel's 35.6 GB for one SeedVR2 clip). Absent
    /// before a job.
    public var peakMB: Double?
    /// MLX's own active-memory peak counter in that job, as the runner reported it; kept for clients, not the peak.
    public var mlxPeakMB: Double?
    enum CodingKeys: String, CodingKey {
        case source; case totalMB = "total_mb", weightsMB = "weights_mb", workingMB = "working_mb", arraysMB = "arrays_mb", cacheMB = "cache_mb"
        case unsplitMB = "unsplit_mb", peakMB = "peak_mb", mlxPeakMB = "mlx_peak_mb"
    }
    public init(
        source: String, totalMB: Double, weightsMB: Double? = nil, workingMB: Double? = nil, arraysMB: Double? = nil, cacheMB: Double? = nil,
        unsplitMB: Double? = nil, peakMB: Double? = nil, mlxPeakMB: Double? = nil
    ) {
        self.source = source; self.totalMB = totalMB; self.weightsMB = weightsMB; self.workingMB = workingMB; self.arraysMB = arraysMB; self.cacheMB = cacheMB
        self.unsplitMB = unsplitMB; self.peakMB = peakMB; self.mlxPeakMB = mlxPeakMB
    }
    /// From a runner's reading. A runner that reports no MLX memory (zero active and cache) is `process`. The parts come
    /// from the runner's one reading; the supervisor's newer sample only adds what grew since.
    public init?(_ m: WorkerMemory) {
        guard m.footprintMB > 0 else { return nil }
        let mlx = m.mlxActiveMB > 0 || m.mlxCacheMB > 0
        let total = m.currentMB, unsplit = max(0, total - m.footprintMB)
        let mlxPeak = mlx ? m.mlxPeakMB.flatMap { $0 > 0 ? $0 : nil } : nil
        let peak = m.jobPeakMB.flatMap { $0 > 0 ? $0 : nil }
        let weights = mlx ? m.mlxWeightsMB.flatMap { $0 > 0 ? $0 : nil } : nil
        self.init(
            source: mlx ? "mlx" : "process", totalMB: total, weightsMB: weights, workingMB: weights.map { max(0, m.mlxActiveMB - $0) },
            arraysMB: mlx && weights == nil ? m.mlxActiveMB : nil, cacheMB: mlx ? m.mlxCacheMB : nil, unsplitMB: mlx && unsplit > 0 ? unsplit : nil,
            peakMB: peak.map { max($0, total) }, mlxPeakMB: mlxPeak)
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
    /// Seconds of audio per second of work, over the minute's finished requests that made audio (each request's own run
    /// time, so idle gaps between requests do not count); nil when none finished.
    public var realTime: Double?
    enum CodingKeys: String, CodingKey {
        case finished, failed, tokens, steps, images, busy
        case windowS = "window_s", meanRunS = "mean_run_s", meanWaitS = "mean_wait_s", audioS = "audio_s", busyS = "busy_s", realTime = "real_time"
    }
    public init(
        windowS: Int = 60, finished: Int = 0, failed: Int = 0, meanRunS: Double? = nil, meanWaitS: Double? = nil, tokens: Int = 0, steps: Int = 0, images: Int = 0,
        audioS: Double = 0, busyS: Double = 0, busy: [Double] = [], realTime: Double? = nil
    ) {
        self.windowS = windowS; self.finished = finished; self.failed = failed; self.meanRunS = meanRunS; self.meanWaitS = meanWaitS; self.tokens = tokens
        self.steps = steps; self.images = images; self.audioS = audioS; self.busyS = busyS; self.busy = busy; self.realTime = realTime
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
