// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

/// One app-owned piece of MCP work: a download, a load or a run (which may download and load first). Its id is the
/// handle `arsenal_wait` and `arsenal_cancel` take; its status line is what every "still running" text and progress
/// message says.
actor MCPJob {
    enum Kind: String, Codable, Sendable { case run, download, load }
    enum Status: String, Codable, Sendable { case running, completed, failed, cancelled, interrupted }
    enum Phase: String, Sendable { case queued, copying, downloading, deriving, loading, running }
    nonisolated let id: String
    nonisolated let kind: Kind
    nonisolated let model: String
    nonisolated let name: String
    nonisolated let operation: String?
    nonisolated let dtype: String
    nonisolated let requester: String
    nonisolated let created: Date
    let environment: MCPEnvironment
    private(set) var status = Status.running
    private(set) var phase = Phase.queued
    private(set) var result: CallTool.Result?
    /// The directory holding this job's output files and `output.txt`.
    var directory: URL?
    // Download figures.
    private(set) var totalBytes: Int64?
    private(set) var bytes: Int64 = 0
    private(set) var rates = MCPRateWindow()
    private(set) var downloadStarted: Date?
    private(set) var downloadFinished: Date?
    private(set) var downloadedBytes: Int64 = 0
    /// The derived dtype being made after a download, when there is one.
    var derivingTo: String?
    // Run figures.
    private(set) var step: Double?
    private(set) var steps: Double?
    private(set) var label: String?
    private(set) var phaseStarted: Date
    /// Bumped on every change a progress message would show.
    private(set) var revision = 0
    private var task: Task<Void, Never>?
    private let onCancel: @Sendable (MCPJob) async -> Void
    private var waiters: [UUID: AsyncStream<Void>.Continuation] = [:]

    init(
        kind: Kind, model: String, name: String, operation: String?, dtype: String, requester: String, environment: MCPEnvironment,
        onCancel: @escaping @Sendable (MCPJob) async -> Void = { _ in }
    ) {
        self.onCancel = onCancel
        id = String(UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "").prefix(10))
        self.kind = kind; self.model = model; self.name = name; self.operation = operation; self.dtype = dtype; self.requester = requester
        self.environment = environment; created = environment.now(); phaseStarted = created
    }
    /// The job's task. A job cancelled before its task was attached cancels the task at once.
    func attach(_ value: Task<Void, Never>) {
        guard status == .running else { value.cancel(); return }
        task = value
    }
    func enter(_ next: Phase) {
        guard status == .running, phase != next else { return }
        if next == .downloading, downloadStarted == nil { downloadStarted = environment.now() }
        if phase == .downloading || phase == .deriving, next != .downloading, next != .deriving { downloadFinished = environment.now() }
        phase = next; phaseStarted = environment.now(); step = nil; steps = nil; label = nil; changed()
    }
    /// A Get's row state: bytes so far, total and phase.
    func download(_ state: PrecisionState) {
        guard status == .running else { return }
        if state.state == .deriving || state.phase == "deriving" { enter(.deriving); return }
        if phase != .downloading { enter(.downloading) }
        if let total = state.totalBytes, total > 0 { totalBytes = total }
        if let done = state.bytes, done >= bytes {
            rates.add(environment.now(), done)
            if done > bytes { bytes = done; downloadedBytes = done; changed() }
        }
    }
    func expectDownload(total: Int64?, networkBytes: Int64? = nil) {
        if let total, total > 0 { totalBytes = total }
        fromLocalCopy = networkBytes == 0 && (total ?? 0) > 0
    }
    /// Every byte comes from a copy already on this Mac (the Hugging Face cache, a clone): nothing is downloaded.
    private(set) var fromLocalCopy = false
    /// Input files copied so far (across volumes), and their total.
    private(set) var copied: Int64 = 0
    private(set) var copyTotal: Int64 = 0
    func copying(_ done: Int64, of total: Int64) {
        guard status == .running else { return }
        if phase != .copying { enter(.copying) }
        copyTotal = total
        if done > copied { copied = done; changed() }
    }
    /// The Get returned: the download part of this job is over.
    func finishDownload() { if downloadStarted != nil, downloadFinished == nil { downloadFinished = environment.now() } }
    /// The raw data of a finished run, for `format:"json"`.
    private(set) var output: JSONObject?
    func setOutput(_ value: JSONObject) { output = value }
    func setDirectory(_ value: URL) { directory = value }
    /// A runner's progress frame (`step`/`total`, `label`).
    func runProgress(_ value: JSONObject) {
        guard status == .running else { return }
        if phase != .running { enter(.running) }
        if let current = value["step"]?.number { step = current }
        if let total = value["total"]?.number { steps = total }
        if let text = value["label"]?.string { label = text }
        changed()
    }
    private func changed() { revision += 1 }
    func complete(_ value: CallTool.Result) {
        guard status == .running else { return }
        result = value; status = value.isError == true ? .failed : .completed; task = nil; finishWaiters()
    }
    func cancel() async {
        guard status == .running else { return }
        status = .cancelled; task?.cancel(); task = nil
        await onCancel(self)
        finishWaiters()
    }
    func interrupt() {
        guard status == .running else { return }
        status = .interrupted; task?.cancel(); task = nil; finishWaiters()
    }
    private func finishWaiters() { for waiter in waiters.values { waiter.finish() }; waiters.removeAll() }
    /// Returns when the job ends or `seconds` pass on the job's clock.
    func waitForEnd(seconds: Double) async {
        guard status == .running, seconds > 0 else { return }
        let key = UUID(), stream = AsyncStream<Void> { waiters[key] = $0 }
        let sleep = environment.sleep
        await withTaskGroup(of: Void.self) { group in
            group.addTask { for await _ in stream {} }
            group.addTask { try? await sleep(seconds) }
            _ = await group.next(); group.cancelAll()
        }
        waiters.removeValue(forKey: key)?.finish()
    }
    var isWaitedOn: Bool { !waiters.isEmpty }
    /// True once the download has a measured speed (3 s and 1 MB), has moved past downloading, or the job ended.
    var downloadMeasured: Bool { status != .running || (phase != .queued && phase != .downloading) || rates.rate(at: environment.now()) != nil }
    var hasDownloaded: Bool { downloadStarted != nil }

    /// What the job is doing now, without a trailing period: "Downloading Parakeet v3 Ultra: 1.2 of 2.5 GB, about 2 min left".
    func statusLine() -> String {
        let now = environment.now()
        switch phase {
        case .queued: return "Starting \(activity()) with \(name)"
        case .copying: return "Copying input files for \(name): \(MCPFormat.progress(copied, of: max(copyTotal, 1)))"
        case .downloading:
            guard let total = totalBytes else { return "Downloading \(name): \(MCPFormat.size(bytes)) so far" }
            let figures = MCPFormat.progress(bytes, of: total)
            if rates.stalled(at: now) >= 30, bytes > 0 { return "Downloading \(name): \(figures), no data for \(MCPFormat.elapsed(rates.stalled(at: now)))" }
            guard let rate = rates.rate(at: now), rate > 0 else { return "Downloading \(name): \(figures), measuring speed" }
            return "Downloading \(name): \(figures), \(MCPFormat.remaining(Double(max(0, total - bytes)) / rate)) left at \(MCPFormat.rate(rate))"
        case .deriving:
            return "Converting \(name)" + (derivingTo.map { " to \($0)" } ?? "") + " (time not measured yet)"
        case .loading: return "Loading \(name) into memory for \(MCPFormat.elapsed(now.timeIntervalSince(phaseStarted)))"
        case .running:
            let elapsed = now.timeIntervalSince(phaseStarted)
            if let step, let steps, steps > 0, step > 0 {
                let left = elapsed / step * max(0, steps - step)
                let estimate = left < 60 ? "about \(max(1, Int(left.rounded()))) s" : MCPFormat.remaining(left)
                return "Running \(name): step \(Int(step)) of \(Int(steps)), \(estimate) left"
            }
            return "Running \(name) for \(MCPFormat.elapsed(elapsed)) (no progress estimate for this operation)"
        }
    }
    /// The figure a progress notification carries: bytes while downloading, steps while running.
    func figure() -> (value: Double, total: Double?)? {
        switch phase {
        case .downloading where bytes > 0: return (Double(bytes), totalBytes.map(Double.init))
        case .copying where copied > 0: return (Double(copied), Double(copyTotal))
        case .running where step != nil: return (step!, steps)
        default: return nil
        }
    }
    func activity() -> String {
        switch kind {
        case .download: return "the download"
        case .load: return "the load"
        case .run: return operation.map { "\($0)" } ?? "the run"
        }
    }
    /// "the download of Parakeet v3 Ultra", "transcribe with Parakeet v3 Ultra".
    func title() -> String {
        switch kind {
        case .download: return "download of \(name)"
        case .load: return "load of \(name)"
        case .run: return "\(operation ?? "run") with \(name)"
        }
    }
    /// "Downloaded Parakeet v3 Ultra (2.5 GB in 3 min 40 s)." once a download finished inside this job.
    func downloadSummary() -> String? {
        guard let started = downloadStarted else { return nil }
        let seconds = (downloadFinished ?? environment.now()).timeIntervalSince(started)
        let size = totalBytes ?? downloadedBytes
        // Core review, 2 Nov: a model the user had deleted came back from the Hugging Face cache and was reported as
        // "Downloaded", so the agent reported a consent breach that did not happen.
        if fromLocalCopy {
            return "Restored \(name) from a copy already on this Mac (\(size > 0 ? MCPFormat.size(size) + ", " : "")\(MCPFormat.elapsed(seconds))); nothing was downloaded."
        }
        return "Downloaded \(name) (\(size > 0 ? MCPFormat.size(size) + " in " : "in ")\(MCPFormat.elapsed(seconds)))."
    }
    func downloadMeasurement() -> (bytes: Int64, seconds: Double)? {
        guard let started = downloadStarted, let finished = downloadFinished, let first = rates.first else { return nil }
        return (bytes - first.1, finished.timeIntervalSince(started))
    }
    func record() -> MCPJobRecord {
        .init(
            id: id, kind: kind, model: model, name: name, operation: operation, dtype: dtype, status: status, created: created,
            finished: status == .running ? nil : environment.now(),
            line: result.map { mcpResultText($0).components(separatedBy: "\n").first ?? "" }, output: directory.map { $0.appendingPathComponent("output.txt").path })
    }
}

/// A Get started over MCP and the jobs waiting on it.
struct MCPGet: Sendable {
    let task: Task<Void, any Error>
    var waiters: Set<String>
}

/// A job as persisted in `mcp-jobs.json`: enough for `arsenal_wait` to answer honestly after a restart.
struct MCPJobRecord: Codable, Sendable, Equatable {
    var id: String
    var kind: MCPJob.Kind
    var model: String
    var name: String
    var operation: String?
    var dtype: String
    var status: MCPJob.Status
    var created: Date
    var finished: Date?
    var line: String?
    var output: String?
}

extension APIRouter {
    static let mcpJobLimit = 128
    var mcpJobsFile: URL { support.appendingPathComponent("mcp-jobs.json") }
    /// Terminal records are kept 24 h. A record still running when the file was written was interrupted by a restart.
    func loadMCPJobRecords() {
        guard let data = try? Data(contentsOf: mcpJobsFile), let records = try? JSONDecoder.mcp.decode([MCPJobRecord].self, from: data) else { return }
        let cutoff = mcpEnvironment.now().addingTimeInterval(-86400)
        mcpJobRecords = Dictionary(
            records.filter { ($0.finished ?? $0.created) >= cutoff }.map { record in
                var record = record
                if record.status == .running { record.status = .interrupted; record.finished = mcpEnvironment.now() }
                return (record.id, record)
            }, uniquingKeysWith: { a, _ in a })
        saveMCPJobRecords()
    }
    func saveMCPJobRecords() {
        let cutoff = mcpEnvironment.now().addingTimeInterval(-86400)
        let records = mcpJobRecords.values.filter { ($0.finished ?? $0.created) >= cutoff }.sorted { $0.created < $1.created }
        guard let data = try? JSONEncoder.mcp.encode(records) else { return }
        try? FileManager.default.createDirectory(at: support, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try? data.write(to: mcpJobsFile, options: .atomic)
    }
    func persist(_ job: MCPJob) async {
        mcpJobRecords[job.id] = await job.record(); saveMCPJobRecords()
    }
    /// Starts `work` as a job. Errors become the job's `[category]` result; a cancelled job keeps its cancelled state.
    func startMCPJob(
        kind: MCPJob.Kind, entry: CatalogEntry, operation: String?, dtype: String, requester: String,
        onCancel: @escaping @Sendable (MCPJob) async -> Void = { _ in }, work: @escaping @Sendable (MCPJob) async throws -> CallTool.Result
    ) async throws -> MCPJob {
        // At the limit, only the oldest finished jobs leave memory; their records and output.txt stay readable.
        if mcpJobs.count >= Self.mcpJobLimit {
            var finished: [MCPJob] = []
            for job in mcpJobs.values where await job.status != .running { finished.append(job) }
            for job in finished.sorted(by: { $0.created < $1.created }).prefix(mcpJobs.count - Self.mcpJobLimit + 1) { mcpJobs.removeValue(forKey: job.id) }
        }
        guard mcpJobs.count < Self.mcpJobLimit else {
            throw MCPFailure(.limit, "Not started: \(Self.mcpJobLimit) Arsenal jobs are already running. Cancel one with \(MCPNames.prefix)cancel, or wait.")
        }
        let job = MCPJob(
            kind: kind, model: entry.id, name: entry.name, operation: operation, dtype: dtype, requester: requester, environment: mcpEnvironment, onCancel: onCancel)
        mcpJobs[job.id] = job
        await persist(job)
        let task = Task { [self] in
            let result: CallTool.Result
            do { result = try await work(job) } catch {
                if await job.status != .running { await self.persist(job); return }
                var failure = await self.mcpFailure(error, entry: entry, verb: kind == .load ? "Not loaded" : kind == .download ? "Not downloaded" : "Not run")
                // A refusal after the download finished keeps the files; say so before the reason.
                if kind != .download, await job.downloadFinished != nil, let summary = await job.downloadSummary() {
                    failure = MCPFailure(failure.category, String(summary.dropLast()) + ", kept on this Mac. " + failure.text)
                }
                result = failure.result
            }
            await job.complete(result)
            await self.persist(job)
        }
        await job.attach(task)
        return job
    }
}

extension JSONEncoder {
    static var mcp: JSONEncoder { let e = JSONEncoder(); e.dateEncodingStrategy = .iso8601; e.outputFormatting = [.sortedKeys]; return e }
}
extension JSONDecoder {
    static var mcp: JSONDecoder { let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d }
}
