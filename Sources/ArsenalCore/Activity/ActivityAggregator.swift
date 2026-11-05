// SPDX-License-Identifier: MIT
import Foundation

/// Short synchronous transactions keep telemetry off the asynchronous frame-delivery path.
private final class ActivityRequestStore: @unchecked Sendable {
    private let lock = NSLock()
    private let now: @Sendable () -> Double
    init(now: @escaping @Sendable () -> Double) { self.now = now }
    private struct Job {
        var model: String
        var request: RequestActivity
        var accepted: Double
        /// Its operation declares audio as its output (speak, generate_sound, enhance, …).
        var audioOutput = false
        /// When its runner reported starting it (a start phase or its first work). The service handing it to the runner
        /// is not a start: a serial runner holds what it was sent until it reaches it.
        var started: Double?
        var tokens = 0, step = 0, samples = 0
        /// This request's audio rate, as its stream announced it (`audio_begin`) or a progress event carried it.
        var sampleRate: Double?
        var audioCredited = 0.0
        /// The audio it credited to each second still in the minute, so its result can correct only its own share.
        var audioBySecond: [Int: Double] = [:]
        var running: Bool { started != nil && request.phase != "done" }
    }
    /// One second of a model's work.
    private struct Second {
        var finished = 0, failed = 0, runs = 0, tokens = 0, steps = 0, images = 0
        var runSum = 0.0, waitSum = 0.0, audio = 0.0
        /// Finished requests that made audio: the audio they made and the seconds they worked, credited together when each
        /// finishes, so the pair enters and leaves the minute at once.
        var speechAudio = 0.0, speechRunSum = 0.0
        /// The part of this second in which at least one of the model's requests was running (0…1).
        var busy = 0.0
        mutating func add(_ o: Second) {
            finished += o.finished; failed += o.failed; runs += o.runs; tokens += o.tokens; steps += o.steps; images += o.images
            runSum += o.runSum; waitSum += o.waitSum; audio += o.audio; busy += o.busy
            speechAudio += o.speechAudio; speechRunSum += o.speechRunSum
        }
    }
    private var jobs: [UUID: Job] = [:]
    private var batches: [String: (Double, Double)] = [:]
    /// Per model, per whole second of uptime: the last minute's work. Counters only, never request contents.
    private var seconds: [String: [Int: Second]] = [:]
    /// Per model: since when at least one of its requests has been running (closed into `seconds` when none is).
    private var busySince: [String: Double] = [:]
    static let window = 60

    /// Adds to a second of a model's minute, the current one by default (call inside the lock).
    private func credit(_ model: String, at second: Int? = nil, _ update: (inout Second) -> Void) {
        let second = second ?? Int(now().rounded(.down))
        var table = seconds[model] ?? [:]
        update(&table[second, default: Second()])
        if table.count > Self.window + 2 { table = table.filter { $0.key >= second - Self.window } }
        seconds[model] = table
    }
    /// The model's busy time from `start` to `end`, split over the seconds it covers (only the last minute is kept).
    private func creditBusy(_ model: String, from start: Double, to end: Double) {
        let start = max(start, end - Double(Self.window + 1))
        guard end > start else { return }
        var second = Int(start.rounded(.down))
        while Double(second) < end {
            let overlap = min(end, Double(second + 1)) - max(start, Double(second))
            if overlap > 0 { credit(model, at: second) { $0.busy += overlap } }
            second += 1
        }
    }
    private func start(_ job: inout Job) {
        guard job.started == nil else { return }
        let time = now()
        job.started = time; job.request.runningSince = Date()
        if busySince[job.model] == nil { busySince[job.model] = time }
    }
    /// Closes the model's busy stretch once none of its requests runs (call after removing a job).
    private func endBusyIfIdle(_ model: String) {
        guard let since = busySince[model], !jobs.values.contains(where: { $0.model == model && $0.started != nil }) else { return }
        creditBusy(model, from: since, to: now()); busySince[model] = nil
    }
    /// Scales the audio a request credited to the seconds still in the minute by `factor` (its result says it made less:
    /// the stream announced the wrong rate). Seconds that have left the minute and other requests' audio stay as they are.
    private func rescaleAudio(_ job: Job, by factor: Double) {
        let oldest = Int(now().rounded(.down)) - Self.window
        for (second, amount) in job.audioBySecond where second >= oldest {
            credit(job.model, at: second) { $0.audio = max(0, $0.audio - amount * (1 - factor)) }
        }
    }
    private static func positive(_ value: JSONValue?) -> Double? { value?.number.flatMap { $0.isFinite && $0 > 0 && $0 < 1e7 ? $0 : nil } }

    func begin(model: String, op: String, requester: String, title: String? = nil, audioOutput: Bool = false) -> UUID {
        return lock.withLock {
            var request = RequestActivity(requester: requester, op: op)
            request.title = title
            jobs[request.id] = Job(model: model, request: request, accepted: now(), audioOutput: audioOutput); return request.id
        }
    }
    /// The service's own view of a request: waiting for memory or a load, or handed to the runner.
    func phase(_ id: UUID, _ phase: String) { lock.withLock { jobs[id]?.request.phase = phase } }
    func receive(_ id: UUID, _ frame: WireFrame) {
        lock.withLock {
            guard var job = jobs[id], case .json(let event) = frame else { return }
            switch event["event"]?.string {
            case "audio_begin":
                if let rate = Self.positive(event["sample_rate"]) { job.sampleRate = rate; jobs[id] = job }
                return
            case "progress": break
            default: return
            }
            for key in ["label", "unit"] {
                if let text = event[key]?.string, text.count <= 128 {
                    if key == "label" { job.request.label = text } else { job.request.unit = text }
                }
            }
            if let rate = Self.positive(event["sample_rate"]) { job.sampleRate = rate }
            if let phase = event["phase"]?.string {
                job.request.phase = phase
                if !["queued", "loading"].contains(phase) { start(&job) }
            }
            func count(_ key: String) -> Int? { event[key]?.number.flatMap { Int(exactly: $0) }.flatMap { $0 >= 0 ? $0 : nil } }
            if let value = count("prompt_tokens") { job.request.promptTokens = value }
            if let value = count("cached_tokens") { job.request.cachedTokens = value }
            if let value = count("generated_tokens") {
                start(&job)
                let added = max(0, value - job.tokens)
                if added > 0 { credit(job.model) { $0.tokens += added } }
                job.tokens = value; job.request.generatedTokens = value
            }
            if let value = count("step") {
                start(&job)
                // A step count that starts again is a new stage: its steps are new work.
                let added = value >= job.step ? value - job.step : value
                if added > 0 { credit(job.model) { $0.steps += added } }
                job.step = value; job.request.step = value
            }
            if let value = count("total") { job.request.total = value }
            if let value = count("audio_samples") {
                start(&job)
                let added = max(0, value - job.samples)
                job.samples = value
                if let rate = job.sampleRate {
                    let produced = Double(added) / rate
                    job.audioCredited += produced; job.request.audioS = Double(value) / rate
                    if produced > 0 {
                        let second = Int(now().rounded(.down))
                        credit(job.model, at: second) { $0.audio += produced }
                        job.audioBySecond[second, default: 0] += produced
                        if job.audioBySecond.count > Self.window + 2 { job.audioBySecond = job.audioBySecond.filter { $0.key >= second - Self.window } }
                    }
                }
            }
            if let timing = event["timing"]?.object {
                job.request.timing = timing; job.request.tps = timing["decode_tps"]?.number ?? timing["prefill_tps"]?.number
            }
            if let batch = event["batch"]?.number, batch.isFinite, batch >= 1 { batches[job.model] = (batch, now()) }
            jobs[id] = job
        }
    }
    var active: Bool { lock.withLock { !jobs.isEmpty } }
    var count: Int { lock.withLock { jobs.count } }
    /// Models with any kept history (tests check it is pruned).
    var trackedModels: Int { lock.withLock { Set(seconds.keys).union(batches.keys).union(busySince.keys).count } }
    func finish(_ id: UUID, _ outcome: ActivityAggregator.Outcome) {
        lock.withLock {
            guard let job = jobs.removeValue(forKey: id) else { return }
            defer { endBusyIfIdle(job.model) }
            switch outcome {
            case .cancelled: return
            case .failed: credit(job.model) { $0.failed += 1 }
            case .finished(let result):
                let end = now()
                // Speech and sound return the audio they made at the top; a voice design inside its preview. A
                // `duration_s` is audio made only beside its format (`sample_rate`), after the request streamed audio, or
                // from an operation that declares audio output. An analysis result's `duration_s` (a classifier's is the
                // clip it listened to) is none of these: it never counts as audio made, so it gives no real-time figure.
                let madeAudio = job.audioOutput || job.samples > 0 || job.audioCredited > 0
                let audio = [result, result?["preview"]?.object].compactMap { $0 }.first {
                    Self.positive($0["duration_s"]) != nil && (madeAudio || Self.positive($0["sample_rate"]) != nil)
                }
                let rate = Self.positive(audio?["sample_rate"]) ?? job.sampleRate
                let duration = Self.positive(audio?["duration_s"]) ?? rate.flatMap { job.samples > 0 ? Double(job.samples) / $0 : nil }
                let more = (duration ?? job.audioCredited) - job.audioCredited
                if more < 0, let duration { rescaleAudio(job, by: duration / job.audioCredited) }
                let images = result?["images"]?.array?.count ?? 0
                credit(job.model) { s in
                    s.finished += 1; s.audio += max(0, more); s.images += images
                    if let started = job.started { s.runs += 1; s.runSum += max(0, end - started); s.waitSum += max(0, started - job.accepted) }
                    if let started = job.started, let made = duration ?? (job.audioCredited > 0 ? job.audioCredited : nil), end > started {
                        s.speechAudio += made; s.speechRunSum += end - started
                    }
                }
            }
        }
    }
    /// Drops minutes that have passed, batches no runner has reported for five seconds, and the history of models that
    /// are gone (call inside the lock).
    private func prune(rows: [ModelRow], current: Int) {
        let known = Set(rows.map(\.id)), working = Set(jobs.values.map(\.model))
        for (model, table) in seconds {
            let kept = table.filter { $0.key >= current - Self.window }
            let gone = !known.contains(model) && !working.contains(model)
            seconds[model] = kept.isEmpty || gone ? nil : kept
        }
        for model in busySince.keys where !known.contains(model) && !working.contains(model) { busySince[model] = nil }
        batches = batches.filter { now() - $0.value.1 < 5 }
    }
    func snapshot(rows: [ModelRow], memory: MemorySnapshot, machine: MachineActivity) -> ActivitySnapshot {
        return lock.withLock {
            let current = Int(now().rounded(.down))
            prune(rows: rows, current: current)
            // Include queued on-demand models even before they become resident.
            let visible = rows.filter { row in row.residentDType != nil || row.state == .loading || jobs.values.contains { $0.model == row.id } }
            return .init(
                models: visible.map { row in
                    let own = jobs.values.filter { $0.model == row.id }.sorted { ($0.accepted, $0.request.startedAt) < ($1.accepted, $1.request.startedAt) }
                    let requests = own.map(Self.shown)
                    let active = requests.first { Self.working($0.phase) }
                    let fallback =
                        own.isEmpty
                        ? (row.state == .loading ? (row.states[row.selectedPrecision]?.phase == "queued" ? "queued" : "loading") : "idle")
                        : own.allSatisfy({ $0.request.phase == "done" }) ? "done" : requests.contains { $0.phase == "loading" } ? "loading" : "queued"
                    var result = ModelActivity(
                        id: row.id, kind: row.kind, state: active?.phase ?? fallback,
                        requests: requests, queued: own.filter { $0.started == nil }.count, running: own.filter(\.running).count,
                        recent: recent(row.id, current: current),
                        batch: active == nil ? nil : batches[row.id]?.0,
                        memoryMB: memory.models[row.id]?.currentMB, memory: memory.models[row.id].flatMap(ModelMemoryActivity.init))
                    result.revision = row.revision
                    if row.revision?.swapping == true { result.state = "reloading" }
                    return result
                }, machine: machine)
        }
    }
    /// A phase in which the runner works on the request (not waiting, loading or finished).
    static func working(_ phase: String) -> Bool { !["queued", "loading", "done"].contains(phase) }
    /// A request as the panel and API show it: until its runner reports a start it waits, whatever the service did with
    /// it ("loading" while its model loads, "queued" otherwise).
    private static func shown(_ job: Job) -> RequestActivity {
        var request = job.request
        if job.started == nil { request.phase = request.phase == "loading" ? "loading" : "queued" }
        return request
    }
    /// The model's last minute (whole seconds before the current one); nil when nothing ran, finished or progressed in it.
    private func recent(_ model: String, current: Int) -> RecentActivity? {
        let table = seconds[model] ?? [:]
        let first = current - Self.window
        var series = (first..<current).map { table[$0] ?? Second() }
        // The stretch still running counts up to the current second.
        if let since = busySince[model] {
            for i in series.indices {
                let overlap = min(Double(first + i + 1), Double(current)) - max(Double(first + i), since)
                if overlap > 0 { series[i].busy += overlap }
            }
        }
        var total = Second()
        for second in series { total.add(second) }
        guard total.finished + total.failed + total.tokens + total.steps > 0 || total.audio > 0 || total.busy > 0 else { return nil }
        return RecentActivity(
            windowS: Self.window, finished: total.finished, failed: total.failed, meanRunS: total.runs > 0 ? total.runSum / Double(total.runs) : nil,
            meanWaitS: total.runs > 0 ? total.waitSum / Double(total.runs) : nil, tokens: total.tokens, steps: total.steps, images: total.images,
            audioS: total.audio, busyS: total.busy, busy: series.map { min(1, max(0, $0.busy)) },
            realTime: total.speechAudio > 0 && total.speechRunSum > 0 ? total.speechAudio / total.speechRunSum : nil)
    }
}

/// Only subscribers create sampling and progress tasks. IORegistry work runs on a separate sampler actor.
public actor ActivityAggregator {
    /// How a request ended: finished with its result, failed, or cancelled (not counted).
    public enum Outcome: Sendable { case finished(JSONObject?), failed, cancelled }
    private struct Subscriber {
        var continuation: AsyncStream<ActivitySnapshot>.Continuation
        /// Wants the machine's readings (the Activity panel, API clients); a tool window following its own request does
        /// not.
        var machine: Bool
    }
    private nonisolated let requests: ActivityRequestStore
    private var subscribers: [UUID: Subscriber] = [:]
    /// The subscription session (any subscriber), and the machine readings within it (machine subscribers only).
    private var generation: UUID?
    private var machineTask: Task<Void, Never>?
    private var machineGeneration: UUID?
    private var requestTask: Task<Void, Never>?
    private var requestGeneration: UUID?
    private var snapshotProvider: (@Sendable () async -> ActivitySnapshot)?
    private let sampler: ActivityMachineSampler
    private let sleep: @Sendable (Duration) async throws -> Void
    private nonisolated let now: @Sendable () -> Double
    private var machine = MachineActivity()
    private var lastPublished: ActivitySnapshot?
    /// The counts and minute last published in full, and when: between full publishes, request ticks carry progress only.
    private var held: (at: Double, models: [String: ModelActivity])?
    /// A request started or ended since the last full publish: the request clock runs until one carries it.
    private var boundaryPending = false
    private(set) var requestTickCount = 0
    public private(set) var sampleCount = 0
    /// Snapshots published while subscribed (tests count redraw sources).
    public private(set) var publishCount = 0
    public var subscriberCount: Int { subscribers.count }
    /// How often the machine is read while Activity is open.
    public nonisolated let machineInterval: Duration
    /// How often running requests' progress is published (tool windows follow a short operation's steps through it).
    static let requestInterval: Duration = .milliseconds(250)
    /// Counts and the minute change at most this often (the user reads them, not watch them flicker); progress bars keep the
    /// request clock.
    static let countsInterval = 1.0
    public init(sample: (@Sendable () -> MachineActivity)? = nil, machineInterval: Duration = .seconds(1)) {
        sampler = ActivityMachineSampler(fixture: sample); self.machineInterval = machineInterval
        sleep = { try await Task.sleep(for: $0) }
        now = { ProcessInfo.processInfo.systemUptime }
        requests = ActivityRequestStore(now: now)
    }
    /// Internal clock seam: tests advance the sampling loop and the request clock without wall-clock sleeps.
    init(
        sample: @escaping @Sendable () -> MachineActivity, machineInterval: Duration, sleep: @escaping @Sendable (Duration) async throws -> Void,
        now: @escaping @Sendable () -> Double = { ProcessInfo.processInfo.systemUptime }
    ) {
        sampler = ActivityMachineSampler(fixture: sample); self.machineInterval = machineInterval; self.sleep = sleep; self.now = now
        requests = ActivityRequestStore(now: now)
    }
    /// `audioOutput`: the operation declares audio as its output, so a `duration_s` in its result is audio it made.
    public nonisolated func begin(model: String, op: String, requester: String, title: String? = nil, audioOutput: Bool = false) -> UUID {
        let id = requests.begin(model: model, op: op, requester: requester, title: title, audioOutput: audioOutput)
        changed(); return id
    }
    public nonisolated func phase(_ id: UUID, _ phase: String) { requests.phase(id, phase) }
    public nonisolated func receive(_ id: UUID, _ frame: WireFrame) { requests.receive(id, frame) }
    public nonisolated func finish(_ id: UUID, _ outcome: Outcome = .finished(nil)) { requests.finish(id, outcome); changed() }
    /// Requests being tracked (tests).
    nonisolated var requestCount: Int { requests.count }
    var trackedModels: Int { requests.trackedModels }
    /// Request boundaries and service changes wake subscribers without a polling clock.
    public nonisolated func changed() { Task { await refreshRequests() } }
    public func snapshot(rows: [ModelRow], memory: MemorySnapshot) -> ActivitySnapshot { requests.snapshot(rows: rows, memory: memory, machine: machine) }
    /// `machine`: also read the machine every second while subscribed. A subscriber that only follows requests (a tool
    /// window) leaves it off, so closing Activity stops the readings whatever else is open.
    public func stream(machine: Bool = true, snapshot: @escaping @Sendable () async -> ActivitySnapshot) -> AsyncStream<ActivitySnapshot> {
        let id = UUID(), (stream, continuation) = AsyncStream<ActivitySnapshot>.makeStream(bufferingPolicy: .bufferingNewest(1))
        subscribers[id] = Subscriber(continuation: continuation, machine: machine)
        if let lastPublished { continuation.yield(lastPublished) }
        continuation.onTermination = { [weak self] _ in Task { await self?.unsubscribe(id) } }
        let token: UUID
        if let generation {
            token = generation
        } else {
            token = UUID(); generation = token; snapshotProvider = snapshot
        }
        if machine { startMachine(token) } else { Task { await self.refresh(token, full: true) } }
        armRequestTicks(token)
        return stream
    }
    private var wantsMachine: Bool { subscribers.values.contains { $0.machine } }
    private func startMachine(_ token: UUID) {
        guard machineTask == nil else { return }
        let machineToken = UUID(); machineGeneration = machineToken
        let interval = machineInterval
        machineTask = Task { [weak self, sampler, sleep] in
            var reset = true
            while !Task.isCancelled {
                let sampled = await sampler.sample(reset: reset); reset = false
                guard let self, await self.accept(sampled, machineToken) else { return }
                await self.refresh(token, full: true)
                do { try await sleep(interval) } catch { return }
            }
        }
    }
    /// A request starting or ending publishes at once when the counts last changed a second or more ago; otherwise the
    /// request clock carries it at the next whole second, so ten short requests in a second publish twice, not twenty
    /// times. While requests run the clock carries every change (progress four times a second, counts once a second); once
    /// none runs and the last end is published, it stops.
    private func refreshRequests() async {
        guard let token = generation else { return }
        if requests.active, requestTask != nil { boundaryPending = true; return }
        if countsDue {
            if requests.active { armRequestTicks(token) } else { stopRequestTicks() }
            await refresh(token, full: true)
        } else {
            boundaryPending = true
            armRequestTicks(token)
        }
    }
    /// The counts may change: a second has passed since they last did (or nothing was published yet).
    private var countsDue: Bool { held.map { now() - $0.at >= Self.countsInterval } ?? true }
    private func stopRequestTicks() { requestTask?.cancel(); requestTask = nil; requestGeneration = nil }
    private func armRequestTicks(_ token: UUID) {
        guard requestTask == nil, requests.active || boundaryPending else { return }
        let requestToken = UUID(); requestGeneration = requestToken
        requestTask = Task { [weak self, sleep] in
            while !Task.isCancelled {
                do { try await sleep(Self.requestInterval) } catch { return }
                guard let self, await self.tick(token, requestToken: requestToken) else { return }
            }
        }
    }
    private func tick(_ token: UUID, requestToken: UUID) async -> Bool {
        guard generation == token, requestGeneration == requestToken, !subscribers.isEmpty else { return false }
        if requests.active {
            requestTickCount += 1
            await refresh(token, full: false); return true
        }
        // Nothing runs: wait for the second, publish the last end once, then stop.
        guard boundaryPending, !countsDue else {
            requestTask = nil; requestGeneration = nil
            if boundaryPending { requestTickCount += 1; await refresh(token, full: true) }
            return false
        }
        return true
    }
    /// `full`: counts and the minute as they are now; otherwise (a request tick within a second of the last full
    /// publish) progress only, with the counts and minute held.
    private func refresh(_ token: UUID, full: Bool) async {
        guard generation == token, !subscribers.isEmpty, let snapshotProvider else { return }
        var value = await snapshotProvider()
        let time = now()
        if full || held.map({ time - $0.at >= Self.countsInterval }) ?? true {
            held = (time, Dictionary(value.models.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })); boundaryPending = false
        } else if let held {
            for i in value.models.indices {
                guard let kept = held.models[value.models[i].id] else { continue }
                value.models[i].queued = kept.queued; value.models[i].running = kept.running; value.models[i].recent = kept.recent
            }
        }
        publish(value, generation: token)
    }
    private func accept(_ value: MachineActivity, _ machineToken: UUID) -> Bool {
        guard machineGeneration == machineToken, wantsMachine else { return false }
        machine = value; sampleCount += 1; return true
    }
    private func publish(_ value: ActivitySnapshot, generation token: UUID) {
        guard generation == token, !subscribers.isEmpty, value != lastPublished else { return }
        lastPublished = value; publishCount += 1
        for subscriber in subscribers.values { subscriber.continuation.yield(value) }
    }
    private func unsubscribe(_ id: UUID) {
        subscribers[id] = nil
        if !wantsMachine {
            machineTask?.cancel(); machineTask = nil; machineGeneration = nil; machine = .init()
        }
        if subscribers.isEmpty {
            requestTask?.cancel(); requestTask = nil
            generation = nil; requestGeneration = nil; snapshotProvider = nil; lastPublished = nil; held = nil; boundaryPending = false
        }
    }
}

extension ModelsService {
    public func activity() async -> ActivitySnapshot {
        let rows = await rows(), memory = await memory()
        return .init(
            models: rows.filter { $0.residentDType != nil }.map {
                .init(id: $0.id, kind: $0.kind, memoryMB: memory.models[$0.id]?.currentMB, memory: memory.models[$0.id].flatMap(ModelMemoryActivity.init))
            })
    }
    /// The Activity panel's stream: requests and the machine's readings.
    public func activityStream() async -> AsyncStream<ActivitySnapshot> { await activityStream(machine: true) }
    public func activityStream(machine: Bool) async -> AsyncStream<ActivitySnapshot> {
        let value = await activity()
        return AsyncStream {
            $0.yield(value); $0.finish()
        }
    }
}
