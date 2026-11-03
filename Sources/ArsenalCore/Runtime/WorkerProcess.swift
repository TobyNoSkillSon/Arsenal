// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import ArsenalWire

public struct WorkerLaunch: Sendable {
    public var executable: URL; public var arguments: [String]; public var kind: String; public var runner: String?; public var environment: [String: String]; public var log: URL
    public init(executable: URL, arguments: [String] = [], kind: String, runner: String? = nil, environment: [String: String] = ProcessInfo.processInfo.environment, log: URL) {
        self.executable = executable; self.arguments = arguments; self.kind = kind; self.runner = runner; self.environment = environment; self.log = log
    }
}
public typealias ProgressTransform = @Sendable (JSONObject) -> JSONObject

public actor WorkerProcess {
    public private(set) var hello: WorkerHello?
    public private(set) var memory = WorkerMemory()
    public private(set) var exited = false
    public private(set) var pid: Int32 = 0
    private nonisolated let telemetryPIDToken = UUID()
    private var child: ChildProcess?
    private var nextID = 1
    private var error: WorkerFailure?
    private var stopping = false
    private var reader: Task<Void, Never>?
    public private(set) var unexpectedExit = false
    public private(set) var successfulRequests = 0
    private(set) var componentCaptureRequests = 0
    private let cancellationGrace: Double
    private struct Pending {
        var reply: CheckedContinuation<JSONObject, any Error>?
        var progress: ProgressTransform?
        var sink: RunSink?
        var timer: Task<Void, Never>
        let stall: Double
        let generation: UUID
        let op: String
    }
    private var pending: [Int: Pending] = [:]
    private struct Delivery { let token: UUID; let task: Task<Void, Never>; let reply: CheckedContinuation<Void, any Error> }
    private var deliveries: [Int: Delivery] = [:]
    private struct ExitWaiter {
        let reply: CheckedContinuation<Void, Never>
        var timer: Task<Void, Never>?
    }
    private var exitWaiters: [UUID: ExitWaiter] = [:]
    private var streams: [UInt32: Int] = [:]
    private var statusDetailsSink: (@Sendable (JSONObject) async -> Void)?
    private var statusSink: (@Sendable (WorkerMemory) async -> Void)?
    /// What each pending request is expected to use (`WorkerMemoryCeiling`; requests without one get `defaultCeiling`),
    /// and the sampler that enforces the memory stops while any request runs.
    private var ceilings: [Int: WorkerMemoryCeiling] = [:]
    private var memoryWatch: Task<Void, Never>?
    /// Footprint sampling period while a request runs; nothing is sampled while idle. A runaway seen on 20 Oct grew
    /// about 1 GB/s, so half a second bounds the overshoot to about half a gigabyte for two syscalls per tick.
    private let memorySampleSeconds: Double
    public private(set) var memoryCeilingStops = 0
    private var memoryStop = false
    /// The Mac-wide stop this worker takes part in (`MacMemoryGuard.shared` outside tests).
    private let macMemory: MacMemoryGuard
    private nonisolated let memoryToken = UUID()
    private var defaultCeiling = WorkerMemoryCeiling.unmetered("the worker")
    public init(cancellationGrace: Double = 2, memorySampleSeconds: Double = 0.5, macMemory: MacMemoryGuard = .shared) {
        self.cancellationGrace = max(0.01, cancellationGrace); self.memorySampleSeconds = max(0.01, memorySampleSeconds); self.macMemory = macMemory
    }
    deinit { WorkerPIDRegistry.remove(telemetryPIDToken); macMemory.leave(memoryToken); child?.signal(SIGKILL) }
    public func start(
        _ launch: WorkerLaunch, handshakeTimeout: Double = 10, status: (@Sendable (WorkerMemory) async -> Void)? = nil, statusDetails: (@Sendable (JSONObject) async -> Void)? = nil
    ) async throws {
        guard child == nil else { throw WorkerFailure(code: "already_started", message: "Worker already started.") }
        let spawned = try ChildProcess(executable: launch.executable, arguments: launch.arguments, environment: RunnerEnvironment.scrub(launch.environment), log: launch.log)
        WorkerPIDRegistry.register(spawned.pid, token: telemetryPIDToken)
        child = spawned; pid = spawned.pid; statusSink = status; statusDetailsSink = statusDetails
        defaultCeiling = .unmetered("the \(launch.kind) worker")
        reader = Task.detached { [weak self] in
            defer { spawned.finishOutput() }
            do {
                while let packet = try await spawned.readPacket() { try await self?.receive(packet.frame, rawJSON: packet.rawJSON, launch: launch) }
                await self?.connectionEnded()
            } catch {
                // Handshake/message validation can also fail after a well-framed packet.
                spawned.finishOutput()
                await self?.fail(WorkerFailure(code: "wire", message: "Worker sent an invalid frame."))
            }
        }
        Task.detached { [weak self] in
            let code = await spawned.awaitExitAsync(); await self?.didExit(code)
        }
        let deadline = Date().addingTimeInterval(handshakeTimeout)
        while hello == nil {
            if let error { throw error }
            if exited { throw WorkerFailure(code: "worker_exit", message: "Worker exited before hello.") }
            guard Date() < deadline else { await fail(WorkerFailure(code: "hello_timeout", message: "Worker did not send hello in time.")); throw self.error! }
            try await Task.sleep(for: .milliseconds(10))
        }
    }
    public func verifyOperations(_ operations: [String]) throws {
        guard let hello, Set(operations).isSubset(of: Set(hello.ops)) else { throw WorkerFailure(code: "handshake", message: "Worker hello omits declared operations.") }
    }
    public func request(
        op: String, params: JSONObject = [:], timeout: Double = 120, firstResponseTimeout: Double? = nil, progress: ProgressTransform? = nil, sink: RunSink? = nil,
        memoryCeiling: WorkerMemoryCeiling? = nil
    )
        async throws -> JSONObject
    {
        guard hello != nil, !exited, error == nil, let child else { throw error ?? WorkerFailure(code: "not_running", message: "Worker is not running.") }
        try Task.checkCancellation()
        let first = firstResponseTimeout ?? timeout
        guard timeout.isFinite, timeout > 0, first.isFinite, first > 0, pending.count < 64, nextID <= Int(Int32.max) else {
            throw WorkerFailure(code: "busy", message: "Runner request queue is full or its deadline is invalid.")
        }
        if op == "measurement_components" || op == "measurement_voice_patch" { componentCaptureRequests += 1 }
        let id = nextID; nextID += 1; let deadline = Date().addingTimeInterval(first)
        // Every request takes part in the Mac-wide stop; a ceiling adds the model's figures and its runaway backstop.
        ceilings[id] = memoryCeiling ?? defaultCeiling; watchMemory()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let generation = UUID()
                pending[id] = Pending(
                    reply: continuation, progress: progress, sink: sink, timer: deadlineTask(id, after: first, generation: generation), stall: timeout, generation: generation,
                    op: op)
                Task.detached { [weak self] in
                    do { try await processIO { try child.write(.json(WorkerRequest(id: id, op: op, params: params).json), deadline: deadline) } } catch {
                        await self?.fail(WorkerFailure(code: "write", message: "Worker request could not be written."))
                    }
                }
            }
        } onCancel: {
            Task { await self.cancel(id) }
        }
    }
    private func deadlineTask(_ id: Int, after seconds: Double, generation: UUID) -> Task<Void, Never> {
        Task { [weak self] in do { try await Task.sleep(for: .seconds(seconds)); await self?.timeout(id, generation: generation) } catch {} }
    }
    private func refresh(_ id: Int) {
        guard var p = pending[id], p.reply != nil else { return }
        p.timer.cancel()
        let generation = UUID()
        p = Pending(
            reply: p.reply, progress: p.progress, sink: p.sink, timer: deadlineTask(id, after: p.stall, generation: generation), stall: p.stall, generation: generation, op: p.op)
        pending[id] = p
    }
    private func cancel(_ id: Int, reason: any Error = CancellationError()) async {
        guard var p = pending[id], let reply = p.reply else { return }
        p.timer.cancel(); p.reply = nil; p.sink = nil
        stopDelivery(id)
        let generation = UUID()
        pending[id] = Pending(
            reply: nil, progress: nil, sink: nil, timer: deadlineTask(id, after: cancellationGrace, generation: generation), stall: p.stall, generation: generation, op: p.op)
        reply.resume(throwing: reason)
        // A cancel acknowledgement is NOT the target's terminal frame. Its drain record remains armed.
        if !stopping && !exited {
            Task { [weak self] in _ = try? await self?.request(op: "cancel", params: ["target": .number(Double(id))], timeout: self?.cancellationGrace ?? 2) }
        }
    }
    private func timeout(_ id: Int, generation: UUID) async {
        guard let p = pending[id], p.generation == generation else { return }
        if p.reply == nil || p.op == "cancel" || p.op == "unload" {
            await fail(WorkerFailure(code: "cancel_timeout", message: "Worker did not finish cancellation in time."))
        } else {
            await cancel(id, reason: WorkerFailure(code: "timeout", message: "Worker request stalled."))
        }
    }
    private func stopDelivery(_ id: Int) {
        guard let delivery = deliveries.removeValue(forKey: id) else { return }
        delivery.task.cancel(); delivery.reply.resume(throwing: CancellationError())
    }
    private func finishDelivery(_ id: Int, token: UUID, error: (any Error)?) {
        guard let delivery = deliveries[id], delivery.token == token else { return }
        deliveries[id] = nil
        if let error { delivery.reply.resume(throwing: error) } else { delivery.reply.resume() }
    }
    private func deliver(_ frame: WireFrame, id: Int) async {
        // Admission/prefill progress is not a first generated result. Keep the longer first-result deadline armed.
        let awaitingFirstResult: Bool
        if case .json(let event) = frame, event["event"]?.string == "progress" {
            awaitingFirstResult = ["queued", "loading", "prefill"].contains(event["phase"]?.string ?? "")
        } else {
            awaitingFirstResult = false
        }
        if !awaitingFirstResult { refresh(id) }
        var delivered = frame
        if case .json(let event) = frame, event["event"]?.string == "progress", let progress = pending[id]?.progress { delivered = .json(progress(event)) }
        guard let sink = pending[id]?.sink else { return }
        let sinkFrame = delivered
        do {
            try await withCheckedThrowingContinuation { (reply: CheckedContinuation<Void, any Error>) in
                let token = UUID()
                let task = Task { [weak self] in
                    do { try await sink(sinkFrame); await self?.finishDelivery(id, token: token, error: nil) } catch { await self?.finishDelivery(id, token: token, error: error) }
                }
                deliveries[id] = Delivery(token: token, task: task, reply: reply)
            }
        } catch { await cancel(id, reason: error) }
    }
    private func receive(_ frame: WireFrame, rawJSON: Data?, launch: WorkerLaunch) async throws {
        if hello == nil {
            guard case .json(let object) = frame else { throw WireError.invalidHandshake }
            let h = try JSONDecoder().decode(WorkerHello.self, from: WireJSON.encode(object)); try h.validate(kind: launch.kind, runner: launch.runner)
            guard h.pid == Int(pid) else { throw WireError.invalidHandshake }; hello = h; return
        }
        switch frame {
        case .json(let object):
            if object["event"]?.string == "status", object["id"] == nil {
                if let m = object["memory"]?.object { memory = try JSONDecoder().decode(WorkerMemory.self, from: WireJSON.encode(m)); await statusSink?(memory) }
                await statusDetailsSink?(object); return
            }
            guard let value = object["id"]?.number, value.isFinite, value >= 1, value <= Double(Int32.max), value.rounded() == value else { throw WireError.malformedFrame }
            let id = Int(value); guard let p = pending[id] else { if object["event"] == nil { streams = streams.filter { $0.value != id } }; return }
            if object["event"]?.string == "progress" {
                for key in ["label", "unit"] where object[key] != nil {
                    guard let text = object[key]?.string, text.count <= 128 else { throw WireError.malformedFrame }
                }
            }
            if object["event"] != nil {
                if let stream = object["stream"]?.number, stream >= 0, stream <= Double(UInt32.max), stream.rounded() == stream { streams[UInt32(stream)] = id }
                await deliver(frame, id: id)
            } else {
                pending[id] = nil; ceilings[id] = nil; p.timer.cancel(); streams = streams.filter { $0.value != id }
                guard let reply = p.reply else { return }
                if object["ok"]?.bool == true {
                    if !["cancel", "unload"].contains(p.op) { successfulRequests += 1 }
                    var result = object
                    if launch.kind == "decisions", let rawJSON {
                        result["_raw_frame"] = .string(String(decoding: rawJSON, as: UTF8.self))
                        result["_raw_body"] = .string(String(decoding: try RawJSONObject.removing(["id", "ok", "timing"], from: rawJSON), as: UTF8.self))
                    }
                    reply.resume(returning: result)
                } else {
                    let details = object["error"]?.object ?? [:]
                    let rawError = try rawJSON.flatMap { try RawJSONObject.member("error", from: $0) }
                    let rawBody = try rawError.flatMap { try RawJSONObject.member("body", from: $0) }.map { String(decoding: $0, as: UTF8.self) }
                    reply.resume(
                        throwing: WorkerFailure(
                            code: details["code"]?.string ?? "worker", message: details["message"]?.string ?? "Worker request failed.", details: details, rawBody: rawBody,
                            rawFrame: rawJSON.map { String(decoding: $0, as: UTF8.self) }))
                }
            }
        case .binary(let stream, _): guard let id = streams[stream] else { throw WireError.malformedFrame }; if pending[id] != nil { await deliver(frame, id: id) }
        }
    }
    private func connectionEnded() async {
        // A memory-ceiling stop fails its requests itself, once the worker has exited.
        if memoryStop { return }
        if !pending.isEmpty {
            let cancelling = pending.values.contains { $0.reply == nil }
            await fail(WorkerFailure(code: cancelling ? "cancelled_exit" : "worker_exit", message: "Worker connection closed."))
        }
    }
    private func didExit(_ code: Int32) async {
        WorkerPIDRegistry.remove(telemetryPIDToken)
        // A memory stop leaves the Mac-wide guard itself, after failing its requests.
        if !memoryStop { endMemoryWatch() }
        unexpectedExit =
            unexpectedExit
            || (!stopping && !["timeout", "cancel_timeout", "cancelled_exit", "write", "memory"].contains(error?.code ?? "") && !pending.values.contains(where: { $0.reply == nil })
                && code != 0)
        exited = true; memory = WorkerMemory()
        for id in Array(exitWaiters.keys) { finishExitWait(id) }
        await statusSink?(memory)
    }
    private func failPending(_ failure: WorkerFailure) {
        let jobs = pending; pending.removeAll(); ceilings.removeAll(); streams.removeAll(); for id in Array(deliveries.keys) { stopDelivery(id) }
        for p in jobs.values { p.timer.cancel(); p.reply?.resume(throwing: failure) }
    }
    private func fail(_ failure: WorkerFailure) async {
        if error == nil { error = failure }; if ["wire", "worker_exit"].contains(error?.code ?? "") && !stopping { unexpectedExit = true }
        failPending(error ?? failure); await terminate()
    }
    /// One sampler per worker, alive only while a request is pending. It joins the Mac-wide guard with the footprint
    /// the worker had when the first of its current requests began.
    private func watchMemory() {
        guard memoryWatch == nil, !exited, !memoryStop else { return }
        let seconds = memorySampleSeconds
        macMemory.join(memoryToken, ceiling: combinedCeiling(), startBytes: WorkerMemoryCeiling.footprint(pid: pid)) { [weak self] failure in
            Task { await self?.stopForMemory(failure) }
        }
        memoryWatch = Task { [weak self] in
            while true {
                do { try await Task.sleep(for: .seconds(seconds)) } catch { return }
                guard let self, await self.checkMemoryCeiling() else { return }
            }
        }
    }
    private func endMemoryWatch() {
        memoryWatch?.cancel(); memoryWatch = nil
        macMemory.leave(memoryToken)
    }
    /// The figures the Mac-wide guard ranks this worker by: the largest expected peak of its pending requests, and Load
    /// Anyway only when every pending request belongs to a forced resident.
    private func combinedCeiling() -> WorkerMemoryCeiling {
        let all = Array(ceilings.values)
        guard var combined = all.max(by: { ($0.expectedMB ?? -1) < ($1.expectedMB ?? -1) }) else { return defaultCeiling }
        combined.forced = all.allSatisfy(\.forced)
        return combined
    }
    /// False ends the sampler: no request is pending, or the worker was stopped.
    private func checkMemoryCeiling() async -> Bool {
        ceilings = ceilings.filter { pending[$0.key]?.reply != nil }
        guard !ceilings.isEmpty, !exited, !stopping, !memoryStop, error == nil, child != nil else {
            if !memoryStop { endMemoryWatch() } else { memoryWatch = nil }
            return false
        }
        let footprint = WorkerMemoryCeiling.footprint(pid: pid)
        // The runaway backstop: the lowest bound among the pending requests that have one.
        if let (ceiling, limit) = ceilings.values.compactMap({ c in c.runawayLimitBytes.map { (c, $0) } }).min(by: { $0.1 < $1.1 }), footprint > limit {
            await stopForMemory(ceiling.runawayFailure(footprintBytes: footprint, limitBytes: limit))
            return false
        }
        if let failure = macMemory.check(memoryToken, ceiling: combinedCeiling(), footprintBytes: footprint) {
            await stopForMemory(failure)
            return false
        }
        return true
    }
    /// A runaway gains nothing from a cooperative exit: kill it at once, wait until it has exited, and only then fail
    /// its requests, so a caller that sees the error also sees the worker stopped (never a stopped worker that still
    /// looks loaded). The error is recorded first: the exit is deliberate, not a crash.
    private func stopForMemory(_ failure: WorkerFailure) async {
        // A stop already under way leaves the guard itself once the worker has exited.
        guard !memoryStop else { return }
        guard !exited, !stopping, error == nil, let child else {
            // Already gone or stopping for another reason: nothing of this worker's is left for the guard to wait on.
            macMemory.leave(memoryToken); return
        }
        memoryWatch?.cancel(); memoryWatch = nil; memoryCeilingStops += 1
        error = failure; memoryStop = true
        // The outcome is the memory error: no stall deadline may answer first while the kill completes.
        for p in pending.values { p.timer.cancel() }
        child.signal(SIGKILL)
        // SIGKILL cannot be ignored; a reaper starved on a loaded Mac only delays this (seen past 2 s in tests).
        await waitExit()
        failPending(failure)
        await terminate()
        macMemory.leave(memoryToken)
    }
    public func unload(grace: Double = 1) async {
        if stopping { await waitExit(); return }; stopping = true
        if !exited && error == nil { _ = try? await request(op: "unload", timeout: grace) }
        await terminate(grace: grace)
        await reader?.value
        await child?.finishReaders()
    }
    private func terminate(grace: Double = 0.2) async {
        guard let child else { return }; Task.detached { _ = try? await processIO { child.closeInput() } }
        if !exited { await waitExit(timeout: grace) }
        if !exited { child.signal(SIGTERM); await waitExit(timeout: grace) }
        if !exited { child.signal(SIGKILL) }
        await waitExit()
    }
    private func waitExit(timeout: Double? = nil) async {
        guard child != nil && !exited else { return }
        let id = UUID()
        // Cleanup often runs in a cancelled deadline task. A cancelled sleep immediately throws and would spin;
        // process-exit continuations suspend regardless of the caller's cancellation state.
        await withCheckedContinuation { reply in
            exitWaiters[id] = ExitWaiter(reply: reply)
            if let timeout {
                exitWaiters[id]?.timer = Task.detached { [weak self] in
                    do { try await Task.sleep(for: .seconds(max(0, timeout))); await self?.finishExitWait(id) } catch {}
                }
            }
        }
    }
    private func finishExitWait(_ id: UUID) {
        guard let waiter = exitWaiters.removeValue(forKey: id) else { return }
        waiter.timer?.cancel(); waiter.reply.resume()
    }
}
