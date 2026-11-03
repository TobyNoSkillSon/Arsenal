// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// A start remains owned before and after await; unload can never miss a late registration.
public actor Supervisor {
    private var workers: [String: WorkerProcess] = [:]
    private struct Starting { let token: UUID; let worker: WorkerProcess; let task: Task<WorkerProcess, any Error> }
    private var starts: [String: Starting] = [:]
    private var unloading: Set<String> = []
    private var closing = false
    private var observedSuccesses: [String: Int] = [:]
    private var crashes: [String: Int] = [:]
    public let crashLimit: Int
    public init(crashLimit: Int = 3) { self.crashLimit = max(1, crashLimit) }
    public func worker(
        id: String, launch: WorkerLaunch, manual: Bool = false, status: (@Sendable (WorkerMemory) async -> Void)? = nil,
        statusDetails: (@Sendable (JSONObject) async -> Void)? = nil
    ) async throws -> WorkerProcess {
        guard !closing else { throw WorkerFailure(code: "shutting_down", message: "Worker supervisor is shutting down.") }
        guard !unloading.contains(id) else { throw WorkerFailure(code: "busy", message: "Runner is still unloading; try again after it stops.") }
        if manual { crashes[id] = nil }
        if let existing = workers[id], !(await existing.exited) { return existing }
        if let existing = workers.removeValue(forKey: id) {
            if await existing.successfulRequests > observedSuccesses[id, default: 0] { crashes[id] = nil }
            await existing.unload()
            if await existing.unexpectedExit, !manual { crashes[id, default: 0] += 1 }
            observedSuccesses[id] = nil
        }
        guard crashes[id, default: 0] < crashLimit else { throw WorkerFailure(code: "quarantined", message: "Runner for \(id) repeatedly crashed; reset it before retrying.") }
        if let start = starts[id] {
            let ready = try await start.task.value
            guard !closing, !unloading.contains(id), starts[id]?.token == start.token || workers[id].map({ $0 === ready }) == true else {
                await ready.unload(grace: 0.2); throw CancellationError()
            }
            return ready
        }
        let worker = WorkerProcess(), token = UUID()
        let task = Task<WorkerProcess, any Error> {
            do { try Task.checkCancellation(); try await worker.start(launch, status: status, statusDetails: statusDetails); return worker } catch {
                await worker.unload(grace: 0.2); throw error
            }
        }
        starts[id] = Starting(token: token, worker: worker, task: task)
        do {
            let ready = try await task.value
            guard !closing, !unloading.contains(id), starts[id]?.token == token else { await ready.unload(grace: 0.2); throw CancellationError() }
            workers[id] = ready; observedSuccesses[id] = 0; starts[id] = nil; return ready
        } catch {
            if starts[id]?.token == token { starts[id] = nil }
            if !(error is CancellationError), !closing, !unloading.contains(id) { crashes[id, default: 0] += 1 }
            throw error
        }
    }
    public func unload(id: String) async {
        unloading.insert(id); defer { unloading.remove(id) }
        let start = starts.removeValue(forKey: id), resident = workers.removeValue(forKey: id)
        if let start { start.task.cancel(); _ = try? await start.task.value; await start.worker.unload(grace: 0.2) }
        if let resident { await resident.unload() }
        observedSuccesses[id] = nil
    }
    public func reset(id: String) async { await unload(id: id); crashes[id] = nil }
    public func shutdown() async {
        closing = true
        while !workers.isEmpty || !starts.isEmpty { for id in Array(Set(workers.keys).union(starts.keys)) { await unload(id: id) } }
    }
    public func hasExited(id: String) async -> Bool { guard let worker = workers[id] else { return true }; return await worker.exited }
    public func workerPIDs() -> Set<Int32> { WorkerPIDRegistry.snapshot() }
    public func residentIDs() -> [String] { workers.keys.sorted() }
    public func request(
        id: String, op: String, params: JSONObject = [:], timeout: Double = 120, firstResponseTimeout: Double? = nil, progress: ProgressTransform? = nil, sink: RunSink? = nil,
        memoryCeiling: WorkerMemoryCeiling? = nil
    ) async throws
        -> JSONObject
    {
        guard !closing else { throw WorkerFailure(code: "shutting_down", message: "Worker supervisor is shutting down.") }
        guard let worker = workers[id] else { throw WorkerFailure(code: "not_running", message: "Runner is not loaded.") }
        let result = try await worker.request(
            op: op, params: params, timeout: timeout, firstResponseTimeout: firstResponseTimeout, progress: progress, sink: sink, memoryCeiling: memoryCeiling)
        crashes[id] = nil; observedSuccesses[id] = await worker.successfulRequests; return result
    }
}
