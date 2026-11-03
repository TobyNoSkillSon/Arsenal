// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

public struct BenchmarkJob: Codable, Sendable {
    public var job: String
    public var request: BenchmarkRequest
    public var state: String
    public var result: ResultsFile?
    public var error: String?
    public init(job: String, request: BenchmarkRequest, state: String = "queued") {
        self.job = job; self.request = request; self.state = state
    }
}

public actor BenchmarkController {
    public typealias Execution = @Sendable (BenchmarkRequest) async throws -> ResultsFile
    private let execute: Execution
    private var jobs: [String: BenchmarkJob] = [:]
    private var running: String?
    public init(execute: @escaping Execution) { self.execute = execute }
    public init(service: any ModelsService, configuration: BenchmarkConfiguration) {
        let harness = BenchmarkHarness(service: service, configuration: configuration)
        execute = { request in try await harness.run(request) }
    }
    @discardableResult public func start(_ request: BenchmarkRequest) throws -> BenchmarkJob {
        try request.validate()
        guard running == nil else { throw BenchmarkError.invalid("A benchmark is already running; wait for its job result.") }
        let id = UUID().uuidString.lowercased(), job = BenchmarkJob(job: id, request: request)
        jobs[id] = job; running = id
        Task { await perform(id: id, request: request) }
        return job
    }
    public func status(_ job: String) -> BenchmarkJob? { jobs[job] }
    public func list() -> [BenchmarkJob] { jobs.values.sorted { $0.job < $1.job } }
    private func perform(id: String, request: BenchmarkRequest) async {
        jobs[id]?.state = "running"
        do {
            let result = try await execute(request)
            jobs[id]?.result = result
            jobs[id]?.state = request.quickCheck ? "quick_check" : (result.conditions.contaminated ? "contaminated" : "succeeded")
        } catch { jobs[id]?.state = "failed"; jobs[id]?.error = error.localizedDescription }
        running = nil
    }
}
