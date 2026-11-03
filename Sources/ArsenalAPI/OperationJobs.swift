// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

struct OperationJob: Codable, Sendable {
    let id: String
    let model: String
    let operation: String
    let createdAt: Date
    var status = "starting"
    var output: JSONObject?
    var error: String?
    /// The failure's structured fields, as a direct request's error body carries them (older records lack them).
    var code: String?
    var details: JSONObject?
    var httpStatus: Int?
    var progress: JSONObject = [:]
    var terminal: Bool { ["succeeded", "failed", "canceled"].contains(status) }
}

extension APIRouter {
    func operationJobDirectory(_ id: String) -> URL { jobsRoot.appendingPathComponent("operations/" + id) }
    func saveOperationJob(_ job: OperationJob) throws {
        let folder = operationJobDirectory(job.id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try JSONEncoder().encode(job).write(to: folder.appendingPathComponent("record.json"), options: .atomic)
    }
    func operationJobPayload(_ job: OperationJob) -> JSONObject {
        var value: JSONObject = [
            "id": .string(job.id), "model": .string(job.model), "operation": .string(job.operation), "status": .string(job.status),
            "urls": ["get": .string(origin + "/arsenal/v1/jobs/" + job.id), "cancel": .string(origin + "/arsenal/v1/jobs/" + job.id + "/cancel")],
            "progress": .object(job.progress)
        ]
        var output = job.output
        if let files = output?["files"]?.array {
            output?["files"] = .array(
                files.map { value in
                    guard var file = value.object, let path = file["path"]?.string, let url = try? outputURL(path) else { return value }
                    file["url"] = .string(url); return .object(file)
                })
        }
        value["output"] = output.map(JSONValue.object) ?? .null
        value["error"] = job.error.map(JSONValue.string) ?? .null
        // The direct request's error object, so a job's insufficient_memory keeps its figures and force hint.
        if job.status == "failed", let message = job.error {
            var failure: JSONObject = ["error": .string(message), "status": .number(Double(job.httpStatus ?? 500))]
            if let code = job.code { failure["code"] = .string(code) }
            if let details = job.details, !details.isEmpty { failure["details"] = .object(details) }
            value["failure"] = .object(failure)
        } else {
            value["failure"] = .null
        }
        return value
    }
    func startOperationJob(_ request: HTTPRequest, _ connection: HTTPConnection, id: String, declaration: ModelOperation, input: JSONObject) async throws {
        guard operationJobs.values.filter({ !$0.terminal }).count < 32 else { throw HTTPFailure(429, "Too many active operation jobs.") }
        guard input["out_dir"] == nil else { throw HTTPFailure(422, "Arsenal manages output directories.") }
        try OperationSchemaValidator.validate(.object(try OrderedJSONMembers.split(input, schema: declaration.inputSchema).request), schema: .object(declaration.inputSchema))
        let jobID = UUID().uuidString.lowercased()
        let job = OperationJob(id: jobID, model: id, operation: declaration.name, createdAt: Date())
        try saveOperationJob(job); operationJobs[jobID] = job
        // Unstructured work is owned by the server, independent of the short creation request's disconnect.
        let task = Task { await self.executeOperationJob(jobID, declaration: declaration, input: input) }
        operationJobTasks[jobID] = task
        try await connection.respond(
            plain(operationJobPayload(job)), status: 202, headers: ["Preference-Applied": "respond-async", "Location": origin + "/arsenal/v1/jobs/" + jobID])
    }
    func executeOperationJob(_ jobID: String, declaration: ModelOperation, input: JSONObject) async {
        guard var job = operationJobs[jobID], !job.terminal else { return }
        job.status = "processing"; operationJobs[jobID] = job; try? saveOperationJob(job)
        let directory = operationJobDirectory(jobID).appendingPathComponent("output")
        do {
            var params = input
            let fileOutput = ["audio", "image", "file"].contains(declaration.output)
            if fileOutput {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
                params["out_dir"] = .string(directory.path)
            }
            if declaration.inputSchema["properties"]?["stream"] != nil { params["stream"] = false }
            var result = try await run(
                job.model, op: declaration.name, params: params,
                sink: { [weak self] frame in
                    if case .json(let value) = frame { await self?.operationJobProgress(jobID, value: value) }
                })
            try Task.checkCancellation()
            if fileOutput { result = try linkedFileResult(result, directory: directory) }
            guard var current = operationJobs[jobID], !current.terminal else { return }
            current.output = result; current.status = "succeeded"; operationJobs[jobID] = current; try saveOperationJob(current)
        } catch {
            if var current = operationJobs[jobID], !current.terminal {
                current.status = error is CancellationError ? "canceled" : "failed"
                let failure = serviceHTTPFailure(error, modelID: current.model)
                current.error = failure.message
                if current.status == "failed" {
                    current.code = failure.code; current.httpStatus = failure.status; current.details = failure.details.isEmpty ? nil : failure.details
                }
                operationJobs[jobID] = current; try? saveOperationJob(current)
            }
        }
        operationJobTasks[jobID] = nil
        // Bound the in-memory cache. Durable records and outputs remain on disk.
        if operationJobs.count > 128 {
            for old in operationJobs.values.filter(\.terminal).sorted(by: { $0.createdAt < $1.createdAt }).prefix(operationJobs.count - 128) { operationJobs[old.id] = nil }
        }
    }
    func operationJobProgress(_ id: String, value: JSONObject) {
        guard var job = operationJobs[id], !job.terminal else { return }
        job.progress = value.filter { ["phase", "progress", "step", "total", "label"].contains($0.key) }; operationJobs[id] = job
    }
    func operationJobRoute(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let parts = request.path.dropFirst("/arsenal/v1/jobs/".count).split(separator: "/").map(String.init)
        guard let id = parts.first, UUID(uuidString: id) != nil, parts.count <= 2 else { throw HTTPFailure(404, "Unknown operation job.") }
        var job: OperationJob
        if let live = operationJobs[id] {
            job = live
        } else {
            guard let data = try? Data(contentsOf: operationJobDirectory(id).appendingPathComponent("record.json")),
                var restored = try? JSONDecoder().decode(OperationJob.self, from: data)
            else { throw HTTPFailure(404, "Unknown operation job.") }
            if !restored.terminal {
                restored.status = "failed"; restored.error = "Arsenal restarted during the operation; it was not regenerated."
                restored.code = "interrupted"; restored.httpStatus = 503; try saveOperationJob(restored)
            }
            operationJobs[id] = restored; job = restored
        }
        if parts.count == 2 {
            guard request.method == "POST", parts[1] == "cancel" else { throw HTTPFailure(405, "Use POST on the cancel URL.") }
            if !job.terminal {
                job.status = "canceled"; operationJobs[id] = job; try saveOperationJob(job)
                operationJobTasks.removeValue(forKey: id)?.cancel()
            }
        } else if request.method != "GET" {
            throw HTTPFailure(405, "Use GET.")
        }
        try await connection.respond(plain(operationJobPayload(job)))
    }
}
