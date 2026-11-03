// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import ArsenalCore

struct PredictionRecord: Codable, Sendable {
    var id: String
    var model: String
    var version: String
    var input: JSONObject
    var status: String
    var paths: [String]
    var error: String?
    var created: String
    var started: String?
    var completed: String?
    var seconds: Double?
    var precision: String
    var runner: String
    var fast: Bool
    var catalogCommit: String?
    var outputType: String?
    var outputDirectory: String?
    var operation: String?
}

extension APIRouter {
    func predictionDirectory() throws -> URL {
        let directory = jobsRoot.appendingPathComponent("replicate", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700]); return directory
    }
    func persist(_ record: PredictionRecord) throws {
        try JSONEncoder().encode(record).write(to: predictionDirectory().appendingPathComponent(record.id + ".json"), options: .atomic)
        cachePrediction(record)
    }
    func cachePrediction(_ record: PredictionRecord) {
        predictions[record.id] = record
        terminalPredictionOrder.removeAll { $0 == record.id }
        if !["starting", "processing"].contains(record.status) {
            terminalPredictionOrder.append(record.id)
            while terminalPredictionOrder.count > Self.terminalPredictionCacheLimit {
                predictions[terminalPredictionOrder.removeFirst()] = nil
            }
        }
    }
    func prediction(_ id: String) throws -> PredictionRecord? {
        // Client-controlled IDs must never become arbitrary record paths.
        guard id.utf8.count == 32, id.utf8.allSatisfy({ "0123456789abcdef".utf8.contains($0) }) else { return nil }
        if let record = predictions[id] { cachePrediction(record); return record }
        let file = try predictionDirectory().appendingPathComponent(id + ".json")
        guard let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 2 * 1024 * 1024,
            let data = try? Data(contentsOf: file), let record = try? JSONDecoder().decode(PredictionRecord.self, from: data), record.id == id
        else { return nil }
        cachePrediction(record)
        return record
    }
    func recoverPredictions() throws {
        guard !recoveredPredictions else { return }
        let directory = try predictionDirectory()
        guard let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsSubdirectoryDescendants, .skipsHiddenFiles])
        else { return }
        for case let url as URL in files where url.pathExtension == "json" {
            guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 2 * 1024 * 1024,
                let data = try? Data(contentsOf: url), var record = try? JSONDecoder().decode(PredictionRecord.self, from: data), url.lastPathComponent == record.id + ".json"
            else { continue }
            if ["starting", "processing"].contains(record.status) {
                record.status = "failed"; record.error = "Arsenal stopped before this prediction completed; it was not regenerated."; record.completed = timestamp()
                try persist(record)
                if let path = record.outputDirectory { removeTransientJob(URL(fileURLWithPath: path)) }
            }
            // Terminal history stays on disk and is loaded only when requested.
        }
        recoveredPredictions = true
    }
    func timestamp() -> String { ISO8601DateFormatter().string(from: Date()) }
    func versionID(_ model: String) -> String { SHA256.hash(data: Data(model.utf8)).map { String(format: "%02x", $0) }.joined() }
    func versionObject(_ model: String) async throws -> [String: Any] {
        if let row = await catalogRows().first(where: { $0["id"] as? String == model }), row["kind"] as? String == "sound" {
            let ops = await service.operations(id: model)
            let declaration = ops.first { $0.name == "generate_sound" } ?? ops.first { $0.name == "generate_music" }
            var schema = declaration?.inputSchema ?? ["type": "object"]
            schema["required"] = ["prompt"]
            var properties = schema["properties"]?.object ?? [:]
            properties["out_dir"] = nil
            properties["op"] = ["type": "string", "enum": .array(ops.filter { ["generate_sound", "generate_music"].contains($0.name) }.map { .string($0.name) })]
            properties["output_type"] = ["type": "string", "enum": ["url", "data_uri"], "default": "url"]
            schema["properties"] = .object(properties)
            return [
                "id": versionID(model), "created_at": "2026-10-03T00:00:00Z", "cog_version": "arsenal-v2",
                "openapi_schema": [
                    "openapi": "3.0.2", "components": ["schemas": ["Input": plain(schema), "Output": ["type": "array", "items": ["type": "string", "format": "uri"]]]]
                ]
            ]
        }
        return [
            "id": versionID(model), "created_at": "2026-10-02T00:00:00Z", "cog_version": "arsenal-v1",
            "openapi_schema": [
                "openapi": "3.0.2",
                "components": [
                    "schemas": [
                        "Input": [
                            "type": "object", "required": ["prompt"],
                            "properties": [
                                "prompt": ["type": "string"], "width": ["type": "integer", "default": 1024], "height": ["type": "integer", "default": 1024],
                                "num_outputs": ["type": "integer", "default": 1], "steps": ["type": "integer"], "seed": ["type": "integer"], "guidance": ["type": "number"],
                                "output_type": ["type": "string", "enum": ["url", "data_uri"], "default": "url"]
                            ]
                        ], "Output": ["type": "array", "items": ["type": "string", "format": "uri"]]
                    ]
                ]
            ]
        ]
    }
    func predictionObject(_ record: PredictionRecord) throws -> [String: Any] {
        let base = origin + "/providers/replicate/v1/predictions/" + record.id
        return [
            "id": record.id, "model": "local/" + record.model, "version": record.version, "input": plain(record.input), "status": record.status,
            "output": record.status == "succeeded"
                ? try record.paths.map { path in
                    if record.outputType == "data_uri" {
                        let file = try outputPath(path); return "data:" + contentType(file) + ";base64," + (try Data(contentsOf: file)).base64EncodedString()
                    }
                    return try outputURL(path)
                } : NSNull(), "error": record.error as Any? ?? NSNull(), "logs": "", "created_at": record.created, "started_at": record.started as Any? ?? NSNull(),
            "completed_at": record.completed as Any? ?? NSNull(), "metrics": record.seconds.map { ["predict_time": $0] } ?? [:], "urls": ["get": base, "cancel": base + "/cancel"],
            "data_removed": false
        ]
    }
    func replicate(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        try recoverPredictions()
        let path = String(request.path.dropFirst("/providers/replicate/v1/".count)), parts = path.split(separator: "/").map(String.init)
        if request.method == "GET", parts.first == "models" {
            let rows = await rows().filter { ["images", "sound"].contains($0["kind"] as? String ?? "") }
            func descriptor(_ row: [String: Any]) async throws -> [String: Any] {
                let id = row["id"] as? String ?? ""
                return [
                    "owner": "local", "name": id, "description": row["summary"] ?? "", "visibility": "private", "url": origin + "/providers/replicate/v1/models/local/" + id,
                    "latest_version": try await versionObject(id), "default_example": NSNull(), "cover_image_url": NSNull(), "run_count": 0
                ]
            }
            if parts.count == 1 {
                var descriptors: [[String: Any]] = []; for row in rows { descriptors.append(try await descriptor(row)) }
                try await connection.respond(["results": descriptors, "next": NSNull(), "previous": NSNull()]); return
            }
            guard parts.count >= 3, parts[1] == "local", let row = rows.first(where: { $0["id"] as? String == parts[2] }) else {
                throw HTTPFailure(404, "Unknown local image or sound model.")
            }
            if parts.count == 3 { try await connection.respond(try await descriptor(row)); return }
            if parts.count == 5, parts[3] == "versions", parts[4] == versionID(parts[2]) { try await connection.respond(try await versionObject(parts[2])); return }
            throw HTTPFailure(404, "Model version not found.")
        }
        if parts.first == "predictions", parts.count >= 2 {
            guard let record = try prediction(parts[1]) else { throw HTTPFailure(404, "Prediction not found.") }
            if request.method == "GET", parts.count == 2 { try await connection.respond(try predictionObject(record)); return }
            if request.method == "POST", parts.count == 3, parts[2] == "cancel" {
                var record = record
                if ["starting", "processing"].contains(record.status) {
                    record.status = "canceled"; record.completed = timestamp(); try persist(record)
                    predictionTasks[record.id]?.cancel()
                }
                try await connection.respond(try predictionObject(record)); return
            }
            throw HTTPFailure(405, "Use GET prediction or POST cancel.")
        }
        let official = parts.count == 4 && parts[0] == "models" && parts[1] == "local" && parts[3] == "predictions"
        guard request.method == "POST", path == "predictions" || official else { throw HTTPFailure(404, "Replicate endpoint not supported.") }
        let body = try object(request)
        try reject(body, ["webhook", "webhook_events_filter", "stream"])
        let model: String
        if official {
            model = parts[2]
        } else {
            let version = try required(body, "version")
            guard
                let found = await installedModels().first(where: {
                    ["images", "sound"].contains($0["kind"] as? String ?? "") && ($0["id"] as? String == version || versionID($0["id"] as? String ?? "") == version)
                })?["id"] as? String
            else { throw HTTPFailure(404, "Unknown installed image or sound version; use the model id or its local version hash.") }
            model = found
        }
        let row = try await resolve(model, kind: nil)
        let sound = row["kind"] as? String == "sound"
        guard sound || row["kind"] as? String == "images" else { throw HTTPFailure(422, "Predictions support image and sound models.") }
        guard let input = body["input"]?.object else { throw HTTPFailure(422, "input must be an object.") }
        let outputType = input["output_type"]?.string ?? "url"
        guard ["url", "data_uri"].contains(outputType) else { throw HTTPFailure(422, "output_type must be url or data_uri.") }
        let allowed: Set<String> =
            sound
            ? ["prompt", "duration", "seed", "steps", "negative_prompt", "guidance", "op", "output_type"]
            : ["prompt", "width", "height", "num_outputs", "n", "steps", "seed", "guidance", "output_type"]
        let unknown = Set(input.keys).subtracting(allowed)
        guard unknown.isEmpty else { throw HTTPFailure(422, "Unsupported prediction input: " + unknown.sorted().joined(separator: ", ")) }
        guard request.headers["cancel-after"] == nil else { throw HTTPFailure(400, "Cancel-After is unsupported; use POST prediction cancel.") }
        var waitSeconds = 0.0
        if let prefer = request.headers["prefer"] {
            guard prefer == "wait" || prefer.hasPrefix("wait=") else { throw HTTPFailure(422, "Unsupported Prefer header.") }
            waitSeconds = prefer == "wait" ? 60 : Double(prefer.dropFirst(5)) ?? 0
            guard waitSeconds > 0, waitSeconds <= 60 else { throw HTTPFailure(422, "Prefer wait must be 1 through 60 seconds.") }
        }
        guard predictionTasks.count < Self.activePredictionLimit else { throw HTTPFailure(429, "At most 32 predictions may run at once; wait for a prediction to finish.") }
        let operation = sound ? input["op"]?.string ?? "generate_sound" : "generate"
        guard !sound || ["generate_sound", "generate_music"].contains(operation) else { throw HTTPFailure(422, "Sound op must be generate_sound or generate_music.") }
        var params = sound ? try replicateSoundParams(input) : try replicateImageParams(input, dir: jobsRoot)
        if sound { guard await service.operations(id: model).contains(where: { $0.name == operation }) else { throw HTTPFailure(422, "Model does not offer \(operation).") } }
        if let refusal = await service.runRefusal(id: model, op: operation) { throw refusal }
        let id = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        var record = PredictionRecord(
            id: id, model: model, version: versionID(model), input: input, status: "starting", paths: [], created: timestamp(),
            precision: row["selectedPrecision"] as? String ?? "bf16", runner: row["runner"] as? String ?? "", fast: row["fast"] as? Bool ?? true,
            catalogCommit: await service.catalogStatus().pinnedCommit, outputType: outputType)
        // Recheck after catalogStatus's actor suspension so concurrent creators
        // cannot over-admit before installing their task in the dictionary.
        guard predictionTasks.count < Self.activePredictionLimit else { throw HTTPFailure(429, "At most 32 predictions may run at once; wait for a prediction to finish.") }
        let directory = try newJob()
        var admitted = false
        defer { if !admitted { removeTransientJob(directory) } }
        params["out_dir"] = .string(directory.path)
        record.outputDirectory = directory.path
        record.operation = operation
        try persist(record)
        predictionTasks[id] = Task { await self.executePrediction(id: id, params: params) }
        admitted = true
        if waitSeconds > 0 {
            let deadline = Date().addingTimeInterval(waitSeconds)
            while Date() < deadline, predictions[id].map({ ["starting", "processing"].contains($0.status) }) == true { try await Task.sleep(for: .milliseconds(20)) }
        }
        guard let current = try prediction(id) else { throw HTTPFailure(500, "Prediction was lost.") }
        try await connection.respond(try predictionObject(current), status: 201, headers: ["Location": origin + "/providers/replicate/v1/predictions/" + id])
    }
    func executePrediction(id: String, params: JSONObject) async {
        var keepOutput = false
        defer {
            predictionTasks[id] = nil
            if !keepOutput, let path = params["out_dir"]?.string { removeTransientJob(URL(fileURLWithPath: path)) }
        }
        do {
            guard var record = predictions[id], record.status == "starting" else { return }
            record.status = "processing"; record.started = timestamp(); try persist(record)
            let result = try await run(record.model, op: record.operation ?? "generate", params: params)
            try Task.checkCancellation()
            guard predictions[id]?.status == "processing" else { return }
            if let path = result["path"]?.string {
                _ = try outputPath(path); record.paths = [path]
            } else {
                record.paths = try (result["images"]?.array ?? result["files"]?.array ?? []).map { image in
                    guard let path = image["path"]?.string else { throw HTTPFailure(502, "Prediction output has no path.") }; _ = try outputPath(path); return path
                }
            }
            guard !record.paths.isEmpty else { throw HTTPFailure(502, "Runner returned no prediction output.") }
            record.status = "succeeded"; record.completed = timestamp(); record.seconds = result["seconds"]?.number; try persist(record)
            keepOutput = true
        } catch {
            if var record = predictions[id], ["starting", "processing"].contains(record.status) {
                record.status = error is CancellationError ? "canceled" : "failed"; record.completed = timestamp()
                record.error = (error as? WorkerFailure)?.message ?? (error as? HTTPFailure)?.message ?? "Prediction failed."; try? persist(record)
            }
        }
    }
}
