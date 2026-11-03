// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    /// Ordered members (OrderedJSONMembers) keep the request body's own bytes, so their key order reaches the worker
    /// exactly as sent: an object-typed member given as an object is replaced by its text; any other member holding an
    /// object or array (a decision state, a list of states) keeps its value and its text travels in the transport
    /// member, so a string stays the text it is.
    func orderedMembers(_ input: JSONObject, body: Data, schema: JSONObject) -> JSONObject {
        let names = OrderedJSONMembers.names(schema)
        guard !names.isEmpty, let fields = try? RawJSON(body).fields() else { return input }
        var input = input, carried: JSONObject = [:]
        for field in fields where names.contains(field.key) {
            let text = String(decoding: field.value.data, as: UTF8.self)
            if OrderedJSONMembers.takesText(field.key, schema: schema) {
                if input[field.key]?.object != nil { input[field.key] = .string(text) }
            } else if input[field.key]?.object != nil || input[field.key]?.array != nil {
                carried[field.key] = .string(text)
            }
        }
        if !carried.isEmpty {
            input[OrderedJSONMembers.transport] = .object((input[OrderedJSONMembers.transport]?.object ?? [:]).merging(carried) { $1 })
        }
        return input
    }
    /// Only verified declarations supplied by ModelsService decide the output
    /// mode. The service validates input before it invokes the runner/sink.
    func universal(_ request: HTTPRequest, _ connection: HTTPConnection, id: String, op: String) async throws {
        guard let declaration = await service.operations(id: id).first(where: { $0.name == op }) else { throw HTTPFailure(404, "Model \(id) does not declare operation \(op).") }
        let input = orderedMembers(try object(request), body: request.body, schema: declaration.inputSchema)
        let row = try await resolve(id, kind: nil, operation: op)
        if request.headers["prefer"]?.lowercased().split(separator: ",").contains(where: { $0.trimmingCharacters(in: .whitespaces) == "respond-async" }) == true {
            try await startOperationJob(request, connection, id: id, declaration: declaration, input: input); return
        }
        if row["kind"] as? String == "decisions", ["decide", "judge"].contains(op) {
            try OperationSchemaValidator.validate(.object(try OrderedJSONMembers.split(input, schema: declaration.inputSchema).request), schema: .object(declaration.inputSchema))
            let raw = try RawJSON(request.body)
            // The URL supplies a missing model; existing member bytes and
            // question/option order are preserved, including NFC/NFD labels.
            var replacements: [String: Data] = raw["model"] == nil ? ["model": jsonData(id)] : [:]
            // An object-typed ordered member sent as its JSON text (validated above as that object) reaches the worker
            // as the object it stands for, as on every other route.
            for name in OrderedJSONMembers.names(declaration.inputSchema) where OrderedJSONMembers.takesText(name, schema: declaration.inputSchema) {
                if let member = raw[name], member.data.first == UInt8(ascii: "\""), let text = try? JSONDecoder().decode(String.self, from: member.data) {
                    replacements[name] = Data(text.utf8)
                }
            }
            let body = replacements.isEmpty ? request.body : try raw.replacing(replacements)
            let reply = try await decisionCall(model: id, op: op, body: body)
            try await connection.respondBytes(reply.body, type: "application/json", status: reply.status, headers: reply.bits.map { ["x-verdict-bits": $0] } ?? [:])
            return
        }
        guard input["out_dir"] == nil else { throw HTTPFailure(422, "Output directories are managed by Arsenal.") }
        var params = input
        let accept = OperationAccept(request.headers["accept"] ?? "").preferred
        if declaration.schemaVersion == 3, accept == "text/event-stream", declaration.stream {
            let directory = ["audio", "image", "file"].contains(declaration.output) ? try newJob() : nil
            if let directory { params["out_dir"] = .string(directory.path) }
            var retained = false
            defer { if let directory, !retained { removeTransientJob(directory) } }
            try await connection.beginStream()
            do {
                var result = try await run(
                    id, op: op, params: params,
                    sink: { frame in
                        if case .json(let value) = frame { try await connection.event(plain(value), name: value["event"]?.string ?? "progress") }
                    })
                if let directory { result = try linkedFileResult(result, directory: directory); retained = true }
                try await connection.event(plain(result), name: "result")
            } catch {
                let failure = serviceHTTPFailure(error)
                try await connection.event(
                    [
                        "error": failure.message, "code": failure.code ?? "execution", "status": failure.status,
                        "details": plain(failure.details)
                    ], name: "error")
            }
            try await connection.finishStream(); return
        }
        if declaration.schemaVersion == 3, accept == "application/json", ["audio", "image", "file"].contains(declaration.output) {
            let directory = try newJob(); params["out_dir"] = .string(directory.path)
            do {
                let result = try await run(id, op: op, params: params)
                try await connection.respond(plain(try linkedFileResult(result, directory: directory)))
            } catch { removeTransientJob(directory); throw error }
            return
        }
        if declaration.schemaVersion == 3, accept == "application/json", declaration.output == "text-stream" {
            if declaration.inputSchema["properties"]?["stream"] != nil { params["stream"] = false }
            try await connection.respond(plain(try await run(id, op: op, params: params))); return
        }
        switch declaration.output {
        case "json":
            try await connection.respond(plain(try await run(id, op: op, params: params)))
        case "text-stream":
            if declaration.stream && input["stream"]?.bool != false {
                if declaration.inputSchema["properties"]?["stream"] != nil { params["stream"] = true }
                let stream = UniversalStream(connection: connection, mode: .text)
                let result = try await run(id, op: op, params: params, sink: { try await stream.receive($0) })
                try await stream.complete(result)
            } else {
                try await connection.respond(plain(try await run(id, op: op, params: params)))
            }
        case "audio":
            if declaration.stream && input["stream"]?.bool == true {
                let stream = UniversalStream(connection: connection, mode: .audio)
                let result = try await run(id, op: op, params: params, sink: { try await stream.receive($0) })
                try await stream.complete(result)
            } else {
                let job = try newJob()
                defer { removeTransientJob(job) }
                params["out_dir"] = .string(job.path)
                let result = try await run(id, op: op, params: params)
                try await sendDeclaredFile(result, output: "audio", connection: connection)
            }
        case "image", "file":
            let job = try newJob()
            defer { removeTransientJob(job) }
            params["out_dir"] = .string(job.path)
            try await sendDeclaredFile(try await run(id, op: op, params: params), output: declaration.output, connection: connection)
        default: throw HTTPFailure(500, "Model declares an unsupported output type.")
        }
    }
    func linkedFileResult(_ result: JSONObject, directory: URL) throws -> JSONObject {
        var payload = result
        let values = result["files"]?.array ?? result["images"]?.array ?? result["path"].map { _ in [JSONValue.object(result)] } ?? result["preview"].map { [$0] } ?? []
        guard !values.isEmpty else { throw HTTPFailure(502, "Runner returned no output file.") }
        var total = 0
        payload["files"] = .array(
            try values.map { value in
                guard var file = value.object, let path = file["path"]?.string else { throw HTTPFailure(502, "Output file needs path.") }
                let url = try OperationArtifacts.confined(path, roots: [directory], code: "invalid_output")
                let bytes = try FileDigest.bytes(url)
                do { try OperationArtifacts.admitArtifact(bytes, total: &total) } catch let failure as WorkerFailure { throw HTTPFailure(502, failure.message) }
                file["url"] = .string(try outputURL(path)); file["mime"] = file["mime"] ?? .string(contentType(url))
                file["bytes"] = .number(Double(bytes)); file["sha256"] = .string(try FileDigest.sha256(url))
                return .object(file)
            })
        return payload
    }
    func sendDeclaredFile(_ result: JSONObject, output: String, connection: HTTPConnection) async throws {
        let paths: [String]
        if let path = result["path"]?.string {
            paths = [path]
        } else if let files = result["images"]?.array ?? result["files"]?.array {
            paths = try files.map { file in
                guard let path = file["path"]?.string else { throw HTTPFailure(502, "Runner returned a file without its path.") }; return path
            }
        } else {
            paths = []
        }
        guard !paths.isEmpty else { throw HTTPFailure(502, "Runner returned no output file.") }
        var outputFiles: [(Data, String)] = [], total = 0
        for path in paths {
            let url = try outputPath(path)
            guard let bytes = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize, bytes <= PCM.maxBytes - total else {
                throw HTTPFailure(502, "Runner output exceeds 64 MiB.")
            }
            total += bytes; outputFiles.append((try Data(contentsOf: url), contentType(url)))
        }
        if outputFiles.count == 1 { try await connection.respondBytes(outputFiles[0].0, type: outputFiles[0].1); return }
        let boundary = "arsenal-" + UUID().uuidString
        var multipart = Data()
        for (index, file) in outputFiles.enumerated() {
            multipart += Data("--\(boundary)\r\nContent-Type: \(file.1)\r\nContent-Disposition: attachment; name=\"output-\(index)\"\r\n\r\n".utf8)
            multipart += file.0; multipart += Data("\r\n".utf8)
        }
        multipart += Data("--\(boundary)--\r\n".utf8)
        try await connection.respondBytes(multipart, type: "multipart/mixed; boundary=" + boundary)
    }

    func contentType(_ url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "png": "image/png"
        case "jpg", "jpeg": "image/jpeg"
        case "wav": "audio/wav"
        case "pcm": "audio/pcm"
        case "flac": "audio/flac"
        case "mp4", "m4v": "video/mp4"
        case "mov": "video/quicktime"
        case "glb": "model/gltf-binary"
        case "ply": "application/octet-stream"
        case "json": "application/json"
        case "txt": "text/plain; charset=utf-8"
        default: "application/octet-stream"
        }
    }
}

private actor UniversalStream {
    enum Mode { case text, audio }
    let connection: HTTPConnection
    let mode: Mode
    var opened = false
    var received = false
    init(connection: HTTPConnection, mode: Mode) { self.connection = connection; self.mode = mode }
    func receive(_ frame: WireFrame) async throws {
        switch (mode, frame) {
        case (.text, .json(let value)):
            guard value["event"]?.string != "progress" else { return }
            if !opened { try await connection.beginStream(); opened = true }
            try await connection.event(plain(value), name: value["event"]?.string ?? "event")
            received = true
        case (.audio, .json(let value)) where value["event"]?.string == "audio_begin":
            guard value["encoding"]?.string == "pcm_s16le" else { throw HTTPFailure(502, "Streaming audio must be pcm_s16le.") }
            let rate = try jsonInteger(value["sample_rate"], name: "Sample rate", range: 8000...192000, status: 502)
            let channels = try jsonInteger(value["channels"], name: "Channels", range: 1...8, status: 502)
            guard !opened else { throw HTTPFailure(502, "Runner restarted its audio stream.") }
            try await connection.beginStream(
                type: "audio/pcm", headers: ["X-Arsenal-Sample-Rate": String(rate), "X-Arsenal-Channels": String(channels), "X-Arsenal-Encoding": "pcm_s16le"])
            opened = true
        case (.audio, .binary(_, let bytes)):
            guard opened else { throw HTTPFailure(502, "Runner sent audio before its format declaration.") }
            try await connection.chunk(bytes); received = received || !bytes.isEmpty
        default: break
        }
    }
    func complete(_ result: JSONObject) async throws {
        if mode == .text {
            if !opened { try await connection.beginStream(); opened = true }
            try await connection.event(plain(result), name: "result")
        } else if !opened || !received {
            throw HTTPFailure(502, "Runner returned no streamed audio; use stream:false for file-only audio.")
        }
        try await connection.finishStream()
    }
}

/// A generic transport is opt-in: a wildcard or a competing type at equal weight keeps the legacy response.
struct OperationAccept {
    let preferred: String?
    init(_ header: String) {
        let ranges: [(type: String, quality: Double)] = header.lowercased().split(separator: ",").compactMap { range in
            let parts = range.split(separator: ";").map { $0.trimmingCharacters(in: .whitespaces) }
            guard let type = parts.first, type.split(separator: "/").count == 2 else { return nil }
            var quality = 1.0
            for parameter in parts.dropFirst() {
                let pair = parameter.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
                if pair.first == "q" {
                    guard pair.count == 2, let weight = Double(pair[1]), weight.isFinite, (0...1).contains(weight) else { return nil }
                    quality = weight
                }
            }
            return (type, quality)
        }
        let best = ranges.map(\.quality).max() ?? 0
        let winners = Set(ranges.filter { $0.quality == best && best > 0 }.map(\.type))
        preferred = winners.count == 1 ? winners.first : nil
    }
}
