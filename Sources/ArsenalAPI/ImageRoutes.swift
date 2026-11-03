// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func imageParams(_ body: JSONObject, dir: URL) throws -> JSONObject {
        let prompt = try required(body, "prompt")
        guard body["size"] == nil || body["size"]?.string != nil else { throw HTTPFailure(422, "size must be a WxH string.") }
        var dimensions: JSONObject = [:]
        if let size = body["size"]?.string {
            let parts = size.split(separator: "x")
            guard parts.count == 2, let width = Int(parts[0]), let height = Int(parts[1]), (64...2048).contains(width), (64...2048).contains(height), width % 16 == 0,
                height % 16 == 0
            else { throw HTTPFailure(422, "size must be WxH, multiples of 16 between 64 and 2048.") }
            dimensions = ["width": .number(Double(width)), "height": .number(Double(height))]
        }
        let count = try jsonInteger(body["n"], name: "n", range: 1...4, default: 1)
        try reject(body, ["quality", "style", "stream", "partial_images", "background", "moderation", "input_fidelity"])
        if let format = body["output_format"]?.string, format != "png" { throw HTTPFailure(400, "Only PNG image output is supported.") }
        if let steps = body["steps"] { _ = try jsonInteger(steps, name: "steps", range: 1...100) }
        if let seed = body["seed"] { _ = try jsonInteger(seed, name: "seed", range: 0...4_294_967_295) }
        if let guidance = body["guidance"] {
            guard let number = guidance.number, number.isFinite, (0...20).contains(number) else { throw HTTPFailure(422, "guidance must be a finite number from 0 to 20.") }
        }
        var params: JSONObject = ["prompt": .string(prompt), "n": .number(Double(count)), "out_dir": .string(dir.path)]
        params.merge(dimensions) { _, new in new }
        for key in ["steps", "seed", "guidance"] { params[key] = body[key] }
        return params
    }
    func replicateImageParams(_ input: JSONObject, dir: URL) throws -> JSONObject {
        var params = try imageParams(input.merging(["n": input["num_outputs"] ?? input["n"] ?? 1]) { _, new in new }, dir: dir)
        for key in ["width", "height"] where input[key] != nil {
            let dimension = try jsonInteger(input[key], name: key, range: 64...2048)
            guard dimension % 16 == 0 else { throw HTTPFailure(422, "\(key) must be a multiple of 16.") }
            params[key] = .number(Double(dimension))
        }
        return params
    }

    func imageOutput(_ result: JSONObject, format: String) throws -> [[String: Any]] {
        // Older image workers answer `images`; schema-3 folders (FLUX 3 Klein) answer `files`, as MCP already reads.
        guard let images = result["images"]?.array ?? result["files"]?.array, !images.isEmpty else { throw HTTPFailure(500, "Runner returned no images.") }
        return try images.map { image in
            guard let path = image["path"]?.string else { throw HTTPFailure(500, "Runner image has no output path.") }
            let file = try outputPath(path)
            if format == "url" { return ["url": try outputURL(path)] }
            guard let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 64 * 1024 * 1024 else {
                throw HTTPFailure(500, "Runner PNG exceeds the 64 MiB limit.")
            }
            let data = try Data(contentsOf: file)
            guard data.prefix(8) == Data([137, 80, 78, 71, 13, 10, 26, 10]) else { throw HTTPFailure(500, "Runner did not return PNG data.") }
            return ["b64_json": data.base64EncodedString()]
        }
    }
    func images(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let body = try object(request), id = try required(body, "model")
        _ = try await resolve(id, kind: "images", operation: "generate")
        let format = body["response_format"]?.string ?? "b64_json"
        guard ["b64_json", "url"].contains(format) else { throw HTTPFailure(422, "response_format must be b64_json or url.") }
        var params = try imageParams(body, dir: jobsRoot)
        let directory = try newJob()
        var keepOutput = false
        defer { if !keepOutput { removeTransientJob(directory) } }
        params["out_dir"] = .string(directory.path)
        let result = try await run(id, op: "generate", params: params)
        try await connection.respond(["created": Int(Date().timeIntervalSince1970), "data": try imageOutput(result, format: format)])
        keepOutput = format == "url"
    }
}
