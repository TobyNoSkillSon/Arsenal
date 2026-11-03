// SPDX-License-Identifier: MIT
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// One artifact policy shared by the service, checker and transport adapters.
public enum OperationArtifacts {
    /// Inline payloads (base64 or JSON bodies, direct byte responses, resource reads) are held in memory: 64 MiB.
    public static let maxBytes = 64 * 1024 * 1024
    /// Files a job writes to its own directory and returns as URLs are served from disk, so they do not share the inline
    /// cap. This per-job bound keeps one job's artifacts from filling the disk; it is documented in docs/API.md.
    public static let maxArtifactBytes = 4 * 1024 * 1024 * 1024
    /// Adds one job artifact to the running total, refusing it past the per-job artifact limit.
    public static func admitArtifact(_ bytes: Int64, total: inout Int) throws {
        guard bytes >= 0, bytes <= Int64(maxArtifactBytes - total) else {
            throw WorkerFailure(code: "invalid_output", message: "Job output files exceed the 4 GiB per-job artifact limit.")
        }
        total += Int(bytes)
    }
    public static func input(_ value: JSONValue, schema: JSONObject, support: URL) throws {
        let root = JSONValue.object(schema)
        var budget = 20_000
        func expanded(_ value: JSONValue, depth: Int = 0) throws -> JSONValue {
            budget -= 1
            guard budget >= 0, depth < 24 else { throw WorkerFailure(code: "op_schema", message: "File schema evaluation limit exceeded.") }
            if var object = value.object {
                if let ref = object.removeValue(forKey: "$ref")?.string {
                    var target = root
                    for part in ref.dropFirst(2).split(separator: "/") {
                        target = target.object?[String(part).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")] ?? .null
                    }
                    return ["allOf": .array([try expanded(target, depth: depth + 1), try expanded(.object(object), depth: depth + 1)])]
                }
                for (key, child) in object { object[key] = try expanded(child, depth: depth + 1) }; return .object(object)
            }
            if let array = value.array { return .array(try array.map { try expanded($0, depth: depth + 1) }) }; return value
        }
        // The budget bounds the schema, not the request: each distinct fragment is expanded once and reused for every value
        // it is tested against (an array of 1,536 video frames tests the same two branches 1,536 times).
        var expansions: [String: JSONValue] = [:]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        func applies(_ value: JSONValue, _ fragment: JSONValue) throws -> Bool {
            let key = String(decoding: try encoder.encode(fragment), as: UTF8.self)
            let schema: JSONValue
            if let cached = expansions[key] {
                schema = cached
            } else {
                schema = try expanded(fragment)
                expansions[key] = schema
            }
            return (try? OperationSchemaValidator.validate(value, schema: schema)) != nil
        }
        func visit(_ value: JSONValue, _ rule: JSONObject, depth: Int) throws {
            guard depth < 24 else { throw WorkerFailure(code: "invalid_request", message: "File schema nesting exceeded.") }
            if let annotation = rule["x-arsenal-file"]?.object {
                guard let path = value["path"]?.string else { throw WorkerFailure(code: "invalid_request", message: "File input needs a local path object.") }
                let url = try confined(path, roots: [support.appendingPathComponent("inbox"), support.appendingPathComponent("jobs")], code: "invalid_request")
                let mime: String
                do { mime = try media(url).mime } catch { throw WorkerFailure(code: "invalid_request", message: "Input file is not a decodable image or supported file format.") }
                let accept = annotation["accept"]?.array?.compactMap(\.string) ?? []
                guard accept.contains(mime) || accept.contains(String(mime.split(separator: "/").first ?? "") + "/*") else {
                    throw WorkerFailure(code: "invalid_request", message: "Input file MIME \(mime) is not accepted.")
                }
            }
            if let ref = rule["$ref"]?.string {
                var target = root
                for part in ref.dropFirst(2).split(separator: "/") {
                    target = target.object?[String(part).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")] ?? .null
                }
                if let target = target.object { try visit(value, target, depth: depth + 1) }
            }
            for key in ["allOf", "anyOf", "oneOf"] {
                for branch in rule[key]?.array ?? [] {
                    guard let branch = branch.object else { continue }
                    if try key == "allOf" || applies(value, .object(branch)) { try visit(value, branch, depth: depth + 1) }
                }
            }
            if let condition = rule["if"], let branch = rule[try applies(value, condition) ? "then" : "else"]?.object {
                try visit(value, branch, depth: depth + 1)
            }
            if let object = value.object {
                for (key, child) in object {
                    if let childRule = (rule["properties"]?.object?[key] ?? rule["additionalProperties"])?.object { try visit(child, childRule, depth: depth + 1) }
                }
            }
            if let values = value.array, let items = rule["items"]?.object { for child in values { try visit(child, items, depth: depth + 1) } }
        }
        try visit(value, schema, depth: 0)
    }
    public static func output(_ result: JSONObject, operation: ModelOperation, directory: URL?) throws -> JSONObject {
        var payload = result; payload.removeValue(forKey: "id"); payload.removeValue(forKey: "ok")
        if let schema = operation.outputSchema {
            do { try OperationSchemaValidator.validate(.object(payload), schema: .object(schema)) } catch {
                throw WorkerFailure(code: "invalid_output", message: "\(operation.name) output does not match its schema: \(error.localizedDescription)")
            }
        }
        guard ["file", "image", "audio"].contains(operation.output) else { return payload }
        guard let directory, let files = payload["files"]?.array, !files.isEmpty else {
            throw WorkerFailure(code: "invalid_output", message: "File operations must return a nonempty files array in their job directory.")
        }
        var ids = Set<String>(), total = 0
        payload["files"] = .array(
            try files.map { value in
                guard var file = value.object, let id = file["id"]?.string, !id.isEmpty, ids.insert(id).inserted, let path = file["path"]?.string else {
                    throw WorkerFailure(code: "invalid_output", message: "Output file IDs must be present and unique.")
                }
                let url = try confined(path, roots: [directory], code: "invalid_output")
                let count = try FileDigest.bytes(url)
                try admitArtifact(count, total: &total)
                let actual = try media(url)
                guard let declared = file["mime"]?.string, declared == actual.mime || !declared.hasPrefix("image/") && actual.mime == "application/octet-stream" else {
                    throw WorkerFailure(code: "invalid_output", message: "Output MIME does not match its file format.")
                }
                if let width = actual.width, let height = actual.height {
                    guard file["width"]?.number == Double(width), file["height"]?.number == Double(height) else {
                        throw WorkerFailure(code: "invalid_output", message: "Output image dimensions do not match its metadata.")
                    }
                }
                file["bytes"] = .number(Double(count)); file["sha256"] = .string(try FileDigest.sha256(url)); return .object(file)
            })
        return payload
    }
    public static func confined(_ path: String, roots: [URL], code: String) throws -> URL {
        let url = URL(fileURLWithPath: path).standardizedFileURL, resolved = url.resolvingSymlinksInPath()
        // A NUL ends the path for C file calls but not for URL standardization, so the two could name different files.
        guard path.hasPrefix("/"), !path.utf8.contains(0),
            roots.contains(where: { root in
                let original = root.standardizedFileURL.path + "/"
                guard url.path.hasPrefix(original) else { return false }
                let tail = String(url.path.dropFirst(original.count))
                let canonical = root.standardizedFileURL.resolvingSymlinksInPath().appendingPathComponent(tail).standardizedFileURL
                return canonical.path == resolved.path
            }),
            (try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])).map({ $0.isRegularFile == true && $0.isSymbolicLink != true }) == true
        else { throw WorkerFailure(code: code, message: "File must be a regular file contained in this Arsenal inbox/job, without symlinks.") }
        return url
    }
    private static func media(_ url: URL) throws -> (mime: String, width: Int?, height: Int?) {
        if let source = CGImageSourceCreateWithURL(url as CFURL, nil), let type = CGImageSourceGetType(source),
            let mime = UTType(type as String)?.preferredMIMEType,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int
        {
            return (mime, width, height)
        }
        if ["png", "jpg", "jpeg", "gif", "webp"].contains(url.pathExtension.lowercased()) {
            throw WorkerFailure(code: "invalid_output", message: "File is not a decodable image.")
        }
        let mime = [
            "wav": "audio/wav", "flac": "audio/flac", "pcm": "audio/pcm", "mp3": "audio/mpeg", "m4a": "audio/mp4", "aiff": "audio/aiff", "caf": "audio/x-caf",
            "mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm", "pdf": "application/pdf", "glb": "model/gltf-binary",
            "json": "application/json", "txt": "text/plain"
        ]
        return (mime[url.pathExtension.lowercased()] ?? "application/octet-stream", nil, nil)
    }
}
