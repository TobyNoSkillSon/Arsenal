// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalWire
import CryptoKit

struct OpCheck {
    let name: String
    let schema: JSONObject
    let output: String
    let stream: Bool
    let params: JSONObject
    let outputSchema: JSONObject?
    var expected: JSONObject?
    var strict = false
}
struct CheckModel {
    let folder: URL
    let kind: String
    let dtype: String
    let ops: [OpCheck]
    let load: JSONObject
    let cancel: JSONObject?
    let kernelOverride: Bool
    init(folder: URL, kindDir: URL?, support: URL) throws {
        self.folder = folder
        let manifest = try Self.toml(folder.appendingPathComponent("model.toml"))
        guard let kind = manifest["kind"]?.string else { throw CheckFailure("model.toml needs kind") }
        self.kind = kind
        let sourceDtype = manifest["weights"]?["dtype"]?.string ?? "fp16"
        dtype = sourceDtype == "fp32" ? "fp16" : sourceDtype
        let fixtureURL = folder.appendingPathComponent("gate/runner-check.json")
        let fixtures = FileManager.default.fileExists(atPath: fixtureURL.path) ? try WireJSON.decode(Data(contentsOf: fixtureURL)) : [:]
        kernelOverride = fixtures["load"]?["config"]?["kernels"] != nil
        var settings: JSONObject = [:]
        var config: JSONObject = ["kernels": []]
        for declaration in manifest["setting"]?.array ?? [] {
            if declaration["applies"]?.string == "load", let id = declaration["id"]?.string, let value = declaration["default"] {
                settings[id] = value
                if let mapping = declaration["maps_to"]?.string, mapping.hasPrefix("config.") { config[String(mapping.dropFirst(7))] = value }
            }
        }
        config["settings"] = .object(settings)
        var loadParams = fixtures["load"]?.object ?? [:]
        if let overrides = loadParams["config"]?.object { for (key, value) in overrides { config[key] = value } }
        for declaration in manifest["setting"]?.array ?? [] {
            if declaration["applies"]?.string == "load", let id = declaration["id"]?.string,
                let value = config["settings"]?[id], let mapping = declaration["maps_to"]?.string,
                mapping.hasPrefix("config.")
            {
                config[String(mapping.dropFirst(7))] = value
            }
        }
        loadParams["config"] = .object(config)
        load = loadParams
        cancel = fixtures["cancel"]?.object
        var definitions: [String: (JSONObject, URL)] = [:]
        let effectiveKindDir = manifest["kind_file"] != nil ? folder.appendingPathComponent("kinds") : kindDir
        if let kindDir = effectiveKindDir {
            let path = kindDir.appendingPathComponent("\(kind).toml")
            if FileManager.default.fileExists(atPath: path.path) {
                for value in try Self.toml(path)["op"]?.array ?? [] {
                    if let op = value.object, let name = op["name"]?.string { definitions[name] = (op, kindDir.appendingPathComponent(kind)) }
                }
            }
        }
        let declared = manifest["op"]?.array ?? []
        if manifest["op"] == nil {
            definitions = definitions.filter { $0.value.0["default"]?.bool == true }
        } else {
            definitions = definitions.filter { entry in declared.contains { $0["name"]?.string == entry.key } }
        }
        for value in declared {
            if let op = value.object, let name = op["name"]?.string { definitions[name] = (op, folder) }
        }
        guard !definitions.isEmpty else { throw CheckFailure("no declared operations; pass --kind-dir only when op is absent") }
        var checks: [OpCheck] = []
        for name in definitions.keys.sorted() {
            let (op, base) = definitions[name]!
            guard let input = op["input"]?.string else { throw CheckFailure("op \(name) needs input schema") }
            let schema = try WireJSON.decode(OperationSchema.read(path: input, folder: base, kindsDirectory: effectiveKindDir))
            var params = try (fixtures["ops"]?.object ?? fixtures)[name]?.object ?? CheckSchema.synthesize(schema).object ?? [:]
            let fields = schema["properties"]?.object ?? [:]
            if fields["out_dir"] != nil || ["file", "image", "audio"].contains(op["output"]?.string ?? "") {
                let out = support.appendingPathComponent("jobs/\(name)")
                try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
                params["out_dir"] = .string(out.path)
            }
            if op["stream"]?.bool == true && fields["stream"] != nil { params["stream"] = true }
            var validated = params; if fields["out_dir"] == nil { validated["out_dir"] = nil }
            try CheckSchema.validate(.object(validated), schema: schema)
            guard params["id"] == nil && params["op"] == nil else { throw CheckFailure("fixtures contain reserved id/op fields") }
            let outputSchema: JSONObject?
            var expected: JSONObject?
            var strict = false
            if let path = op["output_schema"]?.string {
                outputSchema = try WireJSON.decode(OperationSchema.read(path: path, folder: base, kindsDirectory: effectiveKindDir))
            } else {
                outputSchema = nil
            }
            var check = OpCheck(name: name, schema: schema, output: op["output"]?.string ?? "json", stream: op["stream"]?.bool ?? false, params: params, outputSchema: outputSchema)
            check.expected = fixtures["expected"]?[name]?.object; check.strict = manifest["schema"]?.number == 3
            checks.append(check)
        }
        ops = checks
    }
    static func toml(_ url: URL) throws -> JSONObject {
        var parser = TOMLParser(try String(contentsOf: url, encoding: .utf8))
        return try parser.parse().mapValues(json)
    }
    static func json(_ value: TOMLValue) -> JSONValue {
        switch value {
        case .string(let v): return .string(v)
        case .integer(let v): return .number(Double(v))
        case .decimal(let v): return .number(v)
        case .bool(let v): return .bool(v)
        case .array(let v): return .array(v.map(json))
        case .table(let v): return .object(v.mapValues(json))
        }
    }
    static func findKinds(near folder: URL) -> URL? {
        var parent = folder
        for _ in 0..<8 {
            for path in ["Resources/kinds", "kinds"] {
                let kinds = parent.appendingPathComponent(path)
                if FileManager.default.fileExists(atPath: kinds.path) { return kinds }
            }
            parent.deleteLastPathComponent()
        }
        return nil
    }
}

func validateFiles(_ value: JSONValue, inside root: URL) throws -> Int {
    var found = 0
    if let object = value.object {
        if let path = object["path"]?.string {
            let url = URL(fileURLWithPath: path)
            let resolved = url.resolvingSymlinksInPath()
            let prefix = root.resolvingSymlinksInPath().path + "/"
            guard path.hasPrefix("/"), resolved.path.hasPrefix(prefix), resolved.path == url.standardizedFileURL.path,
                let count = object["bytes"]?.number, count >= 0, count.rounded() == count,
                let hash = object["sha256"]?.string, ModelManifest.isHex(hash, count: 64)
            else {
                throw CheckFailure("unsafe or incomplete file result")
            }
            let attributes = try FileManager.default.attributesOfItem(atPath: resolved.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular,
                (attributes[.size] as? NSNumber)?.doubleValue == count
            else { throw CheckFailure("file size/type mismatch") }
            let handle = try FileHandle(forReadingFrom: resolved)
            defer { try? handle.close() }
            var digest = SHA256()
            while let bytes = try handle.read(upToCount: 65536), !bytes.isEmpty { digest.update(data: bytes) }
            let actual = digest.finalize().map { String(format: "%02x", $0) }.joined()
            guard actual == hash.lowercased() else { throw CheckFailure("file SHA-256 mismatch") }
            found += 1
        }
        for child in object.values { found += try validateFiles(child, inside: root) }
    } else if let array = value.array {
        for child in array { found += try validateFiles(child, inside: root) }
    }
    return found
}

func validateOutput(_ op: OpCheck, reply: JSONObject, events: [WireFrame], root: URL) throws {
    let files: Int
    if op.strict {
        let operation = ModelOperation(name: op.name, title: op.name, inputSchema: op.schema, output: op.output, stream: op.stream, outputSchema: op.outputSchema)
        let directory = op.params["out_dir"]?.string.map { URL(fileURLWithPath: $0) } ?? root
        let result = try OperationArtifacts.output(reply, operation: operation, directory: directory)
        files = result["files"]?.array?.count ?? 0
    } else {
        files = try validateFiles(.object(reply), inside: root)
    }
    if let expected = op.expected {
        if let count = expected["files"]?.number, files != Int(count) { throw CheckFailure("output file count differs from frozen fixture") }
        if let ids = expected["ids"]?.array, reply["files"]?.array?.map({ $0["id"] ?? .null }) != ids { throw CheckFailure("output IDs/order differ from frozen fixture") }
    }
    if let schema = op.outputSchema {
        var payload = reply; payload.removeValue(forKey: "id"); payload.removeValue(forKey: "ok"); if !op.strict { payload.removeValue(forKey: "memory") }
        try CheckSchema.validate(.object(payload), schema: schema)
    }
    var streams: [UInt32: Int] = [:]
    var binaryBytes = 0
    var textEvents = 0
    for frame in events {
        switch frame {
        case .json(let event):
            if event["event"]?.string == "audio_begin" {
                guard let sid = event["stream"]?.number, sid >= 0, sid <= Double(UInt32.max), sid.rounded() == sid,
                    let rate = event["sample_rate"]?.number, rate > 0,
                    let channels = event["channels"]?.number, channels > 0, channels <= 32, channels.rounded() == channels,
                    event["encoding"]?.string == "pcm_s16le", streams[UInt32(sid)] == nil
                else { throw CheckFailure("invalid audio_begin") }
                streams[UInt32(sid)] = Int(channels) * 2
            }
            if event["event"]?.string == "token" {
                guard event["text"]?.string != nil else { throw CheckFailure("token needs text") }; textEvents += 1
            }
        case .binary(let stream, let bytes):
            guard let alignment = streams[stream], !bytes.isEmpty, bytes.count % alignment == 0 else {
                throw CheckFailure("binary chunk has no audio_begin or invalid PCM alignment")
            }
            binaryBytes += bytes.count
        }
    }
    switch op.output {
    case "audio": guard files > 0 || binaryBytes > 0 else { throw CheckFailure("audio op produced no file or PCM chunks") }
    case "image", "file": guard files > 0 else { throw CheckFailure("file/image op produced no verified file") }
    case "text-stream": guard !op.stream || textEvents > 0 else { throw CheckFailure("streaming text op produced no token events") }
    case "json": guard reply.keys.contains(where: { !["id", "ok", "memory"].contains($0) }) else { throw CheckFailure("JSON op produced no result fields") }
    default: throw CheckFailure("unsupported output kind \(op.output)")
    }
}
